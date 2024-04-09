from typing import List, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from . import GaussianRasterizationSettings

import glm
import torch
import ctypes

from glm import mat4

from .console_utils import *
from .sh_utils import eval_sh
from .cuda_utils import CHECK_CUDART_ERROR
from .math_utils import normalize, point_padding
from .gl_utils import load_shader_source, use_gl_program
from .gaussian_utils import build_scaling_rotation, strip_symmetric

import OpenGL.GL as gl
from OpenGL.GL import shaders


class GSplatContextManager:
    """
    This is a high performance rendering backend of the gaussian splatting paper
    We use CUDA to perform primitive sorting and then render the splats using geometry shader that emits quads
    There's some cuda-gl interop involved, thus cuda-python is required, wsl2 out of the question

    EGL should be automatically supported by this and thus it should be able to be used in an offline environment
    """

    def __init__(self,
                 init_buffer_size: int = 32768,
                 init_texture_size: List[int] = [512, 512],
                 dtype: str = torch.float,  # +5-10 fps on 3060, not working too well with flame_salmon
                 tex_dtype: str = torch.half,  # +5-10 fps on 3060, not working too well with flame_salmon
                 ):
        self.attr_sizes = [3, 3, 3, 4]  # verts, cov6, rgba4
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.tex_dtype = getattr(torch, tex_dtype) if isinstance(tex_dtype, str) else tex_dtype
        self.gl_tex_dtype = gl.GL_RGBA16F if self.tex_dtype == torch.half else gl.GL_RGBA32F
        self.gl_attr_dtypes = [gl.GL_FLOAT if self.dtype == torch.float else gl.GL_HALF_FLOAT] * len(self.attr_sizes)
        self.uniforms = dotdict()  # uniform values

        # Perform actual rendering
        try:
            from .egl_utils import egl
            self.offline_rendering = egl.eglGetCurrentContext() != egl.EGL_NO_CONTEXT
        except:
            from .gl_utils import eglctx
            self.offline_rendering = eglctx is not None

        self.compile_shaders()
        self.use_gl_program(self.gsplat_program)
        self.opengl_options()
        self.resize_buffers(init_buffer_size)
        self.resize_textures(*init_texture_size)

    def opengl_options(self):
        # Performs face culling
        gl.glDisable(gl.GL_CULL_FACE)

        # Performs z-buffer testing
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_ALPHA_TEST)

        # Enable some masking tests
        gl.glEnable(gl.GL_SCISSOR_TEST)

        # Enable blending for final rendering
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def compile_shaders(self):
        try:
            self.gsplat_program = shaders.compileProgram(
                shaders.compileShader(load_shader_source('gsplat.vert'), gl.GL_VERTEX_SHADER),
                shaders.compileShader(load_shader_source('gsplat.geom'), gl.GL_GEOMETRY_SHADER),
                shaders.compileShader(load_shader_source('gsplat.frag'), gl.GL_FRAGMENT_SHADER)
            )
        except Exception as e:
            print(str(e).encode('utf-8').decode('unicode_escape'))
            raise e

    def use_gl_program(self, program: shaders.ShaderProgram):
        use_gl_program(program)
        self.uniforms.P = gl.glGetUniformLocation(program, "P")
        self.uniforms.VM = gl.glGetUniformLocation(program, "VM")
        self.uniforms.focal = gl.glGetUniformLocation(program, "focal")
        self.uniforms.basisViewport = gl.glGetUniformLocation(program, "basisViewport")

    def upload_gl_uniforms(self, raster_settings: 'GaussianRasterizationSettings'):
        # FIXME: Possible nasty synchronization issue: raster_settings might contain cuda tensors
        # TODO: Add a warning here to make the user pass in cpu tensors to avoid explicit synchronization

        P = raster_settings.projmatrix.detach().cpu().numpy()  # 4,4 # MARK: possible sync
        P = mat4(*P.tolist())
        V = raster_settings.viewmatrix.detach().cpu().numpy()  # 4,4 # MARK: possible sync
        V = mat4(*V.tolist())

        if not self.offline_rendering:
            K = P * glm.affineInverse(V)

            gl_c2w = glm.affineInverse(V)
            gl_c2w[0] *= 1  # do notflip x
            gl_c2w[1] *= -1  # flip y
            gl_c2w[2] *= -1  # flip z
            V = glm.affineInverse(gl_c2w)

            K[2][0] *= -1
            K[2][2] *= -1
            K[2][3] *= -1

            P = K * V

        M = glm.identity(mat4)
        VM = V * M

        gl.glUniformMatrix4fv(self.uniforms.P, 1, gl.GL_FALSE, glm.value_ptr(P))  # o2c
        gl.glUniformMatrix4fv(self.uniforms.VM, 1, gl.GL_FALSE, glm.value_ptr(VM))  # o2w
        gl.glUniform2f(self.uniforms.focal, 0.5 * raster_settings.image_width / raster_settings.tanfovx, 0.5 * raster_settings.image_height / raster_settings.tanfovy)  # focal in pixel space
        gl.glUniform2f(self.uniforms.basisViewport, 1 / raster_settings.image_width, 1 / raster_settings.image_height)  # focal

    def init_gl_buffers(self, v: int = 0):
        from cuda import cudart
        if hasattr(self, 'cu_vbo'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_vbo))

        # This will only init the corresponding buffer object
        element_size = 4 if self.gl_attr_dtypes[0] == gl.GL_FLOAT else 2
        attr_size = sum(self.attr_sizes)
        n_verts_bytes = v * attr_size * element_size

        # Housekeeping
        if hasattr(self, 'vao'):
            gl.glDeleteVertexArrays(1, [self.vao])
            gl.glDeleteBuffers(2, [self.vbo, self.ebo])

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        self.ebo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, n_verts_bytes, ctypes.c_void_p(0), gl.GL_DYNAMIC_DRAW)  # NOTE: Using pointers here won't work

        # https://stackoverflow.com/questions/67195932/pyopengl-cannot-render-any-vao
        cumsum = 0
        for i, (s, t) in enumerate(zip(self.attr_sizes, self.gl_attr_dtypes)):
            gl.glVertexAttribPointer(i, s, t, gl.GL_FALSE, attr_size * element_size, ctypes.c_void_p(cumsum * element_size))  # we use 32 bit float
            gl.glEnableVertexAttribArray(i)
            cumsum += s
        # Register vertex buffer obejct
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard
        try:
            self.cu_vbo = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterBuffer(self.vbo, flags))
        except RuntimeError as e:
            log(red(f'Your system does not support CUDA-GL interop, will use pytorch3d\'s implementation instead'))
            log(red(f'This can be done by specifying {blue("model_cfg.sampler_cfg.use_cudagl=False model_cfg.sampler_cfg.use_diffgl=False")} at the end of your command'))
            log(red(f'Note that this implementation is extremely slow, we recommend running on a native system that support the interop'))
            log(red(f'An alternative is to install diff_point_rasterization and use the approximated tile-based rasterization, enabled by the `render_gs` switch'))
            # raise RuntimeError(str(e) + ": This unrecoverable, please read the error message above")
            raise e

    def init_textures(self, H: int, W: int):
        from cuda import cudart
        if hasattr(self, 'cu_tex'):
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnregisterResource(self.cu_tex))

        if hasattr(self, 'fbo'):
            gl.glDeleteFramebuffers(1, [self.fbo])
            gl.glDeleteTextures(1, [self.tex_rgba, self.tex_atth])

        # Prepare for write frame buffers
        self.tex_rgba = gl.glGenTextures(1)
        self.tex_atth = gl.glGenTextures(1)
        self.fbo = gl.glGenFramebuffers(1)  # generate 1 framebuffer, storereference in fb

        # Init the texture (call the resizing function), will simply allocate empty memory
        # The internal format describes how the texture shall be stored in the GPU. The format describes how the format of your pixel data in client memory (together with the type parameter).
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_rgba)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.gl_tex_dtype, W, H, 0, gl.GL_RGBA, gl.GL_FLOAT, ctypes.c_void_p(0))  # 16 * 4
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.tex_atth)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24, W, H, 0, gl.GL_DEPTH_COMPONENT, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0))  # 32
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)

        # Bind texture to fbo
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.tex_rgba, 0)  # location 0
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.tex_atth, 0)
        gl.glDrawBuffers(1, [gl.GL_COLOR_ATTACHMENT0])

        # Check framebuffer status
        if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE:
            log(red('Framebuffer not complete, exiting...'))
            raise RuntimeError('Incomplete framebuffer')

        # Restore the original state
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Register image to read from
        flags = cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsReadOnly
        self.cu_tex = CHECK_CUDART_ERROR(cudart.cudaGraphicsGLRegisterImage(self.tex_rgba, gl.GL_TEXTURE_2D, flags))

    def resize_textures(self, H: int, W: int):  # analogy to update_gl_buffers
        if not hasattr(self, 'max_H'): self.max_H = 0
        if not hasattr(self, 'max_W'): self.max_W = 0
        if H > self.max_H or W > self.max_W:  # max got updated
            if H > self.max_H: self.max_H = int(H * 1.05)
            if W > self.max_W: self.max_W = int(W * 1.05)
            self.init_textures(self.max_H, self.max_W)

    def resize_buffers(self, v: int = 0):
        if not hasattr(self, 'max_verts'): self.max_verts = 0
        if v > self.max_verts:
            if v > self.max_verts: self.max_verts = int(v * 1.05)
            self.init_gl_buffers(self.max_verts)

    @torch.no_grad()
    def render(self, xyz3: torch.Tensor, cov6: torch.Tensor, rgb3: torch.Tensor, occ1: torch.Tensor, raster_settings: 'GaussianRasterizationSettings'):
        H, W = raster_settings.image_height, raster_settings.image_width
        self.resize_textures(H, W)
        self.resize_buffers(len(xyz3))

        # Compute GS
        data = torch.cat([xyz3, cov6, rgb3, occ1], dim=-1)

        # Sort by view space depth
        view = point_padding(xyz3) @ raster_settings.viewmatrix.to('cuda', non_blocking=True)
        idx = view[..., 2].argsort(descending=True)  # S, sorted
        data = data[idx].ravel().contiguous()  # sorted data on gpu

        # Upload sorted data to OpenGL for rendering
        from cuda import cudart
        from easyvolcap.utils.cuda_utils import CHECK_CUDART_ERROR, FORMAT_CUDART_ERROR
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice

        CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))
        cu_vbo_ptr, cu_vbo_size = CHECK_CUDART_ERROR(cudart.cudaGraphicsResourceGetMappedPointer(self.cu_vbo))

        CHECK_CUDART_ERROR(cudart.cudaMemcpyAsync(cu_vbo_ptr,
                                                  data.data_ptr(),
                                                  data.numel() * data.element_size(),
                                                  kind,
                                                  torch.cuda.current_stream().cuda_stream))
        CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, self.cu_vbo, torch.cuda.current_stream().cuda_stream))

        if self.offline_rendering:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)  # for offscreen rendering to textures
            gl.glClearColor(0, 0, 0, 1.)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.use_gl_program(self.gsplat_program)
        self.upload_gl_uniforms(raster_settings)
        x, y, w, h = gl.glGetIntegerv(gl.GL_VIEWPORT)
        gl.glViewport(0, 0, W, H)
        gl.glScissor(0, 0, W, H)  # only render in this small region of the viewport
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_POINTS, 0, len(idx))  # number of vertices
        gl.glBindVertexArray(0)
        gl.glViewport(x, y, w, h)
        gl.glScissor(x, y, w, h)

        # Prepare the output
        if self.offline_rendering:
            rgba_map = torch.empty((H, W, 4), dtype=self.tex_dtype, device='cuda')  # to hold the data from opengl
            # Texture access and copy could be very slow...
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

            # Copy rendered image and depth back as tensor
            cu_tex = self.cu_tex

            # The resources in resources may be accessed by CUDA until they are unmapped.
            # The graphics API from which resources were registered should not access any resources while they are mapped by CUDA.
            # If an application does so, the results are undefined.
            CHECK_CUDART_ERROR(cudart.cudaGraphicsMapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))
            cu_tex_arr = CHECK_CUDART_ERROR(cudart.cudaGraphicsSubResourceGetMappedArray(cu_tex, 0, 0))
            CHECK_CUDART_ERROR(cudart.cudaMemcpy2DFromArrayAsync(rgba_map.data_ptr(),  # dst
                                                                 W * 4 * rgba_map.element_size(),  # dpitch
                                                                 cu_tex_arr,  # src
                                                                 0,  # wOffset
                                                                 0,  # hOffset
                                                                 W * 4 * rgba_map.element_size(),  # width Width of matrix transfer (columns in bytes)
                                                                 H,  # height
                                                                 kind,  # kind
                                                                 torch.cuda.current_stream().cuda_stream))  # stream
            CHECK_CUDART_ERROR(cudart.cudaGraphicsUnmapResources(1, cu_tex, torch.cuda.current_stream().cuda_stream))

            return rgba_map  # H, W, 4
        else:
            return None

    def rasterize_gaussians(
        self,
        means3D: torch.Tensor,  # N, 3
        means2D: torch.Tensor,  # N, 2
        shs: torch.Tensor,  # N, SH, 3
        colors_precomp: torch.Tensor,  # N, 3
        opacities: torch.Tensor,  # N, 1
        scales: torch.Tensor,  # N, 3
        rotations: torch.Tensor,  # N, 4
        cov3D_precomp: torch.Tensor,  # N, 6
        raster_settings: 'GaussianRasterizationSettings',
    ):
        if cov3D_precomp is None:
            L = build_scaling_rotation(scales, rotations)
            cov3D_precomp = strip_symmetric(L @ L.mT)  # N, 6

        if colors_precomp is None:
            C = raster_settings.campos.to('cuda', non_blocking=True)  # 3,
            dirs = normalize(means3D - C)
            colors_precomp = eval_sh(raster_settings.sh_degree, shs.mT, dirs)
            colors_precomp = (colors_precomp + 0.5).clip(0, 1)

        rgba_map = self.render(means3D, cov3D_precomp, colors_precomp, opacities, raster_settings)

        if self.offline_rendering:
            image, alpha = rgba_map.split([3, 1], dim=-1)
            image, alpha = image.permute(2, 0, 1), alpha.permute(2, 0, 1)

            # FIXME: Alpha channel seems to be bugged
            return image.float(), torch.ones_like(alpha).float()
        else:
            return None, None
