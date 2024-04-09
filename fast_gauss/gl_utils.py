import os
import sys
from typing import Union

from .console_utils import *


# fmt: off
# Environment variable messaging
# Need to export EGL_DEVICE_ID before trying to import egl
# And we need to consider the case when we're performing distributed training
# from easyvolcap.engine import cfg, args
if 'OpenGL' not in sys.modules:
    try:
        from .egl_utils import create_opengl_context, eglContextManager
        eglctx = eglContextManager()
    except Exception as e:
        log(yellow(f'Could not import EGL related modules. {type(e).__name__}: {e}'))
        os.environ['PYOPENGL_PLATFORM'] = ''
        eglctx = None
else:
    eglctx = None

def is_wsl2():
    """Returns True if the current environment is WSL2, False otherwise."""
    return exists("/etc/wsl.conf") and os.environ.get("WSL_DISTRO_NAME")

if is_wsl2():
    os.environ['PYOPENGL_PLATFORM'] = 'glx'

import OpenGL.GL as gl

try:
    from OpenGL.GL import shaders
except Exception as e:
    print(f'WARNING: OpenGL shaders import error encountered, please install the latest PyOpenGL from github using:')
    print(f'pip install git+https://github.com/mcfletch/pyopengl')
    raise e
# fmt: on


def common_opengl_options():
    # Use program point size
    gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)

    # Performs face culling
    gl.glEnable(gl.GL_CULL_FACE)
    gl.glCullFace(gl.GL_BACK)

    # Performs alpha trans testing
    # gl.glEnable(gl.GL_ALPHA_TEST)
    try: gl.glEnable(gl.GL_ALPHA_TEST)
    except gl.GLError as e: pass

    # Performs z-buffer testing
    gl.glEnable(gl.GL_DEPTH_TEST)
    # gl.glDepthMask(gl.GL_TRUE)
    gl.glDepthFunc(gl.GL_LEQUAL)
    # gl.glDepthRange(-1.0, 1.0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Enable some masking tests
    gl.glEnable(gl.GL_SCISSOR_TEST)

    # Enable this to correctly render points
    # https://community.khronos.org/t/gl-point-sprite-gone-in-3-2/59310
    # gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    try: gl.glEnable(gl.GL_POINT_SPRITE)  # MARK: ONLY SPRITE IS WORKING FOR NOW
    except gl.GLError as e: pass
    # gl.glEnable(gl.GL_POINT_SMOOTH) # MARK: ONLY SPRITE IS WORKING FOR NOW

    # # Configure how we store the pixels in memory for our subsequent reading of the FBO to store the rendering into memory.
    # # The second argument specifies that our pixels will be in bytes.
    # gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)


def load_shader_source(file: str = 'splat.frag'):
    # Ideally we can just specify the shader name instead of an variable
    if not exists(file):
        file = f'{dirname(__file__)}/shaders/{file}'
    if not exists(file):
        file = file.replace('shaders/', '')
    if not exists(file):
        raise RuntimeError(f'Shader file: {file} does not exist')
    with open(file, 'r') as f:
        return f.read()


def use_gl_program(program: Union[shaders.ShaderProgram, dict]):
    if isinstance(program, dict):
        # Recompile the program if the user supplied sources
        program = dotdict(program)
        program = shaders.compileProgram(
            shaders.compileShader(program.VERT_SHADER_SRC, gl.GL_VERTEX_SHADER),
            shaders.compileShader(program.FRAG_SHADER_SRC, gl.GL_FRAGMENT_SHADER)
        )
    return gl.glUseProgram(program)
