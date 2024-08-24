import torch
from torch import nn
from typing import NamedTuple, List, Dict

from .gsplat_utils import GSplatContextManager

raster_context = None


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    campos: torch.Tensor
    sh_degree: int = 3
    prefiltered: bool = True
    debug: bool = False
    use_depth: bool = False
    use_normal: bool = False
    solid_mode: bool = False


class GaussianRasterizer:
    def __init__(self,
                 raster_settings: GaussianRasterizationSettings,
                 init_buffer_size: int = 32768,
                 init_texture_size: List[int] = [512, 512],
                 dtype=torch.float,
                 tex_dtype=torch.uint8):
        super().__init__()
        self.raster_settings = raster_settings

        global raster_context
        dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        tex_dtype = getattr(torch, tex_dtype) if isinstance(tex_dtype, str) else tex_dtype
        if raster_context is None or raster_context.dtype != dtype or raster_context.tex_dtype != tex_dtype:
            raster_context = GSplatContextManager(init_buffer_size=init_buffer_size, init_texture_size=init_texture_size, dtype=dtype, tex_dtype=tex_dtype)  # only created once

    def __call__(self,
                 means3D: torch.Tensor,
                 means2D: torch.Tensor,  # only to match the api, can be none, not used
                 opacities: torch.Tensor,
                 shs: torch.Tensor = None,
                 colors_precomp: torch.Tensor = None,
                 scales: torch.Tensor = None,
                 rotations: torch.Tensor = None,
                 cov3D_precomp: torch.Tensor = None
                 ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        # Invoke CUDA-GL rasterization routine
        return raster_context.rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )  # will output 3 channel rgb + 1 channel alpha, instead of radii
