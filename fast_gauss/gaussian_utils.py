import torch
import numpy as np
import torch.nn.functional as F
from .math_utils import point_padding


@torch.jit.script
def in_frustum(xyz: torch.Tensor, full_proj_matrix: torch.Tensor, xy_padding: float = 0.5, padding: float = 0.01):
    ndc = point_padding(xyz) @ full_proj_matrix  # this is now in clip space
    ndc = ndc[..., :3] / ndc[..., 3:]
    return (ndc[..., 2] > -1 - padding) & (ndc[..., 2] < 1 + padding) & (ndc[..., 0] > -1 - xy_padding) & (ndc[..., 0] < 1. + xy_padding) & (ndc[..., 1] > -1 - xy_padding) & (ndc[..., 1] < 1. + xy_padding)  # N,


@torch.jit.script
def rgb2sh0(rgb: torch.Tensor):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


@torch.jit.script
def sh02rgb(sh: torch.Tensor):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@torch.jit.script
def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


@torch.jit.script
def strip_lowerdiag(L: torch.Tensor):
    # uncertainty = torch.zeros((L.shape[0], 6), dtype=L.dtype, device=L.device)

    # uncertainty[:, 0] = L[:, 0, 0].clip(0.0)  # sanitize covariance matrix
    # uncertainty[:, 1] = L[:, 0, 1]
    # uncertainty[:, 2] = L[:, 0, 2]
    # uncertainty[:, 3] = L[:, 1, 1].clip(0.0)  # sanitize covariance matrix
    # uncertainty[:, 4] = L[:, 1, 2]
    # uncertainty[:, 5] = L[:, 2, 2].clip(0.0)  # sanitize covariance matrix
    # return uncertainty

    inds = torch.triu_indices(3, 3, device=L.device)  # 2, 6
    return L[:, inds[0], inds[1]]


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


@torch.jit.script
def build_rotation(q: torch.Tensor):
    assert q.shape[-1] == 4
    # norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    # q = r / norm[:, None]
    q = F.normalize(q, dim=-1)

    R = torch.zeros((q.size(0), 3, 3), dtype=q.dtype, device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@torch.jit.script
def build_scaling_rotation(s: torch.Tensor, q: torch.Tensor):
    L = torch.zeros((s.shape[0], 3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(q)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


@torch.jit.script
def build_cov6(s: torch.Tensor, q: torch.Tensor):
    L = build_scaling_rotation(s, q)
    return strip_lowerdiag(L @ L.mT)


def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))
