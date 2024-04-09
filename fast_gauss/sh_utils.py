import torch


@torch.jit.script
def eval_shfs_4d_00(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814

    l0m0 = C0
    result = l0m0 * sh[..., 0]
    return result


@torch.jit.script
def eval_shfs_4d_10(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199

    l0m0 = C0
    result = l0m0 * sh[..., 0]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    l1m1 = -1 * C1 * y
    l1m0 = C1 * z
    l1p1 = -1 * C1 * x

    result = (result +
              l1m1 * sh[..., 1] +
              l1m0 * sh[..., 2] +
              l1p1 * sh[..., 3])
    return result


@torch.jit.script
def eval_shfs_4d_20(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]

    l0m0 = C0
    result = l0m0 * sh[..., 0]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    l1m1 = -1 * C1 * y
    l1m0 = C1 * z
    l1p1 = -1 * C1 * x

    result = (result +
              l1m1 * sh[..., 1] +
              l1m0 * sh[..., 2] +
              l1p1 * sh[..., 3])

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    l2m2 = C2[0] * xy
    l2m1 = C2[1] * yz
    l2m0 = C2[2] * (2.0 * zz - xx - yy)
    l2p1 = C2[3] * xz
    l2p2 = C2[4] * (xx - yy)

    result = (result +
              l2m2 * sh[..., 4] +
              l2m1 * sh[..., 5] +
              l2m0 * sh[..., 6] +
              l2p1 * sh[..., 7] +
              l2p2 * sh[..., 8])

    return result


@torch.jit.script
def eval_shfs_4d_30(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]
    C4 = [
        2.5033429417967046,
        -1.7701307697799304,
        0.9461746957575601,
        -0.6690465435572892,
        0.10578554691520431,
        -0.6690465435572892,
        0.47308734787878004,
        -1.7701307697799304,
        0.6258357354491761,
    ]

    l0m0 = C0
    result = l0m0 * sh[..., 0]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    l1m1 = -1 * C1 * y
    l1m0 = C1 * z
    l1p1 = -1 * C1 * x

    result = (result +
              l1m1 * sh[..., 1] +
              l1m0 * sh[..., 2] +
              l1p1 * sh[..., 3])

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    l2m2 = C2[0] * xy
    l2m1 = C2[1] * yz
    l2m0 = C2[2] * (2.0 * zz - xx - yy)
    l2p1 = C2[3] * xz
    l2p2 = C2[4] * (xx - yy)

    result = (result +
              l2m2 * sh[..., 4] +
              l2m1 * sh[..., 5] +
              l2m0 * sh[..., 6] +
              l2p1 * sh[..., 7] +
              l2p2 * sh[..., 8])

    l3m3 = C3[0] * y * (3 * xx - yy)
    l3m2 = C3[1] * xy * z
    l3m1 = C3[2] * y * (4 * zz - xx - yy)
    l3m0 = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    l3p1 = C3[4] * x * (4 * zz - xx - yy)
    l3p2 = C3[5] * z * (xx - yy)
    l3p3 = C3[6] * x * (xx - 3 * yy)

    result = (result +
              l3m3 * sh[..., 9] +
              l3m2 * sh[..., 10] +
              l3m1 * sh[..., 11] +
              l3m0 * sh[..., 12] +
              l3p1 * sh[..., 13] +
              l3p2 * sh[..., 14] +
              l3p3 * sh[..., 15])
    return result


@torch.jit.script
def eval_shfs_4d_31(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]

    l0m0 = C0
    result = l0m0 * sh[..., 0]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    l1m1 = -1 * C1 * y
    l1m0 = C1 * z
    l1p1 = -1 * C1 * x

    result = (result +
              l1m1 * sh[..., 1] +
              l1m0 * sh[..., 2] +
              l1p1 * sh[..., 3])

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    l2m2 = C2[0] * xy
    l2m1 = C2[1] * yz
    l2m0 = C2[2] * (2.0 * zz - xx - yy)
    l2p1 = C2[3] * xz
    l2p2 = C2[4] * (xx - yy)

    result = (result +
              l2m2 * sh[..., 4] +
              l2m1 * sh[..., 5] +
              l2m0 * sh[..., 6] +
              l2p1 * sh[..., 7] +
              l2p2 * sh[..., 8])

    l3m3 = C3[0] * y * (3 * xx - yy)
    l3m2 = C3[1] * xy * z
    l3m1 = C3[2] * y * (4 * zz - xx - yy)
    l3m0 = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    l3p1 = C3[4] * x * (4 * zz - xx - yy)
    l3p2 = C3[5] * z * (xx - yy)
    l3p3 = C3[6] * x * (xx - 3 * yy)

    result = (result +
              l3m3 * sh[..., 9] +
              l3m2 * sh[..., 10] +
              l3m1 * sh[..., 11] +
              l3m0 * sh[..., 12] +
              l3p1 * sh[..., 13] +
              l3p2 * sh[..., 14] +
              l3p3 * sh[..., 15])

    t1 = torch.cos(2 * torch.pi * dirs_t / l)

    result = (result +
              t1 * l0m0 * sh[..., 16] +
              t1 * l1m1 * sh[..., 17] +
              t1 * l1m0 * sh[..., 18] +
              t1 * l1p1 * sh[..., 19] +
              t1 * l2m2 * sh[..., 20] +
              t1 * l2m1 * sh[..., 21] +
              t1 * l2m0 * sh[..., 22] +
              t1 * l2p1 * sh[..., 23] +
              t1 * l2p2 * sh[..., 24] +
              t1 * l3m3 * sh[..., 25] +
              t1 * l3m2 * sh[..., 26] +
              t1 * l3m1 * sh[..., 27] +
              t1 * l3m0 * sh[..., 28] +
              t1 * l3p1 * sh[..., 29] +
              t1 * l3p2 * sh[..., 30] +
              t1 * l3p3 * sh[..., 31])

    return result


@torch.jit.script
def eval_shfs_4d_32(sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    C0 = 0.28209479177387814
    C1 = 0.4886025119029199
    C2 = [
        1.0925484305920792,
        -1.0925484305920792,
        0.31539156525252005,
        -1.0925484305920792,
        0.5462742152960396
    ]
    C3 = [
        -0.5900435899266435,
        2.890611442640554,
        -0.4570457994644658,
        0.3731763325901154,
        -0.4570457994644658,
        1.445305721320277,
        -0.5900435899266435
    ]

    l0m0 = C0
    result = l0m0 * sh[..., 0]

    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    l1m1 = -1 * C1 * y
    l1m0 = C1 * z
    l1p1 = -1 * C1 * x

    result = (result +
              l1m1 * sh[..., 1] +
              l1m0 * sh[..., 2] +
              l1p1 * sh[..., 3])

    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    l2m2 = C2[0] * xy
    l2m1 = C2[1] * yz
    l2m0 = C2[2] * (2.0 * zz - xx - yy)
    l2p1 = C2[3] * xz
    l2p2 = C2[4] * (xx - yy)

    result = (result +
              l2m2 * sh[..., 4] +
              l2m1 * sh[..., 5] +
              l2m0 * sh[..., 6] +
              l2p1 * sh[..., 7] +
              l2p2 * sh[..., 8])

    l3m3 = C3[0] * y * (3 * xx - yy)
    l3m2 = C3[1] * xy * z
    l3m1 = C3[2] * y * (4 * zz - xx - yy)
    l3m0 = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    l3p1 = C3[4] * x * (4 * zz - xx - yy)
    l3p2 = C3[5] * z * (xx - yy)
    l3p3 = C3[6] * x * (xx - 3 * yy)

    result = (result +
              l3m3 * sh[..., 9] +
              l3m2 * sh[..., 10] +
              l3m1 * sh[..., 11] +
              l3m0 * sh[..., 12] +
              l3p1 * sh[..., 13] +
              l3p2 * sh[..., 14] +
              l3p3 * sh[..., 15])

    t1 = torch.cos(2 * torch.pi * dirs_t / l)

    result = (result +
              t1 * l0m0 * sh[..., 16] +
              t1 * l1m1 * sh[..., 17] +
              t1 * l1m0 * sh[..., 18] +
              t1 * l1p1 * sh[..., 19] +
              t1 * l2m2 * sh[..., 20] +
              t1 * l2m1 * sh[..., 21] +
              t1 * l2m0 * sh[..., 22] +
              t1 * l2p1 * sh[..., 23] +
              t1 * l2p2 * sh[..., 24] +
              t1 * l3m3 * sh[..., 25] +
              t1 * l3m2 * sh[..., 26] +
              t1 * l3m1 * sh[..., 27] +
              t1 * l3m0 * sh[..., 28] +
              t1 * l3p1 * sh[..., 29] +
              t1 * l3p2 * sh[..., 30] +
              t1 * l3p3 * sh[..., 31])

    t2 = torch.cos(2 * torch.pi * 2 * dirs_t / l)

    result = (result +
              t2 * l0m0 * sh[..., 32] +
              t2 * l1m1 * sh[..., 33] +
              t2 * l1m0 * sh[..., 34] +
              t2 * l1p1 * sh[..., 35] +
              t2 * l2m2 * sh[..., 36] +
              t2 * l2m1 * sh[..., 37] +
              t2 * l2m0 * sh[..., 38] +
              t2 * l2p1 * sh[..., 39] +
              t2 * l2p2 * sh[..., 40] +
              t2 * l3m3 * sh[..., 41] +
              t2 * l3m2 * sh[..., 42] +
              t2 * l3m1 * sh[..., 43] +
              t2 * l3m0 * sh[..., 44] +
              t2 * l3p1 * sh[..., 45] +
              t2 * l3p2 * sh[..., 46] +
              t2 * l3p3 * sh[..., 47])

    return result


def eval_shfs_4d(deg: int, deg_t: int, sh: torch.Tensor, dirs: torch.Tensor, dirs_t: torch.Tensor, l: torch.Tensor):
    # fmt: off
    if deg <= 0:                  return eval_shfs_4d_00(sh, dirs, dirs_t, l)
    elif deg <= 1:                return eval_shfs_4d_10(sh, dirs, dirs_t, l)
    elif deg <= 2:                return eval_shfs_4d_20(sh, dirs, dirs_t, l)
    elif deg <= 3 and deg_t <= 0: return eval_shfs_4d_30(sh, dirs, dirs_t, l)
    elif deg <= 3 and deg_t <= 1: return eval_shfs_4d_31(sh, dirs, dirs_t, l)
    elif deg <= 3 and deg_t <= 2: return eval_shfs_4d_32(sh, dirs, dirs_t, l)
    else: raise NotImplementedError('Unsupported 4DSH dimension')
    # fmt: on


def eval_sh(deg: int, sh: torch.Tensor, dirs: torch.Tensor):
    # fmt: off
    if deg <= 0:                  return eval_shfs_4d_00(sh, dirs, torch.as_tensor([]), torch.as_tensor([]))
    elif deg <= 1:                return eval_shfs_4d_10(sh, dirs, torch.as_tensor([]), torch.as_tensor([]))
    elif deg <= 2:                return eval_shfs_4d_20(sh, dirs, torch.as_tensor([]), torch.as_tensor([]))
    else: raise NotImplementedError('Unsupported SH dimension')
    # fmt: on