import torch
import struct
from .console_utils import *


def to_cuda(batch, device="cuda", ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)):
        batch = [to_cuda(b, device, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: (to_cuda(v, device, ignore_list) if k != "meta" else v) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(device, non_blocking=True)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device=device)
    return batch


def to_x_if(batch, x: str, cond):
    if isinstance(batch, (tuple, list)):
        batch = [to_x(b, x) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_x(v, x) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        if cond(x):
            batch = batch.to(x, non_blocking=True)
    elif isinstance(batch, np.ndarray):  # numpy and others
        if cond(x):
            batch = torch.as_tensor(batch).to(x, non_blocking=True)
    else:
        pass  # do nothing here, used for typed in to_x for methods
        # FIXME: Incosistent behavior here, might lead to undebuggable bugs
    return batch


def to_x(batch, x: str) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)):
        batch = [to_x(b, x) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_x(v, x) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.to(x, non_blocking=True)
    elif isinstance(batch, np.ndarray):  # numpy and others
        batch = torch.as_tensor(batch).to(x, non_blocking=True)
    else:
        pass  # do nothing here, used for typed in to_x for methods
        # FIXME: Incosistent behavior here, might lead to undebuggable bugs
    return batch


def to_tensor(batch, ignore_list: bool = False) -> Union[torch.Tensor, dotdict[str, torch.Tensor]]:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_tensor(b, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_tensor(v, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        pass
    else:  # numpy and others
        batch = torch.as_tensor(batch)
    return batch


def to_list(batch, non_blocking=False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)):
        batch = [to_list(b, non_blocking) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_list(v, non_blocking) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy().tolist()
    elif isinstance(batch, torch.Tensor):
        batch = batch.tolist()
    else:  # others, keep as is
        pass
    return batch


def to_cpu(batch, non_blocking=False, ignore_list: bool = False) -> torch.Tensor:
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_cpu(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_cpu(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking)
    else:  # numpy and others
        batch = torch.as_tensor(batch, device="cpu")
    return batch


def to_numpy(batch, non_blocking=False, ignore_list: bool = False) -> Union[List, Dict, np.ndarray]:  # almost always exporting, should block
    if isinstance(batch, (tuple, list)) and not ignore_list:
        batch = [to_numpy(b, non_blocking, ignore_list) for b in batch]
    elif isinstance(batch, dict):
        batch = dotdict({k: to_numpy(v, non_blocking, ignore_list) for k, v in batch.items()})
    elif isinstance(batch, torch.Tensor):
        batch = batch.detach().to('cpu', non_blocking=non_blocking).numpy()
    else:  # numpy and others
        batch = np.asarray(batch)
    return batch


def get_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply"):
    from trimesh import Trimesh
    from trimesh.visual import TextureVisuals
    from trimesh.visual.material import PBRMaterial, SimpleMaterial
    from easyvolcap.utils.mesh_utils import face_normals, loop_subdivision

    verts, faces = to_numpy([verts, faces])
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    # MARK: used process=False here to preserve vertex order
    mesh = Trimesh(verts, faces, process=False)
    if colors is None:
        # colors = verts
        colors = face_normals(torch.from_numpy(verts), torch.from_numpy(faces).long()) * 0.5 + 0.5
    colors = to_numpy(colors)
    colors = colors.reshape(-1, 3)
    colors = (np.concatenate([colors, np.ones([*colors.shape[:-1], 1])], axis=-1) * 255).astype(np.uint8)
    if len(verts) == len(colors):
        mesh.visual.vertex_colors = colors
    elif len(faces) == len(colors):
        mesh.visual.face_colors = colors

    if normals is not None:
        normals = to_numpy(normals)
        mesh.vertex_normals = normals

    if uv is not None:
        from PIL import Image
        uv = to_numpy(uv)
        uv = uv.reshape(-1, 2)
        img = to_numpy(img)
        img = img.reshape(*img.shape[-3:])
        img = Image.fromarray(np.uint8(img * 255))
        mat = SimpleMaterial(
            image=img,
            diffuse=(0.8, 0.8, 0.8),
            ambient=(1.0, 1.0, 1.0),
        )
        mat.name = os.path.splitext(os.path.split(filename)[1])[0]
        texture = TextureVisuals(uv=uv, material=mat)
        mesh.visual = texture

    return mesh


def get_tensor_mesh_data(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None):

    # pytorch3d wants a tensor
    verts, faces, uv, img, uvfaces = to_tensor([verts, faces, uv, img, uvfaces])
    verts = verts.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    uv = uv.reshape(-1, 2)
    img = img.reshape(img.shape[-3:])
    uvfaces = uvfaces.reshape(-1, 3)

    # textures = TexturesUV(img, uvfaces, uv)
    # meshes = Meshes(verts, faces, textures)
    return verts, faces, uv, img, uvfaces


def export_npz(batch: dotdict, filename: struct):
    export_dotdict(batch, filename)


def export_dotdict(batch: dotdict, filename: struct):
    batch = to_numpy(batch)
    np.savez_compressed(filename, **batch)


def load_mesh(filename: str, device='cuda', load_uv=False, load_aux=False, backend='pytorch3d'):
    from pytorch3d.io import load_ply, load_obj
    if backend == 'trimesh':
        import trimesh
        mesh: trimesh.Trimesh = trimesh.load(filename)
        return mesh.vertices, mesh.faces

    vm, fm = None, None
    if filename.endswith('.npz'):
        mesh = np.load(filename)
        v = torch.from_numpy(mesh['verts'])
        f = torch.from_numpy(mesh['faces'])

        if load_uv:
            vm = torch.from_numpy(mesh['uvs'])
            fm = torch.from_numpy(mesh['uvfaces'])
    else:
        if filename.endswith('.ply'):
            v, f = load_ply(filename)
        elif filename.endswith('.obj'):
            v, faces_attr, aux = load_obj(filename)
            f = faces_attr.verts_idx

            if load_uv:
                vm = aux.verts_uvs
                fm = faces_attr.textures_idx
        else:
            raise NotImplementedError(f'Unrecognized input format for: {filename}')

    v = v.to(device, non_blocking=True).contiguous()
    f = f.to(device, non_blocking=True).contiguous()

    if load_uv:
        vm = vm.to(device, non_blocking=True).contiguous()
        fm = fm.to(device, non_blocking=True).contiguous()

    if load_uv:
        if load_aux:
            return v, f, vm, fm, aux
        else:
            return v, f, vm, fm
    else:
        return v, f


def load_pts(filename: str):
    from pyntcloud import PyntCloud
    cloud = PyntCloud.from_file(filename)
    verts = cloud.xyz
    if 'red' in cloud.points and 'green' in cloud.points and 'blue' in cloud.points:
        r = np.asarray(cloud.points['red'])
        g = np.asarray(cloud.points['green'])
        b = np.asarray(cloud.points['blue'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    elif 'r' in cloud.points and 'g' in cloud.points and 'b' in cloud.points:
        r = np.asarray(cloud.points['r'])
        g = np.asarray(cloud.points['g'])
        b = np.asarray(cloud.points['b'])
        colors = (np.stack([r, g, b], axis=-1) / 255).astype(np.float32)
    else:
        colors = None

    if 'nx' in cloud.points and 'ny' in cloud.points and 'nz' in cloud.points:
        nx = np.asarray(cloud.points['nx'])
        ny = np.asarray(cloud.points['ny'])
        nz = np.asarray(cloud.points['nz'])
        norms = np.stack([nx, ny, nz], axis=-1)
    else:
        norms = None

    # if 'alpha' in cloud.points:
    #     cloud.points['alpha'] = cloud.points['alpha'] / 255

    reserved = ['x', 'y', 'z', 'red', 'green', 'blue', 'r', 'g', 'b', 'nx', 'ny', 'nz']
    scalars = dotdict({k: np.asarray(cloud.points[k])[..., None] for k in cloud.points if k not in reserved})  # one extra dimension at the back added
    return verts, colors, norms, scalars


def export_pts(pts: torch.Tensor, color: torch.Tensor = None, normal: torch.Tensor = None, scalars: dotdict = dotdict(), filename: str = "default.ply", skip_color: bool = False, **kwargs):
    from pandas import DataFrame
    from pyntcloud import PyntCloud

    data = dotdict()
    pts = to_numpy(pts)  # always blocking?
    pts = pts.reshape(-1, 3)
    data.x = pts[:, 0].astype(np.float32)
    data.y = pts[:, 1].astype(np.float32)
    data.z = pts[:, 2].astype(np.float32)

    if color is not None:
        color = to_numpy(color)
        color = color.reshape(-1, 3)
        data.red = (color[:, 0] * 255).astype(np.uint8)
        data.green = (color[:, 1] * 255).astype(np.uint8)
        data.blue = (color[:, 2] * 255).astype(np.uint8)
    elif not skip_color:
        data.red = (pts[:, 0] * 255).astype(np.uint8)
        data.green = (pts[:, 1] * 255).astype(np.uint8)
        data.blue = (pts[:, 2] * 255).astype(np.uint8)

    if normal is not None:
        normal = to_numpy(normal)
        normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-13)
        normal = normal.reshape(-1, 3)
        data.nx = normal[:, 0].astype(np.float32)
        data.ny = normal[:, 1].astype(np.float32)
        data.nz = normal[:, 2].astype(np.float32)

    if scalars is not None:
        scalars = to_numpy(scalars)
        for k, v in scalars.items():
            v = v.reshape(-1, 1)
            data[k] = v[:, 0]

    df = DataFrame(data)
    cloud = PyntCloud(df)  # construct the data
    dir = dirname(filename)
    if dir: os.makedirs(dir, exist_ok=True)
    return cloud.to_file(filename, **kwargs)  # maybe write comments here: comments: list of strings


def export_lines(verts: torch.Tensor, lines: torch.Tensor, color: torch.Tensor = None, filename: str = 'default.ply'):
    if color is None:
        color = verts
    verts, lines, color = to_numpy([verts, lines, color])  # always blocking?
    if color.dtype == np.float32:
        color = (color * 255).astype(np.uint8)
    verts = verts.reshape(-1, 3)
    lines = lines.reshape(-1, 2)
    color = color.reshape(-1, 3)

    # Write to PLY
    with open(filename, 'wb') as f:
        # PLY header
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(verts)}\n".encode())
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(f"element edge {len(lines)}\n".encode())
        f.write(b"property int vertex1\n")
        f.write(b"property int vertex2\n")
        f.write(b"end_header\n")

        # Write vertices and colors
        for v, c in zip(verts, color):
            f.write(struct.pack('fffBBB', v[0], v[1], v[2], c[0], c[1], c[2]))

        # Write lines
        for l in lines:
            f.write(struct.pack('ii', l[0], l[1]))


def export_camera(c2w: torch.Tensor, ixt: torch.Tensor = None, col: torch.Tensor = torch.tensor([50, 50, 200]), axis_size=0.10, filename: str = 'default.ply'):
    verts = []
    lines = []
    rgbs = []

    def add_line(p0: torch.Tensor, p1: torch.Tensor, col: torch.Tensor):
        # Add a and b vertices
        verts.append(p0)  # N, M, 3
        verts.append(p1)  # N, M, 3
        sh = p0.shape[:-1]

        # Add the vertex colors
        col = torch.broadcast_to(col, sh + (3,))
        rgbs.append(col)
        rgbs.append(col)

        # Add the faces
        new = p0.numel() // 3  # number of new elements
        curr = new * (len(verts) - 2)  # assume all previous elements are of the same size
        start = torch.arange(curr, curr + new)
        end = torch.arange(curr + new, curr + new * 2)
        line = torch.stack([start, end], dim=-1)  # NM, 2
        line = line.view(sh + (2,))
        lines.append(line)

    c2w = c2w[..., :3, :]
    p = c2w[..., 3]  # third row (corresponding to 3rd column)

    if ixt is None: aspect = 1.0
    else: aspect = ixt[..., 0, 0][..., None] / ixt[..., 1, 1][..., None]
    if ixt is None: focal = 1000
    else: focal = (ixt[..., 0, 0][..., None] + ixt[..., 1, 1][..., None]) / 2

    axis_size = focal * axis_size / 1000
    xs = axis_size * aspect
    ys = axis_size
    zs = axis_size * aspect * 2

    a = p + xs * c2w[..., 0] + ys * c2w[..., 1] + zs * c2w[..., 2]
    b = p - xs * c2w[..., 0] + ys * c2w[..., 1] + zs * c2w[..., 2]
    c = p - xs * c2w[..., 0] - ys * c2w[..., 1] + zs * c2w[..., 2]
    d = p + xs * c2w[..., 0] - ys * c2w[..., 1] + zs * c2w[..., 2]

    add_line(p, p + axis_size * c2w[..., 0], torch.tensor([255, 64, 64]))
    add_line(p, p + axis_size * c2w[..., 1], torch.tensor([64, 255, 64]))
    add_line(p, p + axis_size * c2w[..., 2], torch.tensor([64, 64, 255]))
    add_line(p, a, col)
    add_line(p, b, col)
    add_line(p, c, col)
    add_line(p, d, col)
    add_line(a, b, col)
    add_line(b, c, col)
    add_line(c, d, col)
    add_line(d, a, col)

    verts = torch.stack(verts)
    lines = torch.stack(lines)
    rgbs = torch.stack(rgbs)

    export_lines(verts, lines, rgbs, filename=filename)


def export_mesh(verts: torch.Tensor, faces: torch.Tensor, uv: torch.Tensor = None, img: torch.Tensor = None, uvfaces: torch.Tensor = None, colors: torch.Tensor = None, normals: torch.Tensor = None, filename: str = "default.ply", subdivision=0):
    if dirname(filename): os.makedirs(dirname(filename), exist_ok=True)

    if subdivision > 0:
        from easyvolcap.utils.mesh_utils import face_normals, loop_subdivision
        verts, faces = loop_subdivision(verts, faces, subdivision)

    if filename.endswith('.npz'):
        def collect_args(**kwargs): return kwargs
        kwargs = collect_args(verts=verts, faces=faces, uv=uv, img=img, uvfaces=uvfaces, colors=colors, normals=normals)
        ret = dotdict({k: v for k, v in kwargs.items() if v is not None})
        export_dotdict(ret, filename)

    elif filename.endswith('.ply') or filename.endswith('.obj'):
        if uvfaces is None:
            mesh = get_mesh(verts, faces, uv, img, colors, normals, filename)
            mesh.export(filename)
        else:
            from pytorch3d.io import save_obj
            verts, faces, uv, img, uvfaces = get_tensor_mesh_data(verts, faces, uv, img, uvfaces)
            save_obj(filename, verts, faces, verts_uvs=uv, faces_uvs=uvfaces, texture_map=img)
    else:
        raise NotImplementedError(f'Unrecognized input format for: {filename}')


def export_pynt_pts_alone(pts, color=None, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    data = {}

    pts = pts if isinstance(pts, np.ndarray) else pts.detach().cpu().numpy()
    pts = pts.reshape(-1, 3)
    data['x'] = pts[:, 0].astype(np.float32)
    data['y'] = pts[:, 1].astype(np.float32)
    data['z'] = pts[:, 2].astype(np.float32)

    if color is not None:
        color = color if isinstance(color, np.ndarray) else color.detach().cpu().numpy()
        color = color.reshape(-1, 3)
        data['red'] = color[:, 0].astype(np.uint8)
        data['green'] = color[:, 1].astype(np.uint8)
        data['blue'] = color[:, 2].astype(np.uint8)
    else:
        data['red'] = (pts[:, 0] * 255).astype(np.uint8)
        data['green'] = (pts[:, 1] * 255).astype(np.uint8)
        data['blue'] = (pts[:, 2] * 255).astype(np.uint8)

    df = pd.DataFrame(data)
    cloud = PyntCloud(df)  # construct the data
    dirname = dirname(filename)
    if dirname: os.makedirs(dirname, exist_ok=True)
    return cloud.to_file(filename)


def export_o3d_pts(pts: torch.Tensor, filename: str = "default.ply"):
    import open3d as o3d
    pts = to_numpy(pts)
    pts = pts.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return o3d.io.write_point_cloud(filename, pcd)


def export_o3d_pcd(pts: torch.Tensor, rgb: torch.Tensor, normal: torch.Tensor, filename="default.ply"):
    import open3d as o3d
    pts, rgb, normal = to_numpy([pts, rgb, normal])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    normal = normal.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.normals = o3d.utility.Vector3dVector(normal)
    return o3d.io.write_point_cloud(filename, pcd)


def export_pcd(pts: torch.Tensor, rgb: torch.Tensor, occ: torch.Tensor, filename="default.ply"):
    import pandas as pd
    from pyntcloud import PyntCloud
    pts, rgb, occ = to_numpy([pts, rgb, occ])
    pts = pts.reshape(-1, 3)
    rgb = rgb.reshape(-1, 3)
    occ = occ.reshape(-1, 1)
    # MARK: CloudCompare bad, set first to 0, last to 1
    for i in range(3):
        rgb[0, i] = 0
        rgb[-1, i] = 1
    occ[0, 0] = 0
    occ[-1, 0] = 1

    data = dotdict()
    data.x = pts[:, 0]
    data.y = pts[:, 1]
    data.z = pts[:, 2]
    # TODO: maybe, for compability, save color as uint?
    # currently saving as float number from [0, 1]
    data.red = rgb[:, 0]
    data.green = rgb[:, 1]
    data.blue = rgb[:, 2]
    data.alpha = occ[:, 0]

    # MARK: We're saving extra scalars for loading in CloudCompare
    # can't assign same property to multiple fields
    data.r = rgb[:, 0]
    data.g = rgb[:, 1]
    data.b = rgb[:, 2]
    data.a = occ[:, 0]
    df = pd.DataFrame(data)

    cloud = PyntCloud(df)  # construct the data
    dirname = dirname(filename)
    if dirname: os.makedirs(dirname, exist_ok=True)
    return cloud.to_file(filename)
