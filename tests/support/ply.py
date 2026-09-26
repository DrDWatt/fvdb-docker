"""Synthetic Gaussian-splat PLY files for tests.

write_splat_ply() produces the same layout the viewers consume: a binary
little-endian 3DGS vertex element and, optionally, the trained cameras that fVDB
stores as extra PLY elements (camera_to_world_matrices, projection_matrices,
image_sizes, median_depths) described by `fvdb_ply_af_*|tensor|` comments.
"""
from pathlib import Path

import numpy as np

VERTEX_PROPS = (["x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
                 "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2"])


def grid_points(n_side=10, extent=1.0):
    """n_side^2 points on a regular grid in the z=0 plane, x/y in [-extent, extent]."""
    lin = np.linspace(-extent, extent, n_side)
    xx, yy = np.meshgrid(lin, lin)
    return np.stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)], axis=1)


def orbit_cameras(n=24, radius=3.0, height=0.5, image_size=(720, 1280), focal=900.0):
    """OpenCV-convention camera-to-world matrices on a circle looking at the origin."""
    c2w, K, sizes = [], [], []
    for i in range(n):
        a = 2 * np.pi * i / n
        pos = np.array([radius * np.cos(a), height, radius * np.sin(a)])
        fwd = -pos / np.linalg.norm(pos)
        right = np.cross(fwd, [0.0, -1.0, 0.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, down, fwd, pos
        c2w.append(m)
        h, w = image_size
        K.append([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        sizes.append([h, w])
    return np.array(c2w), np.array(K, dtype=np.float64), np.array(sizes)


def write_splat_ply(path: Path, points=None, cameras=None) -> Path:
    """Write a 3DGS PLY; `cameras` = (c2w, K, sizes) adds fVDB camera elements."""
    points = grid_points() if points is None else np.asarray(points, dtype=np.float64)
    n = len(points)
    vertex = np.zeros((n, len(VERTEX_PROPS)), dtype=np.float32)
    vertex[:, :3] = points
    vertex[:, 3] = 2.0                      # opacity (pre-sigmoid)
    vertex[:, 4:7] = -4.0                   # log scales
    vertex[:, 7] = 1.0                      # identity quaternion
    vertex[:, 11:14] = 0.5

    header = ["ply", "format binary_little_endian 1.0", "comment fvdb_gs_ply_version fvdb_ply 1.0.0"]
    extra = []
    if cameras is not None:
        c2w, K, sizes = cameras
        m = len(c2w)
        depths = np.linalg.norm(c2w[:, :3, 3], axis=1)
        tensors = [("median_depths", depths.astype(np.float32), "float", (m,)),
                   ("image_sizes", sizes.astype(np.int32), "int", (m, 2)),
                   ("projection_matrices", K.astype(np.float32), "float", (m, 3, 3)),
                   ("camera_to_world_matrices", c2w.astype(np.float32), "float", (m, 4, 4))]
        for name, _, _, shape in tensors:
            header.append(f"comment fvdb_ply_af_1234{name}|tensor|{len(shape)},{','.join(map(str, shape))}")
        extra = tensors
    header.append(f"element vertex {n}")
    header += [f"property float {p}" for p in VERTEX_PROPS]
    for name, arr, ptype, _ in extra:
        header += [f"element {name} {arr.size}", f"property {ptype} value"]
    header.append("end_header")

    path = Path(path)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(vertex.astype("<f4").tobytes())
        for _, arr, ptype, _ in extra:
            f.write(arr.astype("<i4" if ptype == "int" else "<f4").ravel().tobytes())
    return path


def read_positions(path: Path) -> np.ndarray:
    """(N, 3) gaussian centers of an all-float binary PLY (e.g. an extraction result)."""
    names, count, element = [], 0, None
    with open(path, "rb") as f:
        while (line := f.readline().decode("ascii", errors="ignore").strip()) != "end_header":
            if line.startswith("element"):
                element = line.split()[1]
                count = int(line.split()[2]) if element == "vertex" else count
            elif line.startswith("property") and element == "vertex":
                names.append(line.split()[-1])
        data = np.frombuffer(f.read(count * 4 * len(names)), dtype="<f4").reshape(count, len(names))
    return data[:, [names.index("x"), names.index("y"), names.index("z")]]


def read_vertex_count(path: Path) -> int:
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    return 0
