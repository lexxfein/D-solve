from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

SUPPORTED_EXTENSIONS = {".glb", ".gltf", ".obj", ".stl"}


@dataclass
class ModelInfo:
    vertices: int
    faces: int
    meshes: int
    bounds_min: list[float]
    bounds_max: list[float]
    center: list[float]
    size: list[float]


def _load_scene(data: bytes, filename: str) -> trimesh.Scene:
    suffix = ".glb"
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[1].lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported 3D formats are GLB, GLTF, OBJ and STL.")
    try:
        loaded = trimesh.load(io.BytesIO(data), file_type=suffix.lstrip("."), force="scene")
    except Exception as exc:
        raise ValueError(f"Could not read the 3D model: {exc}") from exc
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise ValueError("The uploaded 3D file contains no usable mesh geometry.")
    return loaded


def _world_meshes(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    meshes: list[trimesh.Trimesh] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geom_name = scene.graph[node_name]
        geom = scene.geometry.get(geom_name)
        if not isinstance(geom, trimesh.Trimesh):
            continue
        mesh = geom.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def _all_vertices(scene: trimesh.Scene) -> np.ndarray:
    meshes = _world_meshes(scene)
    if not meshes:
        raise ValueError("No mesh geometry found in the model.")
    return np.vstack([m.vertices for m in meshes]).astype(np.float64)


def inspect_model(data: bytes, filename: str) -> ModelInfo:
    scene = _load_scene(data, filename)
    vertices = _all_vertices(scene)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    faces = sum(len(m.faces) for m in _world_meshes(scene))
    return ModelInfo(
        vertices=int(len(vertices)),
        faces=int(faces),
        meshes=int(len(_world_meshes(scene))),
        bounds_min=mins.round(6).tolist(),
        bounds_max=maxs.round(6).tolist(),
        center=((mins + maxs) / 2).round(6).tolist(),
        size=(maxs - mins).round(6).tolist(),
    )


def _pca_rotation(vertices: np.ndarray) -> np.ndarray:
    center = vertices.mean(axis=0)
    x = vertices - center
    cov = np.cov(x, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    axes = vecs[:, np.argsort(vals)[::-1]]

    # Dental scans normally have the arch as the longest axis. Map it to X.
    x_axis = axes[:, 0]
    # The smallest variance is generally the vertical thickness/height axis.
    y_axis = axes[:, 2]
    z_axis = axes[:, 1]

    # Keep a right-handed frame.
    if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
        z_axis = -z_axis

    # Prefer +Y as the direction containing the upper half of the model.
    centered = vertices - center
    if np.dot(centered.mean(axis=0), y_axis) < 0:
        y_axis = -y_axis
        z_axis = -z_axis

    # Prefer +X to point toward the side with larger projected spread only to
    # make repeated uploads deterministic. The sign itself has no clinical meaning.
    if x_axis[np.argmax(np.abs(x_axis))] < 0:
        x_axis = -x_axis
        z_axis = -z_axis

    # Columns are target axes expressed in source coordinates. The transpose
    # maps source coordinates into the target frame.
    basis = np.column_stack([x_axis, y_axis, z_axis])
    return basis.T


def _compose_transform(tx: float, ty: float, tz: float, rx: float, ry: float, rz: float, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be greater than 0.")
    rotation = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation * float(scale)
    matrix[:3, 3] = [tx, ty, tz]
    return matrix


def _apply_to_scene(scene: trimesh.Scene, matrix: np.ndarray) -> trimesh.Scene:
    scene = scene.copy()
    scene.apply_transform(matrix)
    return scene


def _export_glb(scene: trimesh.Scene) -> bytes:
    try:
        exported = scene.export(file_type="glb")
    except Exception as exc:
        raise ValueError(f"Could not export the aligned GLB: {exc}") from exc
    if not isinstance(exported, (bytes, bytearray)):
        raise ValueError("The 3D exporter did not return a GLB byte stream.")
    return bytes(exported)


def manual_transform(data: bytes, filename: str, *, tx: float, ty: float, tz: float, rx: float, ry: float, rz: float, scale: float) -> bytes:
    scene = _load_scene(data, filename)
    matrix = _compose_transform(tx, ty, tz, rx, ry, rz, scale)
    return _export_glb(_apply_to_scene(scene, matrix))


def _dental_arch_deformation(vertices: np.ndarray, strength: float) -> np.ndarray:
    """Apply a conservative, geometry-only arch-normalization deformation.

    The supplied Tripo scan is a single fused mesh, so there are no independent
    tooth objects that can be translated one-by-one. Instead, the automatic
    mode uses the model's own coordinate system to gently normalize the visible
    upper/lower dental arch: it increases lateral spacing slightly, raises the
    upper central arch and lowers the lower central arch, with smooth falloff.

    This is a visualization/alignment transform, not a clinical orthodontic
    prediction. The deformation is intentionally small and continuous so the
    original surface and textures remain intact.
    """
    if len(vertices) == 0:
        return vertices.copy()

    alpha = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if alpha == 0:
        return vertices.copy()

    v = vertices.astype(np.float64, copy=True)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    y_min, y_max = float(y.min()), float(y.max())
    y_span = max(y_max - y_min, 1e-9)
    x_half = max(float(np.max(np.abs(x))), 1e-9)

    # The model uses Y as its vertical dental axis. These broad bands cover the
    # upper and lower tooth rows while fading smoothly into the gingiva.
    upper = np.exp(-((y - (y_min + 0.61 * y_span)) / (0.18 * y_span)) ** 2)
    lower = np.exp(-((y - (y_min + 0.30 * y_span)) / (0.17 * y_span)) ** 2)
    influence = np.clip(upper + lower, 0.0, 1.0)

    # Suppress deformation close to the very top/bottom of the scan.
    edge_low = np.clip((y - y_min) / (0.08 * y_span), 0.0, 1.0)
    edge_high = np.clip((y_max - y) / (0.08 * y_span), 0.0, 1.0)
    influence *= edge_low * edge_high

    xn = np.clip(np.abs(x) / x_half, 0.0, 1.0)

    # Gently open the arch laterally. The correction is strongest away from the
    # midline and fades to zero at the centreline.
    lateral_shift = 0.035 * np.sign(x) * (1.0 - xn) ** 0.7
    v[:, 0] += alpha * lateral_shift * influence

    # Normalize the vertical arch: upper central teeth rise slightly and lower
    # central teeth drop slightly. This makes the automatic result visibly
    # different from a simple rigid/PCA rotation while remaining conservative.
    arch_shape = 1.0 - xn ** 2
    v[:, 1] += alpha * (0.018 * upper - 0.014 * lower) * arch_shape * influence

    # A small depth normalization reduces accidental forward/backward wobble in
    # the tooth-bearing region without flattening the whole model.
    tooth_band = influence > 0.35
    z_reference = float(np.median(z[tooth_band])) if np.any(tooth_band) else float(np.median(z))
    v[:, 2] += alpha * (-0.025 * (z - z_reference) * influence)

    return v


def _apply_dental_arch_deformation(scene: trimesh.Scene, strength: float) -> trimesh.Scene:
    scene = scene.copy()
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            geometry.vertices = _dental_arch_deformation(geometry.vertices, strength)
            geometry.fix_normals()
    return scene


def auto_align(data: bytes, filename: str, strength: float = 100.0) -> tuple[bytes, dict]:
    scene = _load_scene(data, filename)
    strength = float(np.clip(strength, 0, 100))

    # The supplied dental scan is a single fused mesh. A rigid PCA operation can
    # be almost a no-op when the scan is already upright, which is not useful to
    # the user. Use a smooth dental-arch normalization instead. This visibly
    # changes the tooth-bearing arch while preserving the mesh/materials.
    transformed = _apply_dental_arch_deformation(scene, strength)
    out = _export_glb(transformed)
    return out, {
        "strength": round(strength, 1),
        "translation": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, 0.0],
        "scale": 1.0,
        "deformation": "dental_arch_normalization",
        "note": "Automatic 3D alignment applies a conservative dental-arch normalization to the fused scan: lateral spacing, upper/lower arch height, and tooth-band depth are adjusted smoothly. It does not move individually segmented teeth and is intended for visual design alignment, not clinical prediction.",
    }
