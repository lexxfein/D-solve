import trimesh

from cv.alignment.alignment3d import auto_align, inspect_model, manual_transform


def test_3d_alignment_round_trip():
    mesh = trimesh.creation.box(extents=[4, 2, 1])
    source = mesh.export(file_type="glb")

    info = inspect_model(source, "sample.glb")
    assert info.vertices == 8
    assert info.faces == 12
    assert info.meshes == 1

    aligned, details = auto_align(source, "sample.glb", 100)
    assert len(aligned) > 0
    assert details["strength"] == 100.0

    transformed = manual_transform(
        source,
        "sample.glb",
        tx=1,
        ty=2,
        tz=3,
        rx=5,
        ry=10,
        rz=15,
        scale=1.1,
    )
    transformed_info = inspect_model(transformed, "sample.glb")
    assert transformed_info.center[0] == 1.0
    assert transformed_info.center[1] == 2.0
    assert transformed_info.center[2] == 3.0
