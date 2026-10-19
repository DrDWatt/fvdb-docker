"""Up-axis estimate for PLYs without trained cameras -> viewer ?orient= and a level orbit."""
import numpy as np

from support.ply import grid_points, orbit_cameras, write_splat_ply


def _z_up_scene(n=60, extent=10.0):
    """SYNTHETIC Z-up scene: flat ground Gaussians (thin along z) + a 4 m block rising above it."""
    g = np.linspace(-extent, extent, n)
    X, Y = np.meshgrid(g, g)
    ground = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    h = np.linspace(0, 4, 20)
    A, H = np.meshgrid(np.linspace(-1, 1, 20), h)
    walls = np.r_[np.c_[A.ravel(), np.full(A.size, -1.0), H.ravel()], np.c_[A.ravel(), np.full(A.size, 1.0), H.ravel()]]
    pts = np.r_[ground, walls]
    scales = np.r_[np.tile([-2.0, -2.0, -6.0], (len(ground), 1)), np.tile([-2.0, -6.0, -2.0], (len(walls), 1))]  # thin axis = normal
    return pts, scales


def test_z_up_scene_is_rotated_so_up_is_viewer_y(ss, client):
    from features.orientation import orientation, quat_xyzw
    pts, scales = _z_up_scene()
    write_splat_ply(ss.MODEL_DIR / "zup.ply", points=pts, log_scales=scales)
    o = orientation(ss.MODEL_DIR / "zup.ply", False)
    R = np.asarray(o["matrix"])
    assert np.allclose(R @ [0, 0, 1.0], [0, 1, 0], atol=1e-6) and np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    x, y, z, w = o["quat"]
    assert np.isclose(np.linalg.norm(o["quat"]), 1) and quat_xyzw(R) == o["quat"]
    info = client.get("/flythrough/info/zup.ply").json()
    assert info["mode"] == "arc" and info["orient"] == o["quat"]  # re-oriented: sweep around the clearest view
    # the orbit stays level in the viewer frame: constant height, circling the scene centre
    settings = client.get("/flythrough/settings/zup.ply").json()
    pos = np.asarray(settings["animTracks"][0]["keyframes"]["values"]["position"]).reshape(-1, 3)
    assert np.ptp(pos[:, 1]) < 1e-6 and pos[:, 1].mean() > 0
    assert "&orient=" in ss.viewer_src("zup.ply") and ss.viewer_src("zup.ply").endswith(",".join(str(v) for v in o["quat"]))


def test_fvdb_scenes_and_unflat_scenes_keep_the_viewer_default(ss):
    from features.orientation import orientation, rotation_for_up
    assert np.allclose(rotation_for_up(np.array([0, -1.0, 0])), np.diag([-1.0, -1, 1]))  # fVDB up -y: default
    write_splat_ply(ss.MODEL_DIR / "cams.ply", points=grid_points(12), cameras=orbit_cameras(8))
    assert orientation(ss.MODEL_DIR / "cams.ply", True) is None and "&orient=" not in ss.viewer_src("cams.ply")
    write_splat_ply(ss.MODEL_DIR / "round.ply", points=grid_points(12))  # isotropic Gaussians: no surface cue
    assert orientation(ss.MODEL_DIR / "round.ply", False) is None
