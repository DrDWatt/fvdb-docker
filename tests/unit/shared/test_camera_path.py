"""Flythrough camera path shared by :8085 (server frames) and :8086 (viewer animTrack)."""
import numpy as np
import pytest

from support.ply import orbit_cameras
from viewer_common.camera_path import (
    MIN_VIDEO_SECONDS, build_smooth_path, ease_in_out, frames_for_duration, sample_path, slerp_rotation,
)


def jittered_cameras(n=60, seed=0):
    """Orbit cameras with handheld-style positional jitter and uneven spacing."""
    c2w, K, sizes = orbit_cameras(n)
    rng = np.random.default_rng(seed)
    c2w[:, :3, 3] += rng.normal(0, 0.05, (n, 3))
    keep = np.sort(rng.choice(n, size=int(n * 0.8), replace=False))  # uneven gaps
    return c2w[keep], K[keep], sizes[keep]


def positions(path, samples=300):
    return np.array([sample_path(path, u)[0] for u in np.linspace(0, 1, samples)])


def test_minimum_video_length_is_thirty_seconds():
    assert MIN_VIDEO_SECONDS == 30
    assert frames_for_duration(5, 30) == 900
    assert frames_for_duration(30, 24) == 720
    assert frames_for_duration(45, 30) == 1350


def test_ease_in_out_is_monotonic_and_clamped():
    us = np.linspace(-0.5, 1.5, 201)
    eased = np.array([ease_in_out(u) for u in us])
    assert eased[0] == 0 and eased[-1] == 1
    assert np.all(np.diff(eased) >= -1e-12)
    assert ease_in_out(0.5) == pytest.approx(0.5)


def test_slerp_returns_proper_rotations():
    a = np.eye(3)
    b = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90 deg about z
    for t in np.linspace(0, 1, 11):
        r = slerp_rotation(a, b, t)
        assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(r) == pytest.approx(1.0)


def test_path_moves_at_constant_speed_despite_uneven_cameras():
    c2w, K, sizes = jittered_cameras()
    raw_steps = np.linalg.norm(np.diff(c2w[:, :3, 3], axis=0), axis=1)
    steps = np.linalg.norm(np.diff(positions(build_smooth_path(c2w, K, sizes)), axis=0), axis=1)
    # raw captures vary a lot; the sampled path's step length stays nearly uniform
    assert raw_steps.max() / raw_steps.mean() > 2.0
    assert steps.max() / steps.mean() < 1.6


def test_path_is_smoother_than_raw_cameras():
    c2w, K, sizes = jittered_cameras()
    path = build_smooth_path(c2w, K, sizes)
    jerk = np.linalg.norm(np.diff(positions(path, len(c2w)), n=2, axis=0), axis=1)
    raw_jerk = np.linalg.norm(np.diff(c2w[:, :3, 3], n=2, axis=0), axis=1)
    assert jerk.mean() < raw_jerk.mean() * 0.5


def test_path_does_not_wrap_back_to_first_camera():
    c2w, K, sizes = orbit_cameras(24)
    half = slice(0, 12)                          # half an orbit: start and end far apart
    path = build_smooth_path(c2w[half], K[half], sizes[half])
    start, end = sample_path(path, 0.0)[0], sample_path(path, 1.0)[0]
    assert np.linalg.norm(end - start) > 4.0
    assert np.linalg.norm(start - c2w[0, :3, 3]) < 0.3


def test_sampled_rotations_and_intrinsics_are_valid():
    c2w, K, sizes = orbit_cameras(30)
    path = build_smooth_path(c2w, K, sizes, depths=np.full(30, 3.0))
    for u in np.linspace(0, 1, 25):
        pos, rot, k_norm, depth = sample_path(path, u)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-6)
        assert k_norm[0, 0] == pytest.approx(900 / 1280, rel=1e-6)   # fx normalized by width
        assert k_norm[1, 1] == pytest.approx(900 / 720, rel=1e-6)    # fy normalized by height
        assert depth == pytest.approx(3.0)
        assert np.linalg.norm(pos) == pytest.approx(np.linalg.norm(c2w[0, :3, 3]), rel=0.05)


def test_degenerate_cameras_are_dropped():
    c2w, K, sizes = orbit_cameras(20)
    c2w[3] = 0.0                                  # all-zero pose (det 0)
    c2w[7, 0, 0] = np.inf
    path = build_smooth_path(c2w, K, sizes)
    assert path["num_cameras"] == 18
    assert np.isfinite(positions(path)).all()


def test_fewer_than_two_valid_cameras_raise():
    c2w, K, sizes = orbit_cameras(3)
    c2w[1:] = 0.0
    with pytest.raises(ValueError):
        build_smooth_path(c2w, K, sizes)
