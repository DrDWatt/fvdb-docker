""":8086 flythrough (SuperSplat animTrack settings) and MP4 export sessions."""
import math

import numpy as np
import pytest

from support.media import assert_universal_mp4, jpeg_bytes, probe_mp4
from support.ply import grid_points, orbit_cameras, write_splat_ply


def keyframes(settings):
    track = settings["animTracks"][0]
    return track, np.array(track["keyframes"]["values"]["position"]).reshape(-1, 3)


# ----- flythrough track -----
def test_trained_cameras_become_a_looping_animation_track(client, scene):
    settings = client.get(f"/flythrough/settings/{scene}?duration=30").json()
    track, positions = keyframes(settings)
    assert settings["version"] == 2 and settings["startMode"] == "animTrack"
    assert track["duration"] == 60 and track["loopMode"] == "repeat" and track["smoothness"] == 1
    assert len(track["keyframes"]["times"]) == len(positions) == 600       # every 0.1 s, out and back
    assert np.all(np.diff(track["keyframes"]["times"]) > 0)
    # viewer shows splats rotated (0,0,180): first key = first camera with x/y negated
    c2w = orbit_cameras(24)[0][0]
    assert positions[0] == pytest.approx([-c2w[0, 3], -c2w[1, 3], c2w[2, 3]], abs=0.35)
    # horizontal FOV derived from the trained intrinsics (fx=900 px, width=1280 px)
    fov = track["keyframes"]["values"]["fov"][0]
    assert fov == pytest.approx(math.degrees(2 * math.atan(1280 / (2 * 900))), abs=0.01)
    assert settings["cameras"][0]["initial"]["position"] == pytest.approx(list(positions[0]))


def test_track_plays_out_and_back_so_the_loop_is_seamless(client, scene):
    _, positions = keyframes(client.get(f"/flythrough/settings/{scene}").json())
    forward, back = positions[:300], positions[300:]
    assert np.linalg.norm(positions[0] - positions[-1]) < 0.05              # loop closes
    assert np.allclose(forward[1:], back[::-1][:-1], atol=0.05)             # return retraces the path
    steps = np.linalg.norm(np.diff(forward, axis=0), axis=1)
    assert steps[:5].mean() < steps[140:160].mean() * 0.2                  # eases in from rest


def test_duration_is_clamped_to_the_thirty_second_minimum(client, scene):
    assert client.get(f"/flythrough/settings/{scene}?duration=5").json()["animTracks"][0]["duration"] == 60
    info = client.get(f"/flythrough/info/{scene}?duration=500").json()
    assert (info["duration"], info["min_duration"], info["max_duration"]) == (120, 30, 120)
    assert info["mode"] == "camera_path" and info["num_cameras"] == 24


def test_models_without_cameras_orbit_the_scene(client, ss):
    write_splat_ply(ss.MODEL_DIR / "cloud.ply", points=grid_points(12, extent=2.0))
    settings = client.get("/flythrough/settings/cloud.ply?duration=40").json()
    track, positions = keyframes(settings)
    assert track["duration"] == 40 and len(positions) == 400
    radii = np.linalg.norm(positions[:, [0, 2]], axis=1)
    assert radii.std() < 1e-6 and radii.mean() > 2.0                        # constant-radius orbit
    assert client.get("/flythrough/info/cloud.ply").json()["mode"] == "orbit"


def test_unreadable_models_fall_back_to_the_viewer_default_camera(client, ss):
    (ss.MODEL_DIR / "packed.ply").write_bytes(b"ply\nformat binary_little_endian 1.0\nelement chunk 1\n"
                                              b"property list uchar float packed\nend_header\n\x00")
    settings = client.get("/flythrough/settings/packed.ply").json()
    assert settings["animTracks"] == [] and settings["startMode"] == "default"


def test_unknown_model_is_404(client):
    assert client.get("/flythrough/settings/nope.ply").status_code == 404


# ----- MP4 export -----
def start(client, model, **overrides):
    body = {"model": model, "duration": 30, "fps": 30, "width": 320, "height": 180, **overrides}
    return client.post("/flythrough/export/start", json=body)


def send(client, sid, first, count):
    files = [("frames", (f"f{i}.jpg", jpeg_bytes(320, 180, shade=i), "image/jpeg")) for i in range(first, first + count)]
    return client.post(f"/flythrough/export/{sid}/frames", files=files).json()


def test_export_produces_a_universal_mp4_of_at_least_thirty_seconds(client, ss, scene):
    session = start(client, scene, duration=10).json()                    # below the minimum
    assert (session["num_frames"], session["duration"]) == (900, 30)
    for first in range(0, 900, 150):
        progress = send(client, session["session_id"], first, 150)
    assert progress == {"received": 900, "expected": 900}
    mp4 = client.post(f"/flythrough/export/{session['session_id']}/finish")
    assert mp4.status_code == 200 and mp4.headers["content-type"] == "video/mp4"
    assert 'filename="scene_flythrough.mp4"' in mp4.headers["content-disposition"]
    assert_universal_mp4(probe_mp4(mp4.content), min_seconds=30, fps=30, size=(320, 180))
    assert not list(ss.OUTPUT_DIR.glob("*.mp4"))                         # server copy removed after download


def test_export_honours_fps_and_rejects_incomplete_videos(client, ss, scene):
    session = start(client, scene, fps=24).json()
    assert session["num_frames"] == 720
    send(client, session["session_id"], 0, 100)
    resp = client.post(f"/flythrough/export/{session['session_id']}/finish")
    assert resp.status_code == 400 and "100/720" in resp.json()["message"]
    assert not list(ss.OUTPUT_DIR.glob("*.mp4"))


def test_export_session_validation_and_abort(client, ss, scene):
    assert start(client, scene, width=50).status_code == 400
    assert start(client, "nope.ply").status_code == 404
    assert start(client, scene, fps=500).json()["fps"] == 60
    assert client.post("/flythrough/export/unknown/finish").status_code == 404
    session = start(client, scene).json()
    assert client.delete(f"/flythrough/export/{session['session_id']}").json()["status"] == "ok"
    assert client.post(f"/flythrough/export/{session['session_id']}/frames",
                       files=[("frames", ("a.jpg", jpeg_bytes(), "image/jpeg"))]).status_code == 404
    assert not list(ss.OUTPUT_DIR.glob("*.mp4"))
