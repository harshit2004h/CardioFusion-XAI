"""Tests for the PanEcho echo backend.

None of these need the real 170 MB checkpoint: they build the architecture with
random weights and save/load small synthetic checkpoints. To check the real
weights, run ``python scripts/prepare_panecho_checkpoint.py --source <file>``.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.echo_processing.panecho_reader import read_echo_for_panecho
from src.models.echo.panecho import (
    PanEchoModel,
    decode_outputs,
    heads_in_state_dict,
    load_panecho,
    resolve_checkpoint_path,
)
from src.models.echo.panecho_tasks import ALL_TASK_NAMES, TASKS_BY_NAME
from src.preprocessing.echo.panecho_transforms import clip_indices, clip_starts

SMALL = 64  # ConvNeXt is size-agnostic; small frames keep CPU tests quick


@pytest.fixture(scope="module")
def small_model() -> PanEchoModel:
    torch.manual_seed(0)
    return PanEchoModel(tasks=["EF", "LVSystolicFunction", "RASize", "LVWallMotionAbnormalities"]).eval()


def _write_avi(path: Path, frames: int, size: int = 112) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (size, size))
    for t in range(frames):
        frame = np.zeros((size, size, 3), np.uint8)
        cv2.circle(frame, (size // 2, size // 2), 20 + t % 10, (200, 200, 200), -1)
        writer.write(frame)
    writer.release()
    return path


def test_task_table_matches_release():
    assert len(ALL_TASK_NAMES) == 40
    assert TASKS_BY_NAME["EF"].task_type == "regression"
    assert TASKS_BY_NAME["LVSystolicFunction"].num_outputs == 3
    assert TASKS_BY_NAME["RASize"].invert_sigmoid and TASKS_BY_NAME["RASize"].positive_label == "Dilated"


def test_forward_shapes_and_activations(small_model):
    x = torch.randn(2, 3, 16, SMALL, SMALL)
    with torch.inference_mode():
        out = small_model(x)
    assert out["EF"].shape == (2, 1)
    assert out["LVSystolicFunction"].shape == (2, 3)
    assert torch.allclose(out["LVSystolicFunction"].sum(1), torch.ones(2), atol=1e-5)
    assert ((out["RASize"] >= 0) & (out["RASize"] <= 1)).all()


def test_binary_inversion_matches_release_convention(small_model):
    feats = torch.randn(1, 768)
    small_model.activations = False
    logits = small_model.forward_heads(feats)
    small_model.activations = True
    probs = small_model.forward_heads(feats)
    assert torch.allclose(probs["RASize"], 1 - torch.sigmoid(logits["RASize"]))  # positive = index 0
    assert torch.allclose(probs["LVWallMotionAbnormalities"], torch.sigmoid(logits["LVWallMotionAbnormalities"]))


def test_rejects_wrong_rank(small_model):
    with pytest.raises(ValueError):
        small_model(torch.randn(3, 16, SMALL, SMALL))


def test_multi_clip_average(small_model):
    clips = torch.randn(3, 3, 16, SMALL, SMALL)
    with torch.inference_mode():
        per_clip = small_model(clips)["EF"].mean()
    averaged = small_model.predict_clips(clips)["EF"]
    assert averaged.shape == (1, 1)
    assert torch.isclose(averaged.reshape(()), per_clip, atol=1e-4)


def test_checkpoint_round_trip_formats(tmp_path, small_model):
    state = {f"module.{k}": v for k, v in small_model.state_dict().items()}  # DDP-style prefix
    official_style = tmp_path / "panecho.pt"
    torch.save({"weights": state, "optimizer": {}}, official_style)

    loaded = load_panecho(official_style)
    assert loaded.task_names == small_model.task_names
    x = torch.randn(1, 3, 16, SMALL, SMALL)
    with torch.inference_mode():
        assert torch.allclose(loaded(x)["EF"], small_model(x)["EF"], atol=1e-5)

    subset = load_panecho(official_style, tasks=["EF"])
    assert subset.task_names == ["EF"]
    with pytest.raises(KeyError):
        load_panecho(official_style, tasks=["AVStenosis"])

    # A zip that *wraps* the .pt (how the download was named) is extracted transparently.
    wrapped = tmp_path / "dl" / "panecho.pt.zip"
    wrapped.parent.mkdir()
    with zipfile.ZipFile(wrapped, "w", zipfile.ZIP_STORED) as archive:
        archive.write(official_style, "panecho.pt")
    assert resolve_checkpoint_path(wrapped).name == "panecho.pt"
    assert load_panecho(wrapped).task_names == small_model.task_names


def test_architecture_mismatch_is_reported(tmp_path, small_model):
    state = dict(small_model.state_dict())
    state.pop("encoder.transformer.layers.0.linear1.weight")
    bad = tmp_path / "bad.pt"
    torch.save({"weights": state}, bad)
    with pytest.raises(RuntimeError, match="missing"):
        load_panecho(bad)


def test_heads_in_state_dict_preserves_release_order(small_model):
    assert heads_in_state_dict(small_model.state_dict()) == ["EF", "LVSystolicFunction", "LVWallMotionAbnormalities", "RASize"]


def test_decode_outputs(small_model):
    with torch.inference_mode():
        decoded = decode_outputs(small_model(torch.randn(1, 3, 16, SMALL, SMALL)))
    assert decoded["EF"]["unit"] == "%"
    assert decoded["RASize"]["positive_label"] == "Dilated"
    assert set(decoded["LVSystolicFunction"]["probabilities"]) == set(TASKS_BY_NAME["LVSystolicFunction"].class_names)


def test_clip_sampling():
    assert clip_starts(10, 16, 4) == [0, 0, 0, 0]
    assert clip_starts(116, 16, 1) == [50]
    assert clip_starts(116, 16, 3) == [0, 50, 100]
    assert clip_indices(0, 10, 16).tolist() == list(range(10)) + [9] * 6


def test_reader_produces_panecho_clips(tmp_path):
    clips, meta = read_echo_for_panecho(_write_avi(tmp_path / "a.avi", 40), "avi", num_clips=2)
    assert clips.shape == (2, 3, 16, 224, 224)
    assert clips.dtype == torch.float32
    assert meta["format"] == "AVI" and not meta["short_video_padded"]

    short, meta = read_echo_for_panecho(_write_avi(tmp_path / "b.avi", 5), "avi", num_clips=1)
    assert short.shape == (1, 3, 16, 224, 224) and meta["short_video_padded"]


@pytest.mark.slow
def test_model_manager_prefers_panecho(tmp_path, monkeypatch):
    from src.inference.model_manager import ModelManager

    for name in ("ECHO_MODEL_BACKEND", "PANECHO_CHECKPOINT", "ECG_CHECKPOINT", "BIOMARKER_CHECKPOINT"):
        monkeypatch.delenv(name, raising=False)
    torch.manual_seed(0)
    model = PanEchoModel(tasks=["EF", "RASize"])
    target = tmp_path / "checkpoints" / "echo" / "panecho" / "panecho.pt"
    target.parent.mkdir(parents=True)
    torch.save({"weights": model.state_dict()}, target)

    manager = ModelManager(tmp_path)
    assert manager.echo_backend == "panecho"
    assert manager.status["echo"] == "loaded_panecho_pretrained"

    clips, _ = read_echo_for_panecho(_write_avi(tmp_path / "v.avi", 30), "avi", num_clips=2)
    model_ef = manager.echo_model.predict_clips(clips)["EF"].item()
    if 0 <= model_ef <= 100:
        fusion, auxiliary = manager.predict_echo_panecho(clips)
        assert set(fusion) == {"EF"} and "RASize" in auxiliary
    with pytest.raises(ValueError):
        manager.predict_echo_panecho(torch.zeros(1, 3, 8, 224, 224))

    monkeypatch.setenv("ECHO_MODEL_BACKEND", "r2plus1d")
    assert ModelManager(tmp_path).echo_backend is None  # no R(2+1)D checkpoint in tmp project


def test_reader_handles_multiframe_dicom(tmp_path):
    pytest.importorskip("pydicom")
    from tests.test_dicom import _dataset

    path = tmp_path / "echo.dcm"
    dataset = _dataset(path)
    dataset.Modality = "US"
    dataset.Rows = dataset.Columns = 48
    dataset.NumberOfFrames = 20
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    dataset.FrameTime = 20.0  # ms -> 50 fps
    dataset.PixelData = (np.arange(20 * 48 * 48) % 255).astype(np.uint8).tobytes()
    dataset.save_as(path)

    clips, meta = read_echo_for_panecho(path, "dicom", num_clips=2)
    assert clips.shape == (2, 3, 16, 224, 224)
    assert meta["format"] == "DICOM" and meta["fps"] == pytest.approx(50.0)
