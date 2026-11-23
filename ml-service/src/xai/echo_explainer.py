from __future__ import annotations

from typing import Callable

import numpy as np


def temporal_occlusion_explanation(
    predict_function: Callable[[np.ndarray], float],
    clip: np.ndarray,
    window_frames: int = 4,
    baseline: np.ndarray | None = None,
) -> dict:
    """Estimate temporal importance by replacing consecutive frame windows.

    Expected clip shape is [T, H, W, C] or [K, 3, T, 224, 224] only when the
    caller's prediction function explicitly accepts the latter shape.
    This is a perturbation explanation, not SHAP and not a validated clinical map.
    """
    array = np.asarray(clip, dtype=np.float32)
    if array.ndim not in {4, 5}:
        raise ValueError("clip must be [T,H,W,C] or a model-specific 5D tensor.")
    time_axis = 0 if array.ndim == 4 else 2
    frame_count = array.shape[time_axis]
    if window_frames <= 0 or window_frames > frame_count:
        raise ValueError("window_frames must be between 1 and the number of frames.")
    reference = np.median(array, axis=time_axis, keepdims=True) if baseline is None else np.asarray(baseline, dtype=np.float32)
    reference = np.broadcast_to(reference, array.shape).copy()
    original_score = float(predict_function(array.copy()))
    intervals = []
    scores = []
    for start in range(0, frame_count, window_frames):
        end = min(frame_count, start + window_frames)
        perturbed = array.copy()
        selector = [slice(None)] * array.ndim
        selector[time_axis] = slice(start, end)
        perturbed[tuple(selector)] = reference[tuple(selector)]
        changed_score = float(predict_function(perturbed))
        if not np.isfinite(changed_score):
            raise ValueError("Echo prediction function returned a non-finite score during occlusion.")
        intervals.append({"start_frame": int(start), "end_frame_exclusive": int(end)})
        scores.append(original_score - changed_score)
    return {
        "method": "temporal_window_occlusion",
        "original_output": original_score,
        "window_frames": int(window_frames),
        "windows": [{**interval, "output_difference": float(score)} for interval, score in zip(intervals, scores, strict=True)],
        "warning": "Output differences depend on the occlusion baseline and do not establish anatomical causality.",
    }
