from __future__ import annotations

import numpy as np
import torch


def integrated_gradients_ecg(
    model: torch.nn.Module,
    waveform: np.ndarray | torch.Tensor,
    target_index: int,
    task: str = "diagnostic",
    baseline: np.ndarray | torch.Tensor | None = None,
    steps: int = 32,
    device: str | torch.device = "cpu",
) -> dict:
    """Integrated Gradients for one ECG output with lead- and time-level summaries."""
    if steps < 2:
        raise ValueError("steps must be at least 2.")
    x = torch.as_tensor(waveform, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 12:
        raise ValueError("waveform must have shape [12, samples] or [1, 12, samples].")
    base = torch.zeros_like(x) if baseline is None else torch.as_tensor(baseline, dtype=torch.float32, device=device)
    if base.ndim == 2:
        base = base.unsqueeze(0)
    if base.shape != x.shape:
        raise ValueError("baseline shape must match waveform shape.")
    was_training = model.training
    model.eval()
    alphas = torch.linspace(0.0, 1.0, steps, device=device)
    gradients = []
    try:
        # cuDNN RNN backward requires training-mode reserve space. Keep the
        # model in eval mode (stable dropout/normalization) and use PyTorch's
        # native RNN operations only for this input-gradient calculation.
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            for alpha in alphas:
                point = (base + alpha * (x - base)).detach().requires_grad_(True)
                output = model(point)
                if task not in output:
                    raise KeyError(f"ECG model has no {task!r} head.")
                logits = output[task]
                if target_index < 0 or target_index >= logits.shape[-1]:
                    raise IndexError(f"target_index {target_index} outside {task} head of size {logits.shape[-1]}.")
                score = logits[0, target_index]
                gradient = torch.autograd.grad(score, point, retain_graph=False, create_graph=False)[0]
                gradients.append(gradient)
        stacked = torch.stack(gradients, dim=0)
        average_gradient = torch.trapezoid(stacked, alphas, dim=0) / (alphas[-1] - alphas[0])
        attribution = (x - base) * average_gradient
        attribution_np = attribution.detach().cpu().numpy()[0]
        absolute = np.abs(attribution_np)
        lead_importance = absolute.sum(axis=1)
        time_importance = absolute.sum(axis=0)
        return {
            "method": "IntegratedGradients",
            "task": task,
            "target_index": int(target_index),
            "baseline": "zero_waveform",
            "steps": int(steps),
            "lead_importance": {f"lead_{index + 1}": float(value) for index, value in enumerate(lead_importance)},
            "temporal_importance": time_importance.astype(float).tolist(),
            "attribution_shape": list(attribution_np.shape),
            "warning": "Attributions indicate model sensitivity relative to the chosen baseline; they are not clinical localization or causal evidence.",
        }
    finally:
        model.train(was_training)
