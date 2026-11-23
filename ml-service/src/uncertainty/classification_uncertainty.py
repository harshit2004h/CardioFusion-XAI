from __future__ import annotations

import numpy as np
import torch


def binary_predictive_entropy(probabilities):
    """Return Bernoulli predictive entropy in nats, without calling it confidence."""
    p = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("probabilities must be finite and within [0, 1].")
    clipped = np.clip(p, 1e-12, 1.0 - 1e-12)
    entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
    return float(entropy) if entropy.ndim == 0 else entropy


def monte_carlo_dropout(model, inputs: torch.Tensor, forward_fn=None, samples: int = 20) -> dict[str, np.ndarray]:
    """Estimate stochastic forward-pass dispersion for dropout-enabled models.

    BatchNorm layers are kept in evaluation mode; only dropout modules are enabled.
    The model's original training/evaluation state is restored on completion.
    """
    if samples < 2:
        raise ValueError("samples must be at least 2.")
    was_training = model.training
    model.eval()
    dropout_types = (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d, torch.nn.AlphaDropout)
    dropout_states = {module: module.training for module in model.modules() if isinstance(module, dropout_types)}
    try:
        for module in dropout_states:
            module.train(True)
        outputs = []
        with torch.no_grad():
            for _ in range(samples):
                raw = forward_fn(model, inputs) if forward_fn else model(inputs)
                if isinstance(raw, dict):
                    if "diagnostic" in raw:
                        raw = torch.sigmoid(raw["diagnostic"])
                    elif "logits" in raw:
                        raw = torch.sigmoid(raw["logits"])
                    else:
                        raise ValueError("forward_fn must return a tensor or a recognized logits mapping.")
                outputs.append(raw.detach().cpu().numpy())
        values = np.stack(outputs, axis=0)
        return {"mean": values.mean(axis=0), "standard_deviation": values.std(axis=0, ddof=1), "n_samples": np.asarray(samples)}
    finally:
        for module, state in dropout_states.items():
            module.train(state)
        model.train(was_training)
