from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .config import ExperimentConfig, GeometryConfig
from .layout import InterleavedLayout
from .model import NeuralCellularAutomaton


@dataclass(frozen=True)
class EvaluationResult:
    bits: int
    pairs: int
    step_start: int
    step_end: int
    mean_mse: float
    mean_exact_accuracy: float
    mean_bit_accuracy: float
    stable_accuracy: float
    best_exact_step: int
    best_exact_accuracy: float
    best_mse_step: int
    best_mse: float
    mse_by_step: tuple[float, ...]
    exact_by_step: tuple[float, ...]
    bit_by_step: tuple[float, ...]

    def summary(self) -> dict[str, float | int | str]:
        return {
            "bits": self.bits,
            "pairs": self.pairs,
            "steps": f"{self.step_start}-{self.step_end}",
            "mse": self.mean_mse,
            "exact": self.mean_exact_accuracy,
            "bit": self.mean_bit_accuracy,
            "stable": self.stable_accuracy,
            "best_exact_step": self.best_exact_step,
            "best_exact": self.best_exact_accuracy,
            "best_mse_step": self.best_mse_step,
            "best_mse": self.best_mse,
        }


@torch.no_grad()
def evaluate(
    model: NeuralCellularAutomaton,
    geometry: GeometryConfig,
    *,
    steps: int,
    step_start: int,
    step_end: int,
    batch_size: int = 256,
    max_pairs: int | None = None,
    seed: int = 0,
) -> EvaluationResult:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not 0 <= step_start <= step_end <= steps:
        raise ValueError("evaluation steps must satisfy 0 <= start <= end <= steps")
    layout = InterleavedLayout(geometry)
    operands_a, operands_b = layout.operand_pairs(max_pairs=max_pairs, seed=seed)
    step_count = steps + 1
    mse_sums = torch.zeros(step_count, dtype=torch.float64)
    exact_sums = torch.zeros(step_count, dtype=torch.float64)
    bit_sums = torch.zeros(step_count, dtype=torch.float64)
    stable_sum = 0.0
    pair_count = len(operands_a)
    output_bits = layout.bits + 1
    device = model.device
    was_training = model.training
    model.eval()

    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for offset in range(0, pair_count, batch_size):
                batch_a = operands_a[offset : offset + batch_size]
                batch_b = operands_b[offset : offset + batch_size]
                inputs, targets = layout.render_batch(batch_a, batch_b)
                inputs = inputs.to(device)
                targets = targets.to(device)
                rollout = model(model.initial_state(inputs), steps)
                values = layout.read_output_values(
                    rollout[:, :, model.config.output_channel]
                )
                expected = layout.read_output_values(targets).unsqueeze(1)
                correct_bits = (values > 0) == (expected > 0)
                exact = correct_bits.all(dim=-1)
                mse_sums += (values - expected).square().sum(dim=(0, 2)).cpu()
                exact_sums += exact.sum(dim=0).cpu()
                bit_sums += correct_bits.sum(dim=(0, 2)).cpu()
                stable_sum += float(
                    exact[:, step_start : step_end + 1].all(dim=1).sum()
                )
    finally:
        model.train(was_training)
    mse_curve = mse_sums / (pair_count * output_bits)
    exact_curve = exact_sums / pair_count
    bit_curve = bit_sums / (pair_count * output_bits)
    window = slice(step_start, step_end + 1)
    exact_window = exact_curve[window]
    mse_window = mse_curve[window]
    best_exact_offset = int(exact_window.argmax())
    best_mse_offset = int(mse_window.argmin())
    return EvaluationResult(
        bits=layout.bits,
        pairs=pair_count,
        step_start=step_start,
        step_end=step_end,
        mean_mse=float(mse_window.mean()),
        mean_exact_accuracy=float(exact_window.mean()),
        mean_bit_accuracy=float(bit_curve[window].mean()),
        stable_accuracy=stable_sum / pair_count,
        best_exact_step=step_start + best_exact_offset,
        best_exact_accuracy=float(exact_window[best_exact_offset]),
        best_mse_step=step_start + best_mse_offset,
        best_mse=float(mse_window[best_mse_offset]),
        mse_by_step=tuple(map(float, mse_curve)),
        exact_by_step=tuple(map(float, exact_curve)),
        bit_by_step=tuple(map(float, bit_curve)),
    )


def evaluate_widths(
    model: NeuralCellularAutomaton,
    config: ExperimentConfig,
    widths: tuple[int, ...] = (4, 6, 8),
    *,
    steps: int | None = None,
    step_start: int | None = None,
    step_end: int | None = None,
    batch_size: int = 256,
    max_pairs: int | None = None,
    seed: int = 0,
) -> list[EvaluationResult]:
    if not widths or any(width < 1 for width in widths):
        raise ValueError("widths must contain positive integers")
    steps = config.training.rollout_steps if steps is None else steps
    step_start = config.training.supervision_start if step_start is None else step_start
    step_end = config.training.supervision_end if step_end is None else step_end
    return [
        evaluate(
            model,
            replace(config.geometry, bits=width),
            steps=steps,
            step_start=step_start,
            step_end=step_end,
            batch_size=batch_size,
            max_pairs=max_pairs,
            seed=seed,
        )
        for width in widths
    ]


def format_results(results: list[EvaluationResult]) -> str:
    header = "bits  pairs   steps      mse       exact      bit      stable  best step"
    lines = [header]
    for result in results:
        lines.append(
            f"{result.bits:>4}  {result.pairs:>5}  "
            f"{result.step_start:>3}-{result.step_end:<3}  "
            f"{result.mean_mse:>9.6f}  "
            f"{result.mean_exact_accuracy:>8.2%}  "
            f"{result.mean_bit_accuracy:>7.2%}  "
            f"{result.stable_accuracy:>8.2%}  "
            f"{result.best_exact_step:>4} ({result.best_exact_accuracy:.2%})"
        )
    return "\n".join(lines)
