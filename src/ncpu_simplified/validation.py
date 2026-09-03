from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from .config import ExperimentConfig
from .layout import InterleavedLayout
from .model import NeuralCellularAutomaton, format_bytes, model_memory_bytes
from .training import load_model, supervised_loss


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    config: ExperimentConfig
    schema: str
    parameter_count: int
    memory: dict[str, int]
    checks: tuple[ValidationCheck, ...]

    @property
    def successful(self) -> bool:
        return all(check.passed for check in self.checks)

    def require_success(self) -> None:
        failures = [check for check in self.checks if not check.passed]
        if failures:
            details = "; ".join(f"{check.name}: {check.detail}" for check in failures)
            raise RuntimeError(f"validation failed: {details}")

    def __str__(self) -> str:
        geometry = self.config.geometry
        model = self.config.model
        training = self.config.training
        layout = InterleavedLayout(geometry)
        memory = ", ".join(
            f"{name}={format_bytes(value)}" for name, value in self.memory.items()
        )
        checks = "\n".join(
            f"  [{'ok' if check.passed else 'FAIL'}] {check.name}: {check.detail}"
            for check in self.checks
        )
        return (
            f"{self.schema}\n\n"
            f"grid: {geometry.bits}-bit, "
            f"{layout.height}x{layout.width}, "
            f"stride=({geometry.sx}, {geometry.sy})\n"
            f"model: {model.channels} channels, hidden={model.hidden_size}, "
            f"gate={model.gate}, input={model.input_channel}/{model.input_mode}, "
            f"output={model.output_channel}\n"
            f"fixed kernels: {', '.join(model.fixed_kernels) or 'none'}"
            f"{' + laplacian' if model.fixed_laplacian else ''}\n"
            f"learnable kernels: {model.learnable_kernels} "
            f"({model.learnable_kernel_init} initialization)\n"
            f"time: {training.free_steps} free + "
            f"{training.supervision_steps} supervised "
            f"(steps {training.supervision_start}-{training.supervision_end})\n"
            f"parameters: {self.parameter_count}\n"
            f"memory (raw tensors): {memory}\n"
            f"validation: {'PASSED' if self.successful else 'FAILED'}\n{checks}"
        )


def validate(
    config: ExperimentConfig,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
) -> ValidationReport:
    checks: list[ValidationCheck] = []

    def check(name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception as error:
            checks.append(
                ValidationCheck(name, False, f"{type(error).__name__}: {error}")
            )
        else:
            checks.append(ValidationCheck(name, True, "passed"))

    check("configuration", config.validate)
    config.validate()
    layout = InterleavedLayout(config.geometry)
    model = NeuralCellularAutomaton(config.model).to(device)

    def geometry_check() -> None:
        all_coordinates = (
            layout.input_a_coordinates
            + layout.input_b_coordinates
            + layout.output_coordinates
        )
        if len(set(all_coordinates)) != len(all_coordinates):
            raise AssertionError("input and output cells overlap")
        if not all(
            0 <= row < layout.height and 0 <= column < layout.width
            for row, column in all_coordinates
        ):
            raise AssertionError("a signal cell is outside the grid")
        if layout.output_coordinates[0][0] >= layout.input_a_coordinates[0][0]:
            raise AssertionError("the carry row must be above the first input row")

    check("geometry", geometry_check)

    inputs, targets = layout.render_batch(
        torch.tensor([0, (1 << layout.bits) - 1]),
        torch.tensor([(1 << layout.bits) - 1, (1 << layout.bits) - 1]),
    )

    def encoding_check() -> None:
        if not set(inputs.unique().tolist()) <= {-1.0, 0.0, 1.0}:
            raise AssertionError("input encoding is not ternary")
        if not set(targets.unique().tolist()) <= {-1.0, 0.0, 1.0}:
            raise AssertionError("target encoding is not ternary")
        expected = torch.tensor([(1 << layout.bits) - 1, 2 * ((1 << layout.bits) - 1)])
        if not torch.equal(layout.decode_output(targets), expected):
            raise AssertionError("output decoding does not recover exact sums")
        if not torch.all(inputs[:, layout.input_mask() == 0] == 0):
            raise AssertionError("background cells are not neutral")

    check("ternary encoding and arithmetic", encoding_check)

    def forward_check() -> None:
        state = model.initial_state(inputs.to(device))
        rollout = model(state, 3)
        expected_shape = (
            len(inputs),
            4,
            config.model.channels,
            layout.height,
            layout.width,
        )
        if rollout.shape != expected_shape:
            raise AssertionError(
                f"expected {expected_shape}, received {tuple(rollout.shape)}"
            )
        if not torch.isfinite(rollout).all():
            raise AssertionError("forward evolution produced non-finite values")
        if not torch.equal(state[:, config.model.input_channel], inputs.to(device)):
            raise AssertionError("inputs were not implanted in the configured channel")
        other_channels = [
            channel
            for channel in range(config.model.channels)
            if channel != config.model.input_channel
        ]
        if other_channels and torch.count_nonzero(state[:, other_channels]):
            raise AssertionError("non-input channels do not start neutral")

    check("state construction and forward evolution", forward_check)

    def mutability_check() -> None:
        expected = 0.0 if config.model.input_mode == "frozen" else 1.0
        actual = float(model.update_mask[0, config.model.input_channel, 0, 0])
        if actual != expected:
            raise AssertionError("input-channel update mask contradicts input_mode")
        if float(model.update_mask[0, config.model.output_channel, 0, 0]) != 1.0:
            raise AssertionError("the output channel must be mutable")

    check("input-channel mutability", mutability_check)

    def loss_window_check() -> None:
        sequence = torch.zeros(
            1,
            config.training.rollout_steps + 1,
            config.model.channels,
            layout.height,
            layout.width,
        )
        target = targets[:1]
        start = config.training.supervision_start
        end = config.training.supervision_end + 1
        sequence[:, start:end, config.model.output_channel] = target.unsqueeze(1)
        loss = supervised_loss(
            sequence,
            target,
            layout,
            config.model.output_channel,
            config.training.free_steps,
            config.training.supervision_steps,
        )
        if loss.item() != 0.0:
            raise AssertionError("supervision window includes unintended states")

    check("supervision window", loss_window_check)

    def gradient_check() -> None:
        with torch.random.fork_rng():
            torch.manual_seed(1729)
            # Test the readout gradient independently of stochastic cell firing.
            probe = NeuralCellularAutomaton(replace(config.model, fire_rate=1.0)).to(
                device
            )
            initial = torch.rand(
                1,
                config.model.channels,
                layout.height,
                layout.width,
                device=device,
            ).sub(0.5)
        rollout = probe(initial, 1)
        loss = supervised_loss(
            rollout, targets[:1].to(device), layout, config.model.output_channel, 0, 1
        )
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in probe.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients or not all(
            torch.isfinite(gradient).all() for gradient in gradients
        ):
            raise AssertionError("backpropagation produced invalid gradients")
        if sum(float(gradient.abs().sum()) for gradient in gradients) == 0.0:
            raise AssertionError("all gradients are zero")

    check("finite non-zero gradients", gradient_check)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)

        def checkpoint_check() -> None:
            loaded_model, loaded_config, _ = load_model(checkpoint_path, device=device)
            if loaded_config != config:
                raise AssertionError("checkpoint configuration does not match")
            loaded_model(model.initial_state(inputs[:1].to(device)), 1)

        check("checkpoint", checkpoint_check)

    memory = model_memory_bytes(
        model,
        config.training.batch_size,
        layout.height,
        layout.width,
        config.training.rollout_steps,
    )
    return ValidationReport(
        config=config,
        schema=layout.schema(),
        parameter_count=model.parameter_count,
        memory=memory,
        checks=tuple(checks),
    )
