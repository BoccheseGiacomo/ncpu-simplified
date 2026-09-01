from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GeometryConfig


def integers_to_bits(values: torch.Tensor, width: int) -> torch.Tensor:
    if width < 1:
        raise ValueError("width must be positive")
    values = torch.as_tensor(values, dtype=torch.int64)
    if torch.any(values < 0) or torch.any(values >= 1 << width):
        raise ValueError(f"values must be in [0, {1 << width})")
    shifts = torch.arange(width - 1, -1, -1, device=values.device)
    return ((values.unsqueeze(-1) >> shifts) & 1).to(torch.float32)


def bits_to_integers(bits: torch.Tensor) -> torch.Tensor:
    bits = torch.as_tensor(bits)
    if bits.ndim < 1 or bits.shape[-1] < 1:
        raise ValueError("bits must have a non-empty final dimension")
    if not torch.all((bits == 0) | (bits == 1)):
        raise ValueError("bits must contain only zero and one")
    width = bits.shape[-1]
    weights = 1 << torch.arange(width - 1, -1, -1, device=bits.device)
    return (bits.to(torch.int64) * weights).sum(dim=-1)


@dataclass(frozen=True)
class InterleavedLayout:
    config: GeometryConfig

    def __post_init__(self) -> None:
        self.config.validate()

    @property
    def bits(self) -> int:
        return self.config.bits

    @property
    def width(self) -> int:
        return (
            self.config.border_left + 2 * self.config.sx + 1 + self.config.border_right
        )

    @property
    def height(self) -> int:
        return (
            self.config.border_top
            + self.bits * self.config.sy
            + 1
            + self.config.border_bottom
        )

    @property
    def input_a_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self.config.border_top + (index + 1) * self.config.sy,
                self.config.border_left,
            )
            for index in range(self.bits)
        )

    @property
    def input_b_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self.config.border_top + (index + 1) * self.config.sy,
                self.config.border_left + self.config.sx,
            )
            for index in range(self.bits)
        )

    @property
    def output_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self.config.border_top + index * self.config.sy,
                self.config.border_left + 2 * self.config.sx,
            )
            for index in range(self.bits + 1)
        )

    def _mask(self, coordinates: tuple[tuple[int, int], ...]) -> torch.Tensor:
        mask = torch.zeros(self.height, self.width)
        rows, columns = zip(*coordinates)
        mask[list(rows), list(columns)] = 1.0
        return mask

    def input_mask(self) -> torch.Tensor:
        return self._mask(self.input_a_coordinates + self.input_b_coordinates)

    def output_mask(self) -> torch.Tensor:
        return self._mask(self.output_coordinates)

    def render_batch(
        self, operands_a: torch.Tensor, operands_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operands_a = torch.as_tensor(operands_a, dtype=torch.int64).flatten()
        operands_b = torch.as_tensor(operands_b, dtype=torch.int64).flatten()
        if operands_a.shape != operands_b.shape:
            raise ValueError("operand batches must have the same shape")
        if operands_a.device != operands_b.device:
            raise ValueError("operand batches must be on the same device")
        a_bits = integers_to_bits(operands_a, self.bits)
        b_bits = integers_to_bits(operands_b, self.bits)
        sums = operands_a + operands_b
        sum_bits = integers_to_bits(sums, self.bits + 1)

        inputs = torch.zeros(
            len(operands_a),
            self.height,
            self.width,
            device=operands_a.device,
            dtype=torch.float32,
        )
        targets = torch.zeros_like(inputs)
        a_rows, a_columns = zip(*self.input_a_coordinates)
        b_rows, b_columns = zip(*self.input_b_coordinates)
        out_rows, out_columns = zip(*self.output_coordinates)
        inputs[:, list(a_rows), list(a_columns)] = a_bits.mul(2).sub(1)
        inputs[:, list(b_rows), list(b_columns)] = b_bits.mul(2).sub(1)
        targets[:, list(out_rows), list(out_columns)] = sum_bits.mul(2).sub(1)
        return inputs, targets

    def render(
        self, operand_a: int, operand_b: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.render_batch(
            torch.tensor([operand_a]), torch.tensor([operand_b])
        )
        return inputs[0], targets[0]

    def read_output_values(self, state: torch.Tensor) -> torch.Tensor:
        rows, columns = zip(*self.output_coordinates)
        return state[..., list(rows), list(columns)]

    def decode_output(self, state: torch.Tensor) -> torch.Tensor:
        return bits_to_integers(self.read_output_values(state) > 0)

    def operand_pairs(
        self, max_pairs: int | None = None, seed: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operand_count = 1 << self.bits
        total = operand_count * operand_count
        if max_pairs is None or max_pairs >= total:
            indices = torch.arange(total)
        else:
            if max_pairs < 1:
                raise ValueError("max_pairs must be positive or None")
            generator = torch.Generator().manual_seed(seed)
            indices = torch.randperm(total, generator=generator)[:max_pairs]
        return (
            indices.div(operand_count, rounding_mode="floor"),
            indices % operand_count,
        )

    def schema(self) -> str:
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        for row, column in self.input_a_coordinates:
            cells[row][column] = "A"
        for row, column in self.input_b_coordinates:
            cells[row][column] = "B"
        for row, column in self.output_coordinates:
            cells[row][column] = "O"
        lines = [" ".join(row) for row in cells]
        lines.append(
            "A/B: operands (MSB at top), O: output (carry row at top), .: neutral"
        )
        return "\n".join(lines)
