import pytest
import torch

from ncpu_simplified.config import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
)
from ncpu_simplified.layout import InterleavedLayout, bits_to_integers, integers_to_bits


@pytest.mark.parametrize(
    "config",
    [
        GeometryConfig(bits=0),
        GeometryConfig(sx=0),
        GeometryConfig(sy=0),
        GeometryConfig(border_left=-1),
    ],
)
def test_invalid_geometry_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "config",
    [
        ModelConfig(channels=0),
        ModelConfig(hidden_size=0),
        ModelConfig(fixed_kernels=(), learnable_kernels=0),
        ModelConfig(learnable_kernels=-1),
        ModelConfig(learnable_kernel_init="invalid"),
        ModelConfig(gate="invalid"),
        ModelConfig(fire_rate=0),
        ModelConfig(fire_rate=1.01),
        ModelConfig(input_channel=3),
        ModelConfig(input_mode="invalid"),
        ModelConfig(channels=1, input_mode="frozen"),
    ],
)
def test_invalid_model_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config.validate()


def test_configuration_round_trip_preserves_tuples_and_values():
    original = ExperimentConfig(
        geometry=GeometryConfig(bits=6, sx=2),
        model=ModelConfig(
            fixed_kernels=("identity",),
            fixed_laplacian=True,
            learnable_kernels=3,
            learnable_kernel_init="random",
            gate="sigmoid",
        ),
        training=TrainingConfig(updates=10, free_steps=4, supervision_steps=5),
    )

    restored = ExperimentConfig.from_dict(original.to_dict())

    assert restored == original
    assert isinstance(restored.model.fixed_kernels, tuple)


def test_default_layout_matches_the_single_cell_geometry():
    layout = InterleavedLayout(GeometryConfig())

    assert (layout.height, layout.width) == (19, 13)
    assert layout.input_a_coordinates == ((6, 3), (9, 3), (12, 3), (15, 3))
    assert layout.input_b_coordinates == ((6, 6), (9, 6), (12, 6), (15, 6))
    assert layout.output_coordinates == ((3, 9), (6, 9), (9, 9), (12, 9), (15, 9))


def test_integer_bit_conversion_is_msb_first_and_invertible():
    values = torch.arange(16)

    bits = integers_to_bits(values, 4)

    assert bits.shape == (16, 4)
    assert torch.equal(bits[5], torch.tensor([0.0, 1.0, 0.0, 1.0]))
    assert torch.equal(bits_to_integers(bits), values)


def test_rendered_batches_are_exact_ternary_and_decode_to_sums():
    layout = InterleavedLayout(GeometryConfig(bits=3, sx=4, sy=2))
    operands_a = torch.arange(8).repeat_interleave(8)
    operands_b = torch.arange(8).repeat(8)

    inputs, targets = layout.render_batch(operands_a, operands_b)

    assert set(inputs.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert set(targets.unique().tolist()) == {-1.0, 0.0, 1.0}
    assert torch.equal(layout.decode_output(targets), operands_a + operands_b)
    assert torch.all(inputs[:, layout.input_mask() == 0] == 0)
    assert torch.all(targets[:, layout.output_mask() == 0] == 0)


def test_operand_pair_sampling_is_exhaustive_or_seeded_without_replacement():
    layout = InterleavedLayout(GeometryConfig(bits=2))

    all_a, all_b = layout.operand_pairs()
    sample_a, sample_b = layout.operand_pairs(max_pairs=7, seed=4)
    repeated_a, repeated_b = layout.operand_pairs(max_pairs=7, seed=4)

    assert len(all_a) == 16
    assert len(set(zip(all_a.tolist(), all_b.tolist()))) == 16
    assert len(set(zip(sample_a.tolist(), sample_b.tolist()))) == 7
    assert torch.equal(sample_a, repeated_a)
    assert torch.equal(sample_b, repeated_b)


def test_output_has_one_more_row_than_each_input():
    for bits in (1, 4, 8):
        layout = InterleavedLayout(GeometryConfig(bits=bits))
        assert len(layout.input_a_coordinates) == bits
        assert len(layout.input_b_coordinates) == bits
        assert len(layout.output_coordinates) == bits + 1
        assert layout.output_coordinates[0][0] < layout.input_a_coordinates[0][0]
