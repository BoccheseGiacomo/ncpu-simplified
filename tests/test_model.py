import pytest
import torch

from ncpu_simplified.config import ModelConfig
from ncpu_simplified.model import NeuralCellularAutomaton, Perception, perception_kernel


def test_default_model_has_the_expected_architecture_and_parameter_count():
    model = NeuralCellularAutomaton(ModelConfig())

    assert model.perception.kernel_count == 4
    assert model.rule.hidden.in_channels == 12
    assert model.rule.hidden.out_channels == 57
    assert model.rule.output.out_channels == 3
    assert model.parameter_count == 921


def test_fixed_laplacian_can_be_used_without_learnable_kernels():
    config = ModelConfig(fixed_laplacian=True, learnable_kernels=0)
    perception = Perception(config)

    assert perception.kernel_count == 4
    assert torch.equal(perception.kernel_bank()[-1], perception_kernel("laplacian"))
    assert not getattr(perception, "fixed_3").requires_grad


def test_k_laplacian_initialized_kernels_are_independently_trainable():
    perception = Perception(
        ModelConfig(
            fixed_kernels=(), learnable_kernels=3, learnable_kernel_init="laplacian"
        )
    )

    assert perception.kernel_bank().shape == (3, 3, 3)
    assert all(
        torch.equal(kernel, perception_kernel("laplacian"))
        for kernel in perception.kernel_bank()
    )
    assert sum(parameter.numel() for parameter in perception.parameters()) == 27


def test_random_kernel_bank_is_reproducible_and_does_not_consume_global_rng():
    config = ModelConfig(
        fixed_kernels=(),
        learnable_kernels=3,
        learnable_kernel_init="random",
        random_kernel_seed=12,
    )
    torch.manual_seed(9)
    expected = torch.rand(4)
    torch.manual_seed(9)
    first = Perception(config)
    observed = torch.rand(4)
    second = Perception(config)

    assert torch.equal(expected, observed)
    assert torch.equal(first.kernel_bank(), second.kernel_bank())
    assert not torch.equal(first.kernel_bank()[0], first.kernel_bank()[1])


@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
def test_every_gate_has_correct_shape_and_identity_at_initialization(gate):
    model = NeuralCellularAutomaton(ModelConfig(gate=gate))
    initial = torch.randn(2, 3, 7, 9)

    rollout = model(initial, 3)

    assert rollout.shape == (2, 4, 3, 7, 9)
    assert torch.equal(rollout[:, 0], initial)
    assert torch.equal(rollout[:, -1], initial)
    expected_outputs = 3 if gate == "none" else 6
    assert model.rule.output.out_channels == expected_outputs


def test_frozen_input_channel_cannot_be_updated():
    model = NeuralCellularAutomaton(ModelConfig(input_mode="frozen"))
    with torch.no_grad():
        model.rule.output.weight[:3].fill_(0.1)
    initial = torch.randn(2, 3, 8, 8)

    evolved = model(initial, 2)[:, -1]

    assert torch.equal(evolved[:, 0], initial[:, 0])
    assert not torch.equal(evolved[:, 1], initial[:, 1])


def test_mutable_input_channel_can_evolve():
    model = NeuralCellularAutomaton(ModelConfig(input_mode="mutable"))
    with torch.no_grad():
        model.rule.output.weight[:3].fill_(0.1)
    initial = torch.randn(2, 3, 8, 8)

    evolved = model(initial, 1)[:, -1]

    assert not torch.equal(evolved[:, 0], initial[:, 0])


def test_perception_preserves_grid_size_for_every_padding_mode():
    state = torch.randn(2, 3, 8, 9)
    for padding in ("zeros", "reflect", "replicate", "circular"):
        output = Perception(ModelConfig(padding=padding))(state)
        assert output.shape == (2, 12, 8, 9)
