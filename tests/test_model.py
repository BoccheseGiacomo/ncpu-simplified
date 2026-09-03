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


@pytest.mark.parametrize("channels", [2, 3, 5])
@pytest.mark.parametrize("fire_rate", [1.0, 0.5])
def test_frozen_input_is_preserved_even_with_state_clipping(channels, fire_rate):
    config = ModelConfig(
        channels=channels, input_mode="frozen", max_abs_state=0.5, fire_rate=fire_rate
    )
    model = NeuralCellularAutomaton(config)
    mutable = NeuralCellularAutomaton(ModelConfig(channels=channels))
    assert model.parameter_count == mutable.parameter_count
    inputs = torch.tensor([[[-1.0, 1.0], [1.0, -1.0]]])
    initial = model.initial_state(inputs)
    assert initial.shape[1] == channels
    assert torch.count_nonzero(initial[:, 1:]) == 0
    with torch.no_grad():
        model.rule.output.weight.fill_(0.2)
    rollout = model(initial, 3)
    assert torch.equal(rollout[:, :, 0], inputs.unsqueeze(1).expand(-1, 4, -1, -1))
    assert torch.all(rollout[:, :, 1:].abs() <= 0.5)


def test_default_fire_rate_updates_every_cell_without_sampling(monkeypatch):
    model = NeuralCellularAutomaton(ModelConfig())
    state = torch.zeros(2, 3, 2, 2)

    def constant_delta(perception):
        return torch.ones(
            perception.shape[0],
            model.config.channels,
            perception.shape[2],
            perception.shape[3],
            device=perception.device,
            dtype=perception.dtype,
        )

    monkeypatch.setattr(model.rule, "forward", constant_delta)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *args, **kwargs: pytest.fail("synchronous updates must not sample"),
    )

    assert model.config.fire_rate == 1.0
    assert torch.equal(model.step(state), torch.ones_like(state))


def test_asynchronous_fire_rate_updates_whole_cells_independently(monkeypatch):
    model = NeuralCellularAutomaton(ModelConfig(fire_rate=0.5))
    state = torch.zeros(2, 3, 2, 2)
    draws = torch.tensor(
        [
            [[[0.1, 0.9], [0.8, 0.2]]],
            [[[0.7, 0.3], [0.4, 0.6]]],
        ]
    )

    def constant_delta(perception):
        return torch.ones(
            perception.shape[0],
            model.config.channels,
            perception.shape[2],
            perception.shape[3],
            device=perception.device,
            dtype=perception.dtype,
        )

    def fixed_draws(*shape, device=None):
        assert shape == (2, 1, 2, 2)
        return draws.to(device=device)

    monkeypatch.setattr(model.rule, "forward", constant_delta)
    monkeypatch.setattr(torch, "rand", fixed_draws)

    expected = (draws < 0.5).expand_as(state).to(state.dtype)
    assert torch.equal(model.step(state), expected)


def test_perception_preserves_grid_size_for_every_padding_mode():
    state = torch.randn(2, 3, 8, 9)
    for padding in ("zeros", "reflect", "replicate", "circular"):
        output = Perception(ModelConfig(padding=padding))(state)
        assert output.shape == (2, 12, 8, 9)
