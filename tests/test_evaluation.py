import pytest

from ncpu_simplified import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
)
from ncpu_simplified.evaluation import evaluate, evaluate_widths, format_results
from ncpu_simplified.model import NeuralCellularAutomaton


def evaluation_config():
    return ExperimentConfig(
        geometry=GeometryConfig(bits=2),
        model=ModelConfig(hidden_size=8),
        training=TrainingConfig(
            updates=2,
            batch_size=4,
            free_steps=1,
            supervision_steps=1,
            device="cpu",
        ),
    )


def test_zero_step_untrained_evaluation_has_known_exact_accuracy():
    config = evaluation_config()
    model = NeuralCellularAutomaton(config.model)

    result = evaluate(
        model,
        config.geometry,
        steps=0,
        step_start=0,
        step_end=0,
        batch_size=5,
    )

    assert result.pairs == 16
    assert result.mean_mse == pytest.approx(1.0)
    assert result.mean_exact_accuracy == pytest.approx(1 / 16)
    assert result.stable_accuracy == pytest.approx(1 / 16)
    assert result.best_exact_step == 0


def test_evaluation_sampling_is_deterministic_and_bounded():
    config = evaluation_config()
    model = NeuralCellularAutomaton(config.model)

    first = evaluate(
        model,
        config.geometry,
        steps=1,
        step_start=0,
        step_end=1,
        batch_size=3,
        max_pairs=7,
        seed=12,
    )
    second = evaluate(
        model,
        config.geometry,
        steps=1,
        step_start=0,
        step_end=1,
        batch_size=4,
        max_pairs=7,
        seed=12,
    )

    assert first == second
    assert first.pairs == 7


def test_width_evaluation_reuses_one_rule_on_larger_grids():
    config = evaluation_config()
    model = NeuralCellularAutomaton(config.model)

    results = evaluate_widths(
        model,
        config,
        widths=(1, 2, 3),
        steps=1,
        step_start=0,
        step_end=1,
        batch_size=8,
    )

    assert [result.bits for result in results] == [1, 2, 3]
    assert [result.pairs for result in results] == [4, 16, 64]
    assert "best step" in format_results(results)


@pytest.mark.parametrize(
    "start,end,steps",
    [(-1, 0, 1), (1, 0, 1), (0, 2, 1)],
)
def test_invalid_evaluation_windows_are_rejected(start, end, steps):
    config = evaluation_config()
    with pytest.raises(ValueError):
        evaluate(
            NeuralCellularAutomaton(config.model),
            config.geometry,
            steps=steps,
            step_start=start,
            step_end=end,
        )
