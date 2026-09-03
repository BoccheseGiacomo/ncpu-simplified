from dataclasses import replace
import sys
from types import SimpleNamespace

import pytest
import torch

from ncpu_simplified import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
)
from ncpu_simplified.evaluation import evaluate
from ncpu_simplified.layout import InterleavedLayout
from ncpu_simplified.training import (
    StepMetrics,
    Trainer,
    WandbReporter,
    cosine_learning_rate,
    load_model,
    supervised_loss,
    train_seeds,
)


def tiny_config(**training_changes):
    training = TrainingConfig(
        updates=4,
        batch_size=4,
        free_steps=1,
        supervision_steps=1,
        learning_rate=1e-3,
        final_learning_rate=1e-4,
        validation_every=2,
        checkpoint_every=2,
        device="cpu",
        **training_changes,
    )
    return ExperimentConfig(
        geometry=GeometryConfig(
            bits=1,
            sx=1,
            sy=1,
            border_left=1,
            border_right=1,
            border_top=1,
            border_bottom=1,
        ),
        model=ModelConfig(channels=3, hidden_size=8),
        training=training,
    )


def test_supervised_loss_uses_only_the_post_free_window():
    config = tiny_config()
    layout = InterleavedLayout(config.geometry)
    _, target = layout.render_batch(torch.tensor([1]), torch.tensor([1]))
    rollout = torch.full((1, 4, 3, layout.height, layout.width), 17.0)
    rollout[:, 2:3, 0] = target.unsqueeze(1)

    loss = supervised_loss(
        rollout, target, layout, 0, free_steps=1, supervision_steps=1
    )

    assert loss.item() == 0.0


def test_cosine_schedule_reaches_both_endpoints_and_warms_up():
    config = tiny_config(warmup_updates=2)

    assert cosine_learning_rate(config, 0) == pytest.approx(5e-4)
    assert cosine_learning_rate(config, 1) == pytest.approx(1e-3)
    assert cosine_learning_rate(config, 3) == pytest.approx(1e-4)


def test_training_is_reproducible_with_an_isolated_data_generator():
    config = tiny_config()
    first = Trainer(config)
    second = Trainer(config)

    for _ in range(2):
        first.train_step()
        torch.rand(1000)
        second.train_step()

    for first_parameter, second_parameter in zip(
        first.model.parameters(), second.model.parameters()
    ):
        assert torch.equal(first_parameter, second_parameter)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
@pytest.mark.parametrize("fire_rate", [1.0, 0.5])
def test_checkpoint_resume_matches_uninterrupted_training(tmp_path, device, fire_rate):
    base = tiny_config()
    config = replace(
        base,
        model=replace(base.model, fire_rate=fire_rate),
        training=replace(base.training, device=device),
    )
    uninterrupted = Trainer(config)
    for _ in range(4):
        uninterrupted.train_step()

    interrupted = Trainer(config)
    for _ in range(2):
        interrupted.train_step()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save(checkpoint)
    expected_data = torch.rand(4, generator=interrupted.data_generator)
    expected_cpu = torch.rand(4)
    expected_device = torch.rand(4, device=device)
    resumed = Trainer.from_checkpoint(checkpoint, device=device)
    assert torch.equal(torch.rand(4, generator=resumed.data_generator), expected_data)
    assert torch.equal(torch.rand(4), expected_cpu)
    assert torch.equal(torch.rand(4, device=device), expected_device)
    resumed = Trainer.from_checkpoint(checkpoint)
    assert resumed.device.type == device
    for _ in range(2):
        resumed.train_step()

    assert resumed.current_update == uninterrupted.current_update == 4
    for expected, actual in zip(
        uninterrupted.model.state_dict().values(), resumed.model.state_dict().values()
    ):
        assert torch.equal(expected, actual)


def test_short_fit_writes_loadable_best_and_latest_checkpoints(tmp_path):
    config = tiny_config()
    trainer = Trainer(config)

    history = trainer.fit(tmp_path, progress_every=10)

    assert len(history) == config.training.updates
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()
    restored = Trainer.from_checkpoint(tmp_path / "latest.pt", device="cpu")
    assert restored.current_update == config.training.updates
    assert len(restored.history) == config.training.updates


@pytest.mark.parametrize("input_mode", ["mutable", "frozen"])
def test_batched_exhaustive_validation_matches_evaluator(input_mode):
    base = tiny_config()
    config = replace(base, model=replace(base.model, input_mode=input_mode))
    trainer = Trainer(config)
    with torch.no_grad():
        trainer.model.rule.hidden.weight.zero_()
        trainer.model.rule.hidden.bias.fill_(1.0)
        trainer.model.rule.output.weight[config.model.output_channel].fill_(0.01)

    validation_loss = trainer.validation_loss()
    result = evaluate(
        trainer.model,
        config.geometry,
        steps=config.training.rollout_steps,
        step_start=config.training.supervision_start,
        step_end=config.training.supervision_end,
        batch_size=3,
        seed=config.training.seed,
    )

    assert validation_loss == pytest.approx(result.mean_mse)
    assert validation_loss != pytest.approx(1.0)


@pytest.mark.parametrize("channels", [2, 3, 5])
def test_frozen_training_updates_the_output_with_nonzero_task_gradients(
    channels, monkeypatch
):
    base = tiny_config()
    trainer = Trainer(
        replace(base, model=replace(base.model, channels=channels, input_mode="frozen"))
    )
    inputs, targets = trainer.layout.render_batch(*trainer.layout.operand_pairs())
    monkeypatch.setattr(trainer, "_sample_batch", lambda: (inputs, targets))
    before = trainer.model.rule.output.weight.detach().clone()
    metric = trainer.train_step()
    assert metric.gradient_norm > 0
    after = trainer.model.rule.output.weight
    assert torch.equal(before[0], after[0])
    assert not torch.equal(before[1], after[1])
    assert trainer.model.rule.output.weight.grad[1].abs().sum() > 0


def test_multiple_seeds_share_hyperparameters_and_select_one_best(tmp_path):
    config = tiny_config()

    results = train_seeds(config, (3, 7), tmp_path, progress_every=10)

    assert [result.seed for result in results] == [3, 7]
    assert all(result.checkpoint.is_file() for result in results)
    assert (tmp_path / "best.pt").is_file()
    loaded_configs = [
        load_model(result.checkpoint, device="cpu")[1] for result in results
    ]
    assert loaded_configs[0].geometry == loaded_configs[1].geometry == config.geometry
    assert loaded_configs[0].model == loaded_configs[1].model == config.model
    assert (
        replace(loaded_configs[0].training, seed=0)
        == replace(loaded_configs[1].training, seed=0)
        == config.training
    )


@pytest.mark.parametrize("seeds", [(), (1, 1), (-1,)])
def test_multiple_seed_validation_rejects_invalid_seed_sets(tmp_path, seeds):
    with pytest.raises(ValueError):
        train_seeds(tiny_config(), seeds, tmp_path)


def test_wandb_reporter_is_optional_and_logs_one_step(monkeypatch):
    events = []

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, values, step):
            events.append(("log", values, step))

        def finish(self):
            events.append(("finish", self.summary.copy()))

    fake_run = FakeRun()
    fake_wandb = SimpleNamespace(
        init=lambda **kwargs: events.append(("init", kwargs)) or fake_run
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    reporter = WandbReporter(
        "project",
        tiny_config(),
        run_name="seed-3",
        group="shared-config",
    )

    reporter(
        StepMetrics(
            update=1,
            loss=0.5,
            learning_rate=1e-3,
            gradient_norm=0.2,
            exact_accuracy=0.25,
        )
    )
    reporter.finish(0.4)

    assert events[0][0] == "init"
    assert events[0][1]["name"] == "seed-3"
    assert events[1][0] == "log" and events[1][2] == 1
    assert events[2] == ("finish", {"best_validation_loss": 0.4})


def test_short_training_is_finite_and_changes_parameters():
    trainer = Trainer(tiny_config())
    before = [parameter.detach().clone() for parameter in trainer.model.parameters()]

    metrics = [trainer.train_step(), trainer.train_step()]

    assert all(metric.loss > 0 for metric in metrics)
    assert all(torch.isfinite(torch.tensor(metric.gradient_norm)) for metric in metrics)
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, trainer.model.parameters())
    )
