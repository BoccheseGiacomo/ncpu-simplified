from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from .config import ExperimentConfig
from .layout import InterleavedLayout
from .model import NeuralCellularAutomaton


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def supervised_loss(
    rollout: torch.Tensor,
    target: torch.Tensor,
    layout: InterleavedLayout,
    output_channel: int,
    free_steps: int,
    supervision_steps: int,
) -> torch.Tensor:
    start = free_steps + 1
    end = start + supervision_steps
    if rollout.ndim != 5:
        raise ValueError(
            "rollout must have shape (batch, time, channels, height, width)"
        )
    if start < 1 or end > rollout.shape[1]:
        raise ValueError("rollout does not cover the requested supervision window")
    if target.shape != (rollout.shape[0], rollout.shape[3], rollout.shape[4]):
        raise ValueError("target shape does not match rollout")
    prediction = layout.read_output_values(rollout[:, start:end, output_channel])
    expected = layout.read_output_values(target).unsqueeze(1)
    return (prediction - expected).square().mean()


def cosine_learning_rate(config: ExperimentConfig, update: int) -> float:
    training = config.training
    if not 0 <= update < training.updates:
        raise ValueError("update must be in [0, updates)")
    if training.warmup_updates and update < training.warmup_updates:
        return training.learning_rate * (update + 1) / training.warmup_updates
    decay_updates = training.updates - training.warmup_updates
    decay_index = update - training.warmup_updates
    progress = 1.0 if decay_updates == 1 else decay_index / (decay_updates - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        training.final_learning_rate
        + (training.learning_rate - training.final_learning_rate) * cosine
    )


@dataclass(frozen=True)
class StepMetrics:
    update: int
    loss: float
    learning_rate: float
    gradient_norm: float
    exact_accuracy: float
    validation_loss: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "update": self.update,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "exact_accuracy": self.exact_accuracy,
            "validation_loss": self.validation_loss,
        }


@dataclass(frozen=True)
class SeedResult:
    seed: int
    checkpoint: Path
    best_validation_loss: float
    history: list[dict[str, float | int | None]]


class WandbReporter:
    def __init__(
        self,
        project: str,
        config: ExperimentConfig,
        *,
        run_name: str,
        group: str | None = None,
    ):
        if not project.strip():
            raise ValueError("W&B project cannot be empty")
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "W&B reporting requires: pip install -e '.[tracking]'"
            ) from error
        self.run = wandb.init(
            project=project,
            group=group,
            name=run_name,
            config=config.to_dict(),
        )

    def __call__(self, metrics: StepMetrics) -> None:
        values = {
            key: value
            for key, value in metrics.to_dict().items()
            if key != "update" and value is not None
        }
        self.run.log(values, step=metrics.update)

    def finish(self, best_validation_loss: float) -> None:
        self.run.summary["best_validation_loss"] = best_validation_loss
        self.run.finish()


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model: NeuralCellularAutomaton | None = None,
    ):
        config.validate()
        self.config = config
        self.device = resolve_device(config.training.device)
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        self.model = (
            NeuralCellularAutomaton(config.model) if model is None else model
        ).to(self.device)
        if self.model.config != config.model:
            raise ValueError(
                "model configuration does not match experiment configuration"
            )
        self.layout = InterleavedLayout(config.geometry)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.data_generator = torch.Generator().manual_seed(config.training.seed)
        self.current_update = 0
        self.best_validation_loss = math.inf
        self.history: list[dict[str, float | int | None]] = []

    def _sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        maximum = 1 << self.layout.bits
        operands_a = torch.randint(
            maximum,
            (self.config.training.batch_size,),
            generator=self.data_generator,
        )
        operands_b = torch.randint(
            maximum,
            (self.config.training.batch_size,),
            generator=self.data_generator,
        )
        inputs, targets = self.layout.render_batch(operands_a, operands_b)
        return inputs.to(self.device), targets.to(self.device)

    def train_step(self) -> StepMetrics:
        self.model.train()
        learning_rate = cosine_learning_rate(self.config, self.current_update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = self._sample_batch()
        initial = self.model.initial_state(inputs)
        rollout = self.model(initial, self.config.training.rollout_steps)
        loss = supervised_loss(
            rollout,
            targets,
            self.layout,
            self.config.model.output_channel,
            self.config.training.free_steps,
            self.config.training.supervision_steps,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss at update {self.current_update + 1}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.training.grad_clip is None:
            squared_norm = sum(
                parameter.grad.detach().square().sum()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )
            gradient_norm = float(squared_norm.sqrt())
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
            )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update {self.current_update + 1}"
            )
        self.optimizer.step()

        with torch.no_grad():
            final_values = self.layout.read_output_values(
                rollout[:, -1, self.config.model.output_channel]
            )
            target_values = self.layout.read_output_values(targets)
            exact = ((final_values > 0) == (target_values > 0)).all(dim=1)

        self.current_update += 1
        return StepMetrics(
            update=self.current_update,
            loss=float(loss.detach()),
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            exact_accuracy=float(exact.float().mean()),
        )

    @torch.no_grad()
    def validation_loss(self) -> float:
        self.model.eval()
        operands_a, operands_b = self.layout.operand_pairs()
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        weighted_loss = 0.0
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.training.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.config.training.seed)
            for offset in range(0, len(operands_a), self.config.training.batch_size):
                batch_a = operands_a[offset : offset + self.config.training.batch_size]
                batch_b = operands_b[offset : offset + self.config.training.batch_size]
                inputs, targets = self.layout.render_batch(batch_a, batch_b)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                rollout = self.model(
                    self.model.initial_state(inputs),
                    self.config.training.rollout_steps,
                )
                loss = supervised_loss(
                    rollout,
                    targets,
                    self.layout,
                    self.config.model.output_channel,
                    self.config.training.free_steps,
                    self.config.training.supervision_steps,
                )
                weighted_loss += float(loss) * len(batch_a)
        return weighted_loss / len(operands_a)

    def fit(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        progress_every: int = 25,
        callback: Callable[[StepMetrics], None] | None = None,
    ) -> list[dict[str, float | int | None]]:
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        while self.current_update < self.config.training.updates:
            metrics = self.train_step()
            validation_loss = None
            should_validate = (
                metrics.update % self.config.training.validation_every == 0
                or metrics.update == self.config.training.updates
            )
            if should_validate:
                validation_loss = self.validation_loss()
            is_best = (
                validation_loss is not None
                and validation_loss < self.best_validation_loss
            )
            if is_best:
                self.best_validation_loss = validation_loss

            metrics = StepMetrics(
                update=metrics.update,
                loss=metrics.loss,
                learning_rate=metrics.learning_rate,
                gradient_norm=metrics.gradient_norm,
                exact_accuracy=metrics.exact_accuracy,
                validation_loss=validation_loss,
            )
            self.history.append(metrics.to_dict())
            if is_best:
                self.save(checkpoint_dir / "best.pt")
            if callback is not None:
                callback(metrics)
            if metrics.update % progress_every == 0 or should_validate:
                validation_text = (
                    ""
                    if validation_loss is None
                    else f" validation={validation_loss:.6g}"
                )
                print(
                    f"update {metrics.update:5d}/{self.config.training.updates} "
                    f"loss={metrics.loss:.6g} exact={metrics.exact_accuracy:.2%} "
                    f"lr={metrics.learning_rate:.3g}{validation_text}"
                )
            if (
                metrics.update % self.config.training.checkpoint_every == 0
                or metrics.update == self.config.training.updates
            ):
                self.save(checkpoint_dir / "latest.pt")
        return self.history

    def checkpoint(self) -> dict:
        checkpoint = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.current_update,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "data_rng_state": self.data_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return checkpoint

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "Trainer":
        # RNG states must stay on CPU; model/optimizer loading handles device transfer.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        config_data = checkpoint["config"]
        if device is not None:
            config_data = {
                **config_data,
                "training": {**config_data["training"], "device": device},
            }
        config = ExperimentConfig.from_dict(config_data)
        trainer = cls(config)
        trainer.model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_update = int(checkpoint["update"])
        trainer.best_validation_loss = float(checkpoint["best_validation_loss"])
        trainer.history = list(checkpoint["history"])
        trainer.data_generator.set_state(checkpoint["data_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if trainer.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return trainer


def load_model(
    path: str | Path, device: str = "auto"
) -> tuple[NeuralCellularAutomaton, ExperimentConfig, dict]:
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = NeuralCellularAutomaton(config.model).to(resolved)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def train_seeds(
    config: ExperimentConfig,
    seeds: tuple[int, ...],
    checkpoint_dir: str | Path = "checkpoints",
    *,
    resume: bool = False,
    progress_every: int = 25,
    wandb_project: str | None = None,
    wandb_group: str | None = None,
) -> list[SeedResult]:
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results: list[SeedResult] = []

    for seed in seeds:
        model_config = config.model
        if model_config.learnable_kernel_init == "random":
            model_config = replace(model_config, random_kernel_seed=seed)
        seed_config = replace(
            config,
            model=model_config,
            training=replace(config.training, seed=seed),
        )
        seed_dir = checkpoint_dir / f"seed_{seed}"
        latest = seed_dir / "latest.pt"
        if resume and latest.is_file():
            trainer = Trainer.from_checkpoint(
                latest, device=seed_config.training.device
            )
            if trainer.config != seed_config:
                raise ValueError(f"seed {seed} checkpoint configuration does not match")
        else:
            trainer = Trainer(seed_config)

        reporter = (
            WandbReporter(
                wandb_project,
                seed_config,
                run_name=f"seed-{seed}",
                group=wandb_group,
            )
            if wandb_project is not None
            else None
        )
        try:
            history = trainer.fit(
                seed_dir,
                progress_every=progress_every,
                callback=reporter,
            )
        finally:
            if reporter is not None:
                reporter.finish(trainer.best_validation_loss)
        results.append(
            SeedResult(
                seed=seed,
                checkpoint=seed_dir / "best.pt",
                best_validation_loss=trainer.best_validation_loss,
                history=history,
            )
        )

    best = min(results, key=lambda result: result.best_validation_loss)
    shutil.copy2(best.checkpoint, checkpoint_dir / "best.pt")
    return results
