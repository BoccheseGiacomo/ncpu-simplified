import torch

from ncpu_simplified import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
    validate,
)
from ncpu_simplified.training import Trainer
from ncpu_simplified.visualize import create_app, rollout_rgb, save_gif


def small_config():
    return ExperimentConfig(
        geometry=GeometryConfig(bits=1),
        model=ModelConfig(hidden_size=8),
        training=TrainingConfig(
            updates=1,
            batch_size=2,
            free_steps=1,
            supervision_steps=1,
            validation_every=1,
            checkpoint_every=1,
            device="cpu",
        ),
    )


def test_default_validate_runs_every_check_and_reports_parameter_count():
    report = validate(ExperimentConfig())

    report.require_success()
    assert report.successful
    assert report.parameter_count == 921
    assert len(report.checks) == 7
    assert "steps 71-140" in str(report)


def test_validation_is_not_tied_to_default_geometry_or_hidden_width():
    report = validate(small_config())

    report.require_success()
    assert report.successful


def test_rgb_conversion_uses_three_channels_and_valid_byte_range():
    rollout = torch.tensor(
        [[[[0.0]], [[1.0]], [[-1.0]]], [[[10.0]], [[-10.0]], [[0.0]]]]
    )

    rgb = rollout_rgb(rollout)

    assert rgb.shape == (2, 1, 1, 3)
    assert rgb.dtype == torch.uint8
    assert int(rgb.min()) >= 0 and int(rgb.max()) <= 255


def test_gif_and_http_inference_work_from_a_native_checkpoint(tmp_path):
    trainer = Trainer(small_config())
    trainer.train_step()
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)

    gif = save_gif(torch.zeros(3, 3, 4, 5), tmp_path / "rollout.gif", scale=2)
    assert gif.is_file() and gif.stat().st_size > 0

    client = create_app(checkpoint, device="cpu").test_client()
    assert client.get("/").status_code == 200
    assert client.get("/health").json["status"] == "ok"
    config_response = client.get("/api/config")
    assert config_response.status_code == 200
    response = client.post("/api/infer", json={"bits": 2, "a": 2, "b": 3, "steps": 3})

    assert response.status_code == 200
    payload = response.json
    assert payload["expected"] == 5
    assert len(payload["outputs"]) == 4
    assert len(payload["frames"]) == 4
    assert len(payload["frames"][0]) == payload["height"]
    assert len(payload["frames"][0][0]) == payload["width"]


def test_http_interface_rejects_invalid_requests(tmp_path):
    trainer = Trainer(small_config())
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    client = create_app(checkpoint, device="cpu").test_client()

    assert client.post("/api/infer", data="not json").status_code == 400
    assert client.post("/api/infer", json={"bits": 0}).status_code == 400
    assert client.post("/api/infer", json={"bits": 2, "a": 4}).status_code == 400
    assert client.post("/api/infer", json={"steps": 1001}).status_code == 400
