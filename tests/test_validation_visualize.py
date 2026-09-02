import html
import re
from dataclasses import replace

import pytest
import torch
from PIL import Image

from ncpu_simplified import (
    ExperimentConfig,
    GeometryConfig,
    InterleavedLayout,
    ModelConfig,
    TrainingConfig,
    validate,
)
from ncpu_simplified.training import Trainer
from ncpu_simplified.visualize import (
    create_app,
    evolution_phase,
    notebook_viewer,
    rollout_payload,
    rollout_rgb,
    save_gif,
)


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


@pytest.mark.parametrize(
    "step, expected",
    [
        (0, "Free evolution"),
        (70, "Free evolution"),
        (71, "Supervision window"),
        (140, "Supervision window"),
        (141, "Beyond training window"),
        (200, "Beyond training window"),
    ],
)
def test_phase_boundaries(step, expected):
    assert evolution_phase(step, TrainingConfig()) == expected


def test_phase_without_free_evolution():
    config = TrainingConfig(free_steps=0, supervision_steps=3)
    assert evolution_phase(1, config) == "Supervision window"
    assert evolution_phase(4, config) == "Beyond training window"
    with pytest.raises(ValueError):
        evolution_phase(-1, config)


@pytest.mark.parametrize("channels", [1, 2, 3, 5])
def test_payload_keeps_raw_channels_and_actual_port_geometry(channels):
    config = replace(small_config(), model=ModelConfig(channels=channels))
    layout = InterleavedLayout(
        GeometryConfig(
            bits=6,
            sx=2,
            sy=4,
            border_left=1,
            border_right=5,
            border_top=2,
            border_bottom=0,
        )
    )
    states = torch.zeros(4, channels, layout.height, layout.width)
    row, column = layout.output_coordinates[-1]
    states[:, 0, row, column] = torch.tensor([-1e-9, 0, 1e-9, 1.5])
    before = states.clone()
    payload = rollout_payload(states, layout, config, 31, 32)
    assert payload["bits"] == 6 and payload["trained_bits"] == 1
    assert payload["expected"] == 63
    assert payload["outputs"] == [0, 0, 1, 1]
    assert payload["ports"]["a"] == layout.input_a_coordinates
    assert payload["ports"]["b"] == layout.input_b_coordinates
    assert payload["ports"]["output"] == layout.output_coordinates
    assert payload["height"] == layout.height
    assert payload["width"] == layout.width
    torch.testing.assert_close(torch.tensor(payload["states"]), before)
    torch.testing.assert_close(states, before)
    assert torch.tensor(payload["frames"]).shape == (4, layout.height, layout.width, 3)
    if channels < 3:
        assert torch.all(torch.tensor(payload["frames"])[..., channels:] == 128)


def test_payload_reads_the_configured_channel():
    config = replace(small_config(), model=ModelConfig(input_channel=2))
    layout = InterleavedLayout(config.geometry)
    states = torch.zeros(1, 3, layout.height, layout.width)
    row, column = layout.output_coordinates[-1]
    states[0, 2, row, column] = 1
    assert rollout_payload(states, layout, config, 0, 1)["outputs"] == [1]


def test_payload_does_not_change_double_precision_values_or_decoding():
    config = small_config()
    layout = InterleavedLayout(config.geometry)
    states = torch.zeros(1, 3, layout.height, layout.width, dtype=torch.float64)
    row, column = layout.output_coordinates[-1]
    states[0, 0, row, column] = 1e-300
    data = rollout_payload(states, layout, config, 0, 1)
    assert data["outputs"] == [1]
    assert data["states"][0][0][row][column] == 1e-300


def test_non_finite_inference_is_reported_instead_of_decoded(tmp_path):
    trainer = Trainer(small_config())
    with torch.no_grad():
        trainer.model.rule.output.weight.fill_(float("nan"))
    checkpoint = tmp_path / "nonfinite.pt"
    trainer.save(checkpoint)
    client = create_app(checkpoint, device="cpu").test_client()
    response = client.post("/api/infer", json={"a": 0, "b": 1, "steps": 3})
    assert response.status_code == 422
    assert "non-finite" in response.json["error"]
    assert "outputs" not in response.json


def test_payload_rejects_mismatched_dimensions_and_operands():
    config = small_config()
    layout = InterleavedLayout(config.geometry)
    states = torch.zeros(2, 3, layout.height, layout.width)
    with pytest.raises(ValueError, match="dimensions"):
        rollout_payload(states[:, :2], layout, config, 0, 1)
    for a in (-1, 2, True, 0.5):
        with pytest.raises(ValueError, match="operands"):
            rollout_payload(states, layout, config, a, 1)


@pytest.mark.parametrize(
    "states",
    [
        torch.zeros(0, 3, 4, 4),
        torch.zeros(3, 4, 4),
        torch.zeros(1, 0, 4, 4),
        torch.zeros(1, 3, 4, 4, dtype=torch.int64),
        torch.full((1, 3, 4, 4), float("nan")),
        torch.full((1, 3, 4, 4), float("inf")),
    ],
)
def test_invalid_states_cannot_be_displayed_as_plausible_colors(states):
    with pytest.raises(ValueError):
        rollout_rgb(states)


def test_server_runs_200_steps_without_changing_training_window(tmp_path):
    config = replace(small_config(), training=TrainingConfig(device="cpu"))
    trainer = Trainer(config)
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    client = create_app(checkpoint, device="cpu").test_client()
    metadata = client.get("/api/config").json
    assert metadata["steps"] == 200
    assert metadata["training_steps"] == 140
    response = client.post("/api/infer", json={"bits": 4, "a": 3, "b": 5, "steps": 200})
    assert response.status_code == 200
    data = response.json
    assert len(data["states"]) == len(data["outputs"]) == len(data["frames"]) == 201
    assert data["steps"] == 200
    assert (data["free_steps"], data["supervision_start"], data["supervision_end"]) == (
        70,
        71,
        140,
    )
    layout = InterleavedLayout(replace(config.geometry, bits=4))
    inputs, _ = layout.render_batch(torch.tensor([3]), torch.tensor([5]))
    torch.testing.assert_close(
        torch.tensor(data["states"])[0], trainer.model.initial_state(inputs)[0]
    )


@pytest.mark.parametrize("rate", [1.0, 0.5])
@pytest.mark.parametrize("input_mode", ["mutable", "frozen"])
def test_viewer_respects_update_configuration(tmp_path, rate, input_mode):
    config = replace(
        small_config(),
        model=ModelConfig(
            hidden_size=8,
            input_mode=input_mode,
            fire_rate=rate,
        ),
    )
    trainer = Trainer(config)
    with torch.no_grad():
        trainer.model.rule.output.weight.fill_(0.01)
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    client = create_app(checkpoint, device="cpu").test_client()
    data = client.post("/api/infer", json={"a": 1, "b": 0, "steps": 4}).json
    assert data["fire_rate"] == rate and data["input_mode"] == input_mode
    if input_mode == "frozen":
        states = torch.tensor(data["states"])
        torch.testing.assert_close(states[:, 0], states[0:1, 0].expand_as(states[:, 0]))


def test_http_rejects_non_integer_fields(tmp_path):
    trainer = Trainer(small_config())
    checkpoint = tmp_path / "model.pt"
    trainer.save(checkpoint)
    client = create_app(checkpoint, device="cpu").test_client()
    for field in ("bits", "a", "b", "steps"):
        for value in (True, None, 1.5, "2", [], {}):
            response = client.post("/api/infer", json={field: value})
            assert response.status_code == 400


def test_notebook_is_self_contained_and_uses_longer_inference():
    config = small_config()
    layout = InterleavedLayout(config.geometry)
    states = torch.zeros(201, 3, layout.height, layout.width)
    player = notebook_viewer(states, layout, config, 0, 1)
    markup = player._repr_html_()
    assert 'sandbox="allow-scripts"' in markup
    assert player.width == "100%" and player.height == 1100
    document = html.unescape(re.search(r'srcdoc="([\s\S]*)"\s*></iframe>', markup)[1])
    assert '<form id="inference"' not in document
    assert '"steps": 200' in document
    assert '"supervision_end": 2' in document
    assert '<div id="decoded">' in document
    assert 'id="phase"' in document
    for removed in (
        "outputs still neutral",
        "Mixed colors combine",
        "<table",
        "Each component:",
    ):
        assert removed not in document


def test_annotated_and_plain_gif_exports(tmp_path):
    config = small_config()
    layout = InterleavedLayout(config.geometry)
    states = torch.zeros(4, 3, layout.height, layout.width)
    annotated = save_gif(
        states,
        tmp_path / "annotated.gif",
        scale=2,
        layout=layout,
        config=config,
        operand_a=0,
        operand_b=1,
    )
    with Image.open(annotated) as image:
        assert image.n_frames == 4
        assert image.width > layout.width * 2
        assert image.height > layout.height * 2
        assert image.info["duration"] == 80
    with pytest.raises(ValueError, match="annotated"):
        save_gif(states, tmp_path / "partial.gif", layout=layout)
