from __future__ import annotations

import argparse
import html
import webbrowser
from dataclasses import replace
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from PIL import Image, ImageDraw, ImageFont

from .config import ExperimentConfig, TrainingConfig
from .layout import InterleavedLayout
from .training import load_model


def evolution_phase(step: int, training: TrainingConfig) -> str:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step <= training.free_steps:
        return "Free evolution"
    if step <= training.supervision_end:
        return "Supervision window"
    return "Beyond training window"


def _viewer_config(config: ExperimentConfig) -> dict:
    return {
        "bits": config.geometry.bits,
        "trained_bits": config.geometry.bits,
        "steps": max(200, config.training.rollout_steps),
        "training_steps": config.training.rollout_steps,
        "free_steps": config.training.free_steps,
        "supervision_start": config.training.supervision_start,
        "supervision_end": config.training.supervision_end,
        "channels": config.model.channels,
        "input_channel": config.model.input_channel,
        "input_mode": config.model.input_mode,
        "fire_rate": config.model.fire_rate,
    }


def _validate_rollout(rollout: torch.Tensor) -> None:
    if rollout.ndim != 4 or min(rollout.shape) < 1:
        raise ValueError("rollout must have shape (time, channels, height, width)")
    if not torch.is_floating_point(rollout) or not torch.isfinite(rollout).all():
        raise ValueError("rollout must contain finite floating-point states")


def rollout_rgb(rollout: torch.Tensor) -> torch.Tensor:
    _validate_rollout(rollout)
    channels = rollout.shape[1]
    if channels >= 3:
        color = rollout[:, :3]
    else:
        missing = torch.zeros(
            rollout.shape[0],
            3 - channels,
            rollout.shape[2],
            rollout.shape[3],
            device=rollout.device,
            dtype=rollout.dtype,
        )
        color = torch.cat((rollout, missing), dim=1)
    return (
        ((torch.tanh(color) + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)
    )


def rollout_payload(
    rollout: torch.Tensor,
    layout: InterleavedLayout,
    config: ExperimentConfig,
    operand_a: int,
    operand_b: int,
) -> dict:
    """Describe an existing trajectory without changing or rerunning the model."""
    config.validate()
    _validate_rollout(rollout)
    if tuple(rollout.shape[1:]) != (
        config.model.channels,
        layout.height,
        layout.width,
    ):
        raise ValueError("rollout dimensions do not match the layout and model")
    for operand in (operand_a, operand_b):
        if type(operand) is not int or not 0 <= operand < (1 << layout.bits):
            raise ValueError("operands must be integers within the layout bit width")
    states = rollout.detach().cpu()
    outputs = layout.decode_output(states[:, config.model.input_channel])
    return {
        **_viewer_config(config),
        "bits": layout.bits,
        "steps": states.shape[0] - 1,
        "a": operand_a,
        "b": operand_b,
        "expected": operand_a + operand_b,
        "outputs": outputs.tolist(),
        "states": states.tolist(),
        "frames": rollout_rgb(states).tolist(),
        "height": layout.height,
        "width": layout.width,
        "ports": {
            "a": layout.input_a_coordinates,
            "b": layout.input_b_coordinates,
            "output": layout.output_coordinates,
        },
    }


def _viewer_document(config: dict, payload: dict | None, duration_ms: int) -> str:
    environment = Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        autoescape=True,
        undefined=StrictUndefined,
    )
    return environment.get_template("index.html").render(
        config=config, embedded_data=payload, duration_ms=duration_ms
    )


def notebook_viewer(
    rollout: torch.Tensor,
    layout: InterleavedLayout,
    config: ExperimentConfig,
    operand_a: int,
    operand_b: int,
    *,
    duration_ms: int = 80,
    height: int = 1100,
):
    """Return a self-contained notebook player; no running server is required."""
    from IPython.display import IFrame

    if duration_ms < 1 or height < 1:
        raise ValueError("duration_ms and height must be positive")
    payload = rollout_payload(rollout, layout, config, operand_a, operand_b)
    document = _viewer_document(_viewer_config(config), payload, duration_ms)
    return IFrame(
        "about:blank",
        width="100%",
        height=height,
        extras=[
            'title="NCA inference"',
            'sandbox="allow-scripts"',
            'style="width:100%;border:0;max-width:1100px"',
            f'srcdoc="{html.escape(document, quote=True)}"',
        ],
    )


def save_gif(
    rollout: torch.Tensor,
    path: str | Path,
    *,
    duration_ms: int = 80,
    scale: int = 12,
    layout: InterleavedLayout | None = None,
    config: ExperimentConfig | None = None,
    operand_a: int | None = None,
    operand_b: int | None = None,
) -> Path:
    if duration_ms < 1 or scale < 1:
        raise ValueError("duration_ms and scale must be positive")
    metadata = (layout, config, operand_a, operand_b)
    annotated = any(item is not None for item in metadata)
    if annotated and any(item is None for item in metadata):
        raise ValueError("annotated GIFs need layout, config, and both operands")
    if annotated:
        payload = rollout_payload(rollout, layout, config, operand_a, operand_b)
    frames = rollout_rgb(rollout).cpu().numpy()
    images = [
        Image.fromarray(frame).resize(
            (frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST
        )
        for frame in frames
    ]
    if annotated:
        images = [
            _annotate_frame(frame, step, payload, config.training, scale)
            for step, frame in enumerate(images)
        ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _annotate_frame(
    frame: Image.Image, step: int, payload: dict, training: TrainingConfig, scale: int
) -> Image.Image:
    left, top = 18, 112
    width = max(380, frame.width + left + 84)
    image = Image.new("RGB", (width, frame.height + top + 46), "#161a1f")
    image.paste(frame, (left, top))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=12)
    large = ImageFont.load_default(size=28)
    foreground = "#e7ebef"
    phase = evolution_phase(step, training)
    draw.text((left, 4), "Decoded now", fill=foreground, font=font)
    draw.text((left, 17), str(payload["outputs"][step]), fill=foreground, font=large)
    draw.text(
        (left, 51), f"t = {step} / {payload['steps']}", fill=foreground, font=font
    )
    draw.text(
        (left, 67),
        f"Status: {phase}",
        fill="#ff8585" if phase == "Free evolution" else foreground,
        font=font,
    )
    draw.text(
        (left, 83),
        f"Expected: {payload['a']} + {payload['b']} = {payload['expected']}",
        fill=foreground,
        font=font,
    )
    for group, label in (("a", "A"), ("b", "B"), ("output", "O")):
        column = payload["ports"][group][0][1]
        draw.text(
            (left + (column + 0.5) * scale, top - 14),
            label,
            fill=foreground,
            font=font,
            anchor="mt",
        )
        for index, (row, column) in enumerate(payload["ports"][group]):
            x, y = left + column * scale, top + row * scale
            draw.rectangle((x, y, x + scale - 1, y + scale - 1), outline="black")
            if scale >= 4:
                draw.rectangle(
                    (x + 1, y + 1, x + scale - 2, y + scale - 2), outline="white"
                )
            if group == "output":
                label = f"2^{payload['bits'] - index}"
                if index == 0:
                    label += " carry"
                draw.text(
                    (left + frame.width + 6, y), label, fill=foreground, font=font
                )
    channels = "   ".join(
        f"{color}: ch{index}" if index < payload["channels"] else f"{color}: 0"
        for index, color in enumerate(("R", "G", "B"))
    )
    draw.text(
        (left, top + frame.height + 8),
        channels + "   |   0: gray",
        fill=foreground,
        font=font,
    )
    return image


def create_app(checkpoint_path: str | Path, device: str = "auto") -> Flask:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            "Train the model in run/run.ipynb first."
        )
    model, config, checkpoint = load_model(checkpoint_path, device=device)
    app = Flask(__name__)
    metadata = {
        **_viewer_config(config),
        "checkpoint": checkpoint_path.name,
        "update": int(checkpoint["update"]),
    }

    @app.get("/")
    def index():
        return _viewer_document(metadata, None, 80)

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            checkpoint=str(checkpoint_path),
            update=int(checkpoint["update"]),
        )

    @app.get("/api/config")
    def api_config():
        return jsonify(metadata)

    @app.post("/api/infer")
    def api_infer():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="request body must be a JSON object"), 400
        bits = data.get("bits", config.geometry.bits)
        operand_a = data.get("a", 0)
        operand_b = data.get("b", 0)
        steps = data.get("steps", metadata["steps"])
        if any(type(value) is not int for value in (bits, operand_a, operand_b, steps)):
            return jsonify(error="bits, operands, and steps must be integers"), 400
        if not 1 <= bits <= 16:
            return jsonify(error="bits must be in [1, 16]"), 400
        maximum = 1 << bits
        if not 0 <= operand_a < maximum or not 0 <= operand_b < maximum:
            return jsonify(error=f"operands must be in [0, {maximum})"), 400
        if not 1 <= steps <= 1000:
            return jsonify(error="steps must be in [1, 1000]"), 400

        layout = InterleavedLayout(replace(config.geometry, bits=bits))
        inputs, _ = layout.render_batch(
            torch.tensor([operand_a]), torch.tensor([operand_b])
        )
        with torch.inference_mode():
            rollout = model(model.initial_state(inputs.to(model.device)), steps)[0]
        if not torch.isfinite(rollout).all():
            return (
                jsonify(error="Model produced non-finite states; shorten the rollout"),
                422,
            )
        return jsonify(rollout_payload(rollout, layout, config, operand_a, operand_b))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ncpu-simplified visualizer")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    arguments = parser.parse_args()
    app = create_app(arguments.checkpoint, arguments.device)
    url = f"http://{arguments.host}:{arguments.port}"
    print(f"Visualizer: {url}")
    if arguments.open_browser:
        webbrowser.open(url)
    app.run(host=arguments.host, port=arguments.port, debug=False)


if __name__ == "__main__":
    main()
