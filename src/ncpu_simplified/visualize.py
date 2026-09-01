from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image

from .layout import InterleavedLayout
from .training import load_model


def rollout_rgb(rollout: torch.Tensor) -> torch.Tensor:
    if rollout.ndim != 4:
        raise ValueError("rollout must have shape (time, channels, height, width)")
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


def save_gif(
    rollout: torch.Tensor,
    path: str | Path,
    *,
    duration_ms: int = 80,
    scale: int = 12,
) -> Path:
    if duration_ms < 1 or scale < 1:
        raise ValueError("duration_ms and scale must be positive")
    frames = rollout_rgb(rollout).cpu().numpy()
    images = [
        Image.fromarray(frame).resize(
            (frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST
        )
        for frame in frames
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


def create_app(checkpoint_path: str | Path, device: str = "auto") -> Flask:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint_path}. "
            "Train the model in run/run.ipynb first."
        )
    model, config, checkpoint = load_model(checkpoint_path, device=device)
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        return jsonify(
            status="ok",
            checkpoint=str(checkpoint_path),
            update=int(checkpoint["update"]),
        )

    @app.get("/api/config")
    def api_config():
        return jsonify(
            bits=config.geometry.bits,
            steps=config.training.rollout_steps,
            channels=config.model.channels,
            checkpoint=checkpoint_path.name,
            update=int(checkpoint["update"]),
        )

    @app.post("/api/infer")
    def api_infer():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="request body must be a JSON object"), 400
        try:
            bits = int(data.get("bits", config.geometry.bits))
            operand_a = int(data.get("a", 0))
            operand_b = int(data.get("b", 0))
            steps = int(data.get("steps", config.training.rollout_steps))
        except (TypeError, ValueError):
            return jsonify(error="bits, operands, and steps must be integers"), 400
        if not 1 <= bits <= 16:
            return jsonify(error="bits must be in [1, 16]"), 400
        maximum = 1 << bits
        if not 0 <= operand_a < maximum or not 0 <= operand_b < maximum:
            return jsonify(error=f"operands must be in [0, {maximum})"), 400
        if not 1 <= steps <= 1000:
            return jsonify(error="steps must be in [1, 1000]"), 400

        from dataclasses import replace

        layout = InterleavedLayout(replace(config.geometry, bits=bits))
        inputs, _ = layout.render_batch(
            torch.tensor([operand_a]), torch.tensor([operand_b])
        )
        with torch.no_grad():
            rollout = model(model.initial_state(inputs.to(model.device)), steps)[0]
        decoded = layout.decode_output(rollout[:, config.model.input_channel]).cpu()
        return jsonify(
            bits=bits,
            a=operand_a,
            b=operand_b,
            expected=operand_a + operand_b,
            outputs=decoded.tolist(),
            frames=rollout_rgb(rollout).cpu().tolist(),
            height=layout.height,
            width=layout.width,
            channels=config.model.channels,
        )

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
