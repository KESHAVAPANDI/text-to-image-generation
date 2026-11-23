"""Text to image web app, offline mode.

Loads Stable Diffusion XL and the LCM-LoRA adapter from folders on disk,
so the app never contacts Hugging Face at run time.

Set the folders with environment variables, or keep the defaults:
    MODEL_PATH: C:\\Users\\kesha\\models\\sdxl
    LORA_PATH:  C:\\Users\\kesha\\models\\lcm_lora_sdxl

Run from this folder:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
See ../README.md and ../docs/SETUP.md for the full setup guide.
"""

import logging
import os
import threading
import uuid
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

OUTPUT_FOLDER = Path(__file__).resolve().parent / "static" / "generated_images"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", r"C:\Users\kesha\models\sdxl")
LORA_PATH = os.environ.get("LORA_PATH", r"C:\Users\kesha\models\lcm_lora_sdxl")
DEVICE_SETTING = os.environ.get("DEVICE", "auto").lower()
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5000"))
DEFAULT_STEPS = int(os.environ.get("DEFAULT_STEPS", "4"))
DEFAULT_GUIDANCE = float(os.environ.get("DEFAULT_GUIDANCE", "1.0"))
DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "768"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "768"))
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "1000"))
ENABLE_ATTENTION_SLICING = os.environ.get("ENABLE_ATTENTION_SLICING", "0") == "1"

pipe = None
model_status = "loading"
model_error = None


def resolve_device():
    """Pick CUDA when it is available, otherwise fall back to CPU."""
    if DEVICE_SETTING == "cuda":
        return "cuda"
    if DEVICE_SETTING == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = resolve_device()


def load_model():
    """Load local SDXL and fuse the local LCM-LoRA adapter. Runs in a background thread."""
    global pipe, model_status, model_error
    try:
        logger.info("Loading local model %s on %s. This may take some time.", MODEL_PATH, DEVICE)
        if DEVICE == "cuda":
            torch.backends.cudnn.benchmark = True

        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        loaded = AutoPipelineForText2Image.from_pretrained(MODEL_PATH, torch_dtype=dtype)
        loaded.scheduler = LCMScheduler.from_config(loaded.scheduler.config)
        loaded.to(DEVICE, dtype=dtype)
        if ENABLE_ATTENTION_SLICING:
            loaded.enable_attention_slicing()
        loaded.load_lora_weights(LORA_PATH)
        loaded.fuse_lora()

        pipe = loaded
        model_status = "ready"
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001 - report any load failure to /health
        model_status = "error"
        model_error = str(exc)
        logger.exception("Failed to load the model.")


def clamp_int(value, default, minimum, maximum):
    """Parse an int from user input and clamp it to a safe range."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


def clamp_float(value, default, minimum, maximum):
    """Parse a float from user input and clamp it to a safe range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, number))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    """Readiness check: loading, ready, or error, plus the device and model."""
    payload = {"status": model_status, "device": DEVICE, "model": MODEL_PATH}
    if model_error:
        payload["error"] = model_error
    status_code = 200 if model_status == "ready" else 503
    return jsonify(payload), status_code


@app.route("/generate", methods=["POST"])
def generate():
    if pipe is None:
        if model_status == "loading":
            return jsonify({"error": "Model is still loading. Please try again in a moment."}), 503
        return jsonify({"error": f"Model failed to load: {model_error}"}), 500

    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt", "")).strip()
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Prompt is too long. Maximum is {MAX_PROMPT_LENGTH} characters."}), 400

    negative_prompt = str(data.get("negative_prompt", "")).strip() or None
    steps = clamp_int(data.get("num_inference_steps"), DEFAULT_STEPS, 1, 8)
    guidance = clamp_float(data.get("guidance_scale"), DEFAULT_GUIDANCE, 0.0, 7.0)
    width = clamp_int(data.get("width"), DEFAULT_WIDTH, 256, 1024)
    height = clamp_int(data.get("height"), DEFAULT_HEIGHT, 256, 1024)
    width -= width % 8
    height -= height % 8

    seed = data.get("seed")
    try:
        seed = int(seed) if seed is not None and str(seed).strip() != "" else -1
    except (TypeError, ValueError):
        seed = -1
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    logger.info("Generating image for prompt: %s (seed=%s, steps=%s)", prompt[:120], seed, steps)
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            width=width,
            height=height,
            generator=generator,
        ).images[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        logger.warning("CUDA out of memory while generating.")
        return jsonify({"error": "CUDA out of memory. Try a smaller image size (512 or 768)."}), 500
    except Exception as exc:  # noqa: BLE001 - surface generation failures as JSON
        logger.exception("Error generating image.")
        return jsonify({"error": f"Failed to generate image: {exc}"}), 500

    filename = f"{uuid.uuid4().hex}.png"
    image_path = OUTPUT_FOLDER / filename
    image.save(image_path)
    logger.info("Image saved: %s", image_path)

    return jsonify(
        {
            "image_url": f"/static/generated_images/{filename}",
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "width": width,
            "height": height,
        }
    )


@app.route("/static/generated_images/<filename>")
def serve_image(filename):
    if "/" in filename or "\\" in filename or not filename.endswith(".png"):
        abort(404)
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    app.run(host=HOST, port=PORT, debug=os.environ.get("FLASK_DEBUG", "0") == "1")
