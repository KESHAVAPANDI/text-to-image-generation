# Setup guide

Detailed setup for both modes. The short version is in the main [README](../README.md).

## Prerequisites

* Python 3.10 or newer. Check with `python --version`.
* Git.
* A CUDA GPU with 8 GB or more VRAM is recommended for 1024px images. 512px and 768px need less.
* About 8 GB of free disk space for the model.
* On Windows, use PowerShell or Command Prompt. On macOS and Linux, use a terminal.

## Step 1: clone and create a virtual environment

```bash
git clone https://github.com/KESHAVAPANDI/text-to-image-generation.git
cd text-to-image-generation
python -m venv venv
```

Activate it:

```bash
# macOS / Linux
source venv/bin/activate

# Windows PowerShell
venv\Scripts\Activate.ps1

# Windows Command Prompt
venv\Scripts\activate.bat
```

## Step 2: install PyTorch, then the other dependencies

PyTorch is installed separately, because the correct wheel depends on your CUDA version. The tested combination is PyTorch 2.6.0 with CUDA 12.5.

```bash
# CUDA 12.5, the tested combination
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu125

# CPU only, if you have no GPU. Generation will be slow.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

Then install the rest:

```bash
pip install -r requirements.txt
```

If you are on a newer PyTorch and CUDA, the commands above may need adjusting. Check the PyTorch install page for your platform and install a matching build, then install `requirements.txt` as normal.

## Step 3a: online mode (s4)

No model download step is needed. The app downloads the model from Hugging Face on first run, about 7 GB, once.

```bash
cd s4
python app.py
```

Wait for `Model loaded successfully!` in the terminal, then open http://127.0.0.1:5000.

If Hugging Face is slow or blocked on your network, use offline mode instead.

## Step 3b: offline mode (s5)

Download the models once, then run without internet access.

### Download the SDXL base model

From Hugging Face, `stabilityai/stable-diffusion-xl-base-1.0`:

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

You need the model files in a folder, for example `C:\Users\kesha\models\sdxl` on Windows or `~/models/sdxl` elsewhere. The folder should contain the usual Diffusers layout: `model_index.json`, plus `text_encoder`, `text_encoder_2`, `unet`, `vae`, and `scheduler` subfolders. The easiest way to get that layout is:

```bash
# From the repository root, with the virtual environment active
python -c "from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', local_dir='models/sdxl')"
```

Or download the files from the Hugging Face page and place them in that layout manually.

### Download the LCM-LoRA adapter

From Hugging Face, `latent-consistency/lcm-lora-sdxl`:

https://huggingface.co/latent-consistency/lcm-lora-sdxl

Place it in a folder, for example `C:\Users\kesha\models\lcm_lora_sdxl`, containing the LoRA weights (`pytorch_lora_weights.safetensors` and its config).

### Point the app at the folders

Set environment variables before running the app:

```bash
# macOS / Linux
export MODEL_PATH="$HOME/models/sdxl"
export LORA_PATH="$HOME/models/lcm_lora_sdxl"

# Windows PowerShell
$env:MODEL_PATH = "C:\Users\kesha\models\sdxl"
$env:LORA_PATH = "C:\Users\kesha\models\lcm_lora_sdxl"
```

Or copy `.env.example` to `.env` in the repository root and edit the two paths there.

Then run:

```bash
cd "s5 (offline)"
python app.py
```

Open http://127.0.0.1:5000.

## Verifying the install

1. The terminal shows `Model loaded successfully!`
2. `GET http://127.0.0.1:5000/health` returns `{"status": "ready", ...}`
3. A prompt in the UI returns an image in a few seconds on GPU

If step 1 never appears, read the terminal output: the first run downloads the model, and a network error there looks like a model error.

## Troubleshooting

### CUDA out of memory

Use 512 or 768px in the UI, close other GPU applications, and restart the app. The app clears the CUDA cache after an out of memory error, but the model itself still needs its VRAM.

### The model download is very slow or fails

Hugging Face downloads can be slow on some networks. Try again, or switch to offline mode and download the model files with a download manager or from a faster network, then copy them over.

### `torch` import error, or a CUDA version mismatch

Your PyTorch build does not match your CUDA driver. Reinstall PyTorch for your CUDA version, as in Step 2, and confirm it with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` should print `True` on a CUDA machine.

### Port 5000 is already in use

Set a different port and restart:

```bash
PORT=5001 python app.py
```

### The page loads but generation never finishes

Check the terminal. If the model is still loading, `/health` returns `status: loading`. On CPU, a single generation can take several minutes: that is expected, not a hang.

### Images are saved, but the UI shows a broken image

Check that `static/generated_images/` exists inside the folder you ran the app from (`s4` or `s5 (offline)`), and that the PNG file is there. If the folder is missing, restart the app: it creates the folder on startup.
