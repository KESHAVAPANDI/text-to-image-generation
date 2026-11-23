# Offline mode (s5)

Models are loaded from folders on disk. The app never contacts Hugging Face at run time.

* Base model folder: `MODEL_PATH`, default `C:\Users\kesha\models\sdxl`
* Adapter folder: `LORA_PATH`, default `C:\Users\kesha\models\lcm_lora_sdxl`
* Default size: 768 x 768, to use less VRAM

## Run

1. Download the models once. The exact links and folder layout are in [docs/SETUP.md](../docs/SETUP.md).
2. Point the app at them:

```bash
# macOS / Linux
export MODEL_PATH="$HOME/models/sdxl"
export LORA_PATH="$HOME/models/lcm_lora_sdxl"

# Windows PowerShell
$env:MODEL_PATH = "C:\Users\kesha\models\sdxl"
$env:LORA_PATH = "C:\Users\kesha\models\lcm_lora_sdxl"
```

3. Run the app:

```bash
pip install -r ../requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

## Settings

See the configuration table in the main [README](../README.md). The useful variables here are `MODEL_PATH`, `LORA_PATH`, `DEVICE`, `HOST`, `PORT`, `DEFAULT_STEPS`, and `DEFAULT_GUIDANCE`.
