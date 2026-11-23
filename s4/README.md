# Online mode (s4)

Models are fetched from Hugging Face on first run. No local model files needed.

* Base model: `stabilityai/stable-diffusion-xl-base-1.0`
* Adapter: `latent-consistency/lcm-lora-sdxl`
* Default size: 1024 x 1024

## Run

```bash
pip install -r ../requirements.txt
python app.py
```

Then open http://127.0.0.1:5000.

The first run downloads about 7 GB and takes a while. Later runs reuse the cached model.

## Settings

See the configuration table in the main [README](../README.md). The useful variables here are `MODEL_ID`, `LORA_ID`, `DEVICE`, `HOST`, `PORT`, `DEFAULT_STEPS`, and `DEFAULT_GUIDANCE`.
