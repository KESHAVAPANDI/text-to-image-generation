# Vercel live demo (real generation)

This folder is the live version of the text to image UI, deployed on Vercel.

* `index.html`: the same clean UI as `s4/templates/index.html`, with generation wired to a free hosted image model instead of the local Flask backend.
* `demo.jpg`: no longer used by the page. It stays in the repo history only.

## How generation works here

Every prompt generates a real, new image through Pollinations (Flux model), free with no API key and no sign in. The page requests `https://image.pollinations.ai/prompt/<your prompt>?width=&height=&seed=&model=` and shows the image that comes back. Same prompt and seed gives the same image, a new seed gives a variation.

Controls on this page are only the ones the hosted model supports: model (Flux or Turbo), size, and seed. Steps, guidance, and negative prompt are local SDXL controls, so they live in the Flask app only.

## Why not the real SDXL app?

The real app runs SDXL + LCM-LoRA on a CUDA GPU, with a 7 GB model. Vercel is serverless hosting with no GPU, strict function size and time limits, so the real generation cannot run there. If you want your own SDXL model live on the web, the honest options are a GPU host (for example a Hugging Face Space with a GPU, or a small GPU VPS) running the Flask app in `s4/`, or a paid image API behind a serverless function.

The real app still runs locally with `python app.py`, see the main README.
