# Vercel UI preview (demo only)

This folder is a static preview of the text to image UI, built for reviewing the design on Vercel.

* `index.html`: the same UI as `s4/templates/index.html`, with the backend calls replaced by a simulated generation.
* `demo.jpg`: a sample image shown for every prompt in this preview.

## Why a preview, not the real app?

The real app runs SDXL + LCM-LoRA on a CUDA GPU, with a 7 GB model. Vercel is serverless hosting with no GPU, strict function size and time limits, so the real generation cannot run there. The preview lets you click through the UI, try the controls, and judge the design on phone and desktop.

The real app still runs locally with `python app.py`, see the main README.
