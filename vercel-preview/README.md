# Vercel UI preview (design only)

This folder is a static preview of the text to image UI, built for reviewing the design on Vercel. It does not generate images.

* `index.html`: the same clean UI as `s4/templates/index.html`, with light and dark themes and a theme toggle in the top bar. Generation is disabled here on purpose.
* `demo.jpg`: no longer used by the page. It stays in the repo history only.

## Why a preview, not the real app?

The real app runs SDXL + LCM-LoRA on a CUDA GPU, with a 7 GB model. Vercel is serverless hosting with no GPU, strict function size and time limits, so the real generation cannot run there. This preview lets you click through the UI, try the controls, and judge the design in light and dark mode on phone and desktop. Clicking Generate on this page shows a plain note that generation runs in the local app, it never shows a fake image.

The real app still runs locally with `python app.py`, see the main README.
