from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
import os
import uuid

app = Flask(__name__)

# Set up output folder
OUTPUT_FOLDER = "static/generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_PATH = r"C:\Users\kesha\models\sdxl"  # Local SDXL Model Path
LORA_PATH = r"C:\Users\kesha\models\lcm_lora_sdxl"  # LoRA Weights Path

print("Loading model... (This may take some time)")

# Set PyTorch to optimize memory usage
torch.backends.cudnn.benchmark = True  

# Load SDXL with float16 to save memory
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL_PATH, 
    torch_dtype=torch.float16
)

# Apply LCM LoRA
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda", dtype=torch.float16)
pipe.load_lora_weights(LORA_PATH)
pipe.fuse_lora()

print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    print(f"Generating image for: {prompt}")

    try:
        # Lower resolution to reduce memory usage
        image = pipe(
            prompt=prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            width=768,   # Change to 512 or 768
            height=768
        ).images[0]

        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(OUTPUT_FOLDER, filename)
        image.save(image_path)

        print(f"Image saved: {image_path}")
        return jsonify({"image_url": f"/static/generated_images/{filename}"})

    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM! Trying to free memory...")
        torch.cuda.empty_cache()
        return jsonify({"error": "CUDA Out of Memory. Try reducing image size."}), 500

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": "Failed to generate image"}), 500

@app.route('/static/generated_images/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
