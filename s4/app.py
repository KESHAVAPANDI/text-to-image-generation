from flask import Flask, request, jsonify, send_from_directory, render_template
import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
import os
import uuid  # To generate unique filenames

app = Flask(__name__)

# Set up output folder
OUTPUT_FOLDER = "static/generated_images"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the model
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

print("Loading model... (This may take some time)")
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16"
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

pipe.load_lora_weights(adapter_id)
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
        # Generate the image with efficient settings
        image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=1.0).images[0]

        # Generate a unique filename
        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(OUTPUT_FOLDER, filename)
        image.save(image_path)

        print(f"Image saved: {image_path}")

        return jsonify({"image_url": f"/static/generated_images/{filename}"})

    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return jsonify({"error": "Failed to generate image"}), 500

@app.route('/static/generated_images/<filename>')
def serve_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
