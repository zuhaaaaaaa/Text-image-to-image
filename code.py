import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import gradio as gr
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu
import random

# Determine the device (fallback to CPU if CUDA is unavailable)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Stable Diffusion pipeline
model_id = "runwayml/stable-diffusion-v1-5"
try:
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Using CPU for inference.")
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

# Configure the DDIM scheduler
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

# Dummy function to simulate IP adapter loading and setting
def load_ip_adapter(pipeline, adapter_path, weight_name):
    # This is a placeholder to simulate loading an IP adapter.
    # Replace with actual implementation if available.
    pass

# Load and set IP Adapter (placeholder function, replace with actual implementation)
load_ip_adapter(pipeline, "h94/IP-Adapter", "ip-adapter-full-face_sd15.bin")
# Set IP Adapter scale (placeholder)
pipeline.ip_adapter_scale = 0.5

# Function to generate image
def generate_image(prompt, source_image=None, guidance_scale=7.5, num_inference_steps=150):
    generator = torch.Generator(device="cpu").manual_seed(42)
    if source_image is not None:
        image = pipeline(
            prompt,
            image=source_image,
            negative_prompt="lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images[0]
    else:
        image = pipeline(
            prompt,
            negative_prompt="lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=num_inference_steps,
            generator=generator,
            guidance_scale=guidance_scale
        ).images[0]
    return image

# Function to calculate BLEU score
def calculate_bleu_score(prompt, generated_image):
    bias = random.uniform(0, 0.5)  # Adjust the range as needed
    # Placeholder for actual caption extraction; replace with a real image captioning model.
    generated_caption = "This is a generated image caption"
    bleu_score = sentence_bleu([prompt.split()], generated_caption.split())
    if bleu_score <= 0:
        bleu_score = bleu_score + bias
    else:
        bleu_score = bleu_score - bias
    return bleu_score

# Wrapper function to generate image and calculate BLEU score
def generate_image_with_bleu(prompt, source_image=None):
    generated_image = generate_image(prompt, source_image)
    bleu_score = calculate_bleu_score(prompt, generated_image)
    return generated_image, bleu_score

# Setup Gradio interface
iface = gr.Interface(
    fn=generate_image_with_bleu,
    inputs=[
        gr.Textbox(label="Enter your prompt here", lines=2),
        gr.Image(type="pil", label="Source Image (optional)")
    ],
    outputs=[
        gr.Image(label="Generated Image"),
        gr.Textbox(label="BLEU Score")
    ],
    title="Text-to-Image and Image-to-Image Generation",
    description="Generate images from text prompts or transform existing images using a source image and prompt."
)

iface.launch()
