# Import dependencies
import gradio as gr
from diffusers import StableDiffusionPipeline
from transformers import CLIPProcessor, CLIPModel
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import ToTensor
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load CLIP model for text-image similarity
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load Stable Diffusion pipeline
print("Loading Stable Diffusion pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

# Load DeepLabV3+ model for semantic segmentation
print("Loading DeepLabV3+ segmentation model...")
seg_model = deeplabv3_resnet50(pretrained=True).eval()

# Function to process inputs and generate outputs
def generate_image(prompt, reference_image=None):
    try:
        # Generate image using Stable Diffusion
        generated_image = pipe(prompt).images[0]

        # Handle reference image if provided
        if reference_image:
            reference_image = reference_image.resize((512, 512))
            # Calculate similarity between text and reference image
            inputs = clip_processor(text=prompt, images=reference_image, return_tensors="pt", padding=True)
            outputs = clip_model(**inputs)
            similarity = outputs.logits_per_image.softmax(dim=1).item()
            print(f"Text-Image Similarity: {similarity}")

        # Apply semantic segmentation
        input_tensor = ToTensor()(generated_image).unsqueeze(0)
        with torch.no_grad():
            output = seg_model(input_tensor)["out"][0]
        segmentation_mask = output.argmax(0).byte().cpu().numpy()

        # Overlay segmentation mask (example visualization)
        plt.imshow(np.asarray(generated_image))
        plt.imshow(segmentation_mask, alpha=0.5, cmap="jet")
        plt.axis("off")
        plt.savefig("segmented_output.png")
        plt.close()

        return generated_image,
    except Exception as e:
        return f"Error: {str(e)}"


# Gradio interface
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter a text prompt", lines=2),
        gr.Image(label="Reference Image (Optional)", type="pil"),
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Multi-Modal Image Generator",
    description="Generate images from text prompts with optional reference images and segmentation.",
)

# Launch the Gradio app
interface.launch()