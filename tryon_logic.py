import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import logging
import os

# Local imports
from utils_mask import get_mask_location, pil_to_binary_mask # Assuming utils_mask.py is in the same dir
from preprocess.humanparsing.run_parsing import Parsing # Ensure these modules are packaged
from preprocess.openpose.run_openpose import OpenPose # Ensure these modules are packaged
from densepose_wrapper import DensePosePredictor # Use the wrapper

# Diffusers/Transformers imports (ensure correct versions are installed)
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor, CLIPVisionModelWithProjection,
    CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Global Variables & Model Loading ---
# It's crucial to load models ONLY ONCE when the worker starts.
# RunPod handler's __init__ or a similar mechanism is ideal.

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BASE_PATH = 'yisol/IDM-VTON' # Or local path in container
DTYPE = torch.float16 # Use float16 for efficiency on GPUs

pipe = None
parsing_model = None
openpose_model = None
densepose_predictor = None # Use the wrapper class instance

tensor_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

def initialize_models():
    """Loads all models into memory. Call this once."""
    global pipe, parsing_model, openpose_model, densepose_predictor
    logger.info("Initializing all models...")

    # --- Load VTON Pipeline ---
    logger.info("Loading VTON Pipeline components...")
    unet = UNet2DConditionModel.from_pretrained(
        BASE_PATH, subfolder="unet", torch_dtype=DTYPE)
    tokenizer_one = AutoTokenizer.from_pretrained(
        BASE_PATH, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(
        BASE_PATH, subfolder="tokenizer_2", use_fast=False)
    text_encoder_one = CLIPTextModel.from_pretrained(
        BASE_PATH, subfolder="text_encoder", torch_dtype=DTYPE)
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
        BASE_PATH, subfolder="text_encoder_2", torch_dtype=DTYPE)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        BASE_PATH, subfolder="image_encoder", torch_dtype=DTYPE)
    vae = AutoencoderKL.from_pretrained(
        BASE_PATH, subfolder="vae", torch_dtype=DTYPE)
    unet_encoder = UNet2DConditionModel_ref.from_pretrained(
        BASE_PATH, subfolder="unet_encoder", torch_dtype=DTYPE)
    scheduler = DDPMScheduler.from_pretrained(
        BASE_PATH, subfolder="scheduler")

    pipe = TryonPipeline.from_pretrained(
        BASE_PATH, # This might load some components again, check if necessary
        unet=unet, vae=vae, feature_extractor=CLIPImageProcessor(),
        text_encoder=text_encoder_one, text_encoder_2=text_encoder_two,
        tokenizer=tokenizer_one, tokenizer_2=tokenizer_two,
        scheduler=scheduler, image_encoder=image_encoder, torch_dtype=DTYPE)
    pipe.unet_encoder = unet_encoder

    pipe.to(DEVICE)
    pipe.unet_encoder.to(DEVICE) # Ensure this is also moved
    logger.info("VTON Pipeline loaded.")

    # --- Load Preprocessing Models ---
    logger.info("Loading preprocessing models (OpenPose, Parsing)...")
    # Assuming device ID 0 is the target GPU
    parsing_model = Parsing(0) # Pass GPU ID if needed by the class
    openpose_model = OpenPose(0) # Pass GPU ID if needed by the class
    # Move OpenPose internal model to device if not done automatically
    if hasattr(openpose_model, 'preprocessor') and hasattr(openpose_model.preprocessor, 'body_estimation') and hasattr(openpose_model.preprocessor.body_estimation, 'model'):
         openpose_model.preprocessor.body_estimation.model.to(DEVICE)
    else:
        logger.warning("Could not explicitly move OpenPose model to device.")
    logger.info("Preprocessing models loaded.")

    # --- Initialize DensePose ---
    logger.info("Loading DensePose model...")
    densepose_predictor = DensePosePredictor() # Uses device set in its config
    logger.info("DensePose model loaded.")

    logger.info("All models initialized successfully.")


def start_tryon(
    human_img_pil: Image.Image,
    garm_img_pil: Image.Image,
    mask_img_pil: Image.Image, # Allow passing explicit mask
    garment_des: str,
    use_auto_mask: bool,
    use_crop: bool,
    denoise_steps: int,
    seed: int
    ) -> tuple[Image.Image, Image.Image]:
    """
    Performs virtual try-on.

    Args:
        human_img_pil: PIL Image of the human (RGB).
        garm_img_pil: PIL Image of the garment (RGB).
        mask_img_pil: PIL Image of the mask (optional, used if use_auto_mask=False).
        garment_des: Text description of the garment.
        use_auto_mask: If True, generate mask using OpenPose and Parsing.
        use_crop: If True, crop the human image.
        denoise_steps: Number of diffusion steps.
        seed: Random seed for generation.

    Returns:
        Tuple containing (result_image_pil, visualization_mask_pil).
    """
    global pipe, parsing_model, openpose_model, densepose_predictor
    if not all([pipe, parsing_model, openpose_model, densepose_predictor]):
         raise RuntimeError("Models not initialized. Call initialize_models() first.")

    # --- Image Preprocessing ---
    garm_img = garm_img_pil.convert("RGB").resize((768, 1024))
    human_img_orig = human_img_pil.convert("RGB")

    if use_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        human_img = human_img_orig.crop((left, top, right, bottom)).resize((768, 1024))
        crop_size = human_img.size # Keep original crop size for pasting back
    else:
        human_img = human_img_orig.resize((768, 1024))
        left, top = 0, 0 # Set crop coords for potential paste later

    # --- Mask Generation ---
    human_img_for_masking = human_img.resize((384, 512)) # Size used in original code
    if use_auto_mask:
        logger.info("Generating mask automatically...")
        keypoints = openpose_model(human_img_for_masking)
        model_parse, _ = parsing_model(human_img_for_masking)
        # Assuming 'upper_body' category for now, make this configurable if needed
        mask, mask_gray_pil = get_mask_location('hd', "upper_body", model_parse, keypoints, width=768, height=1024)
        logger.info("Auto mask generated.")
    else:
        logger.info("Using provided mask.")
        if mask_img_pil is None:
            raise ValueError("Mask image must be provided if use_auto_mask is False.")
        # Ensure mask is binary (0 or 1) and correct size
        mask = pil_to_binary_mask(mask_img_pil.convert("L").resize((768, 1024))) # L ensures grayscale
        # Create a grayscale vis version (optional, like original code)
        mask_np = (1 - transforms.ToTensor()(mask)) * tensor_transform(human_img)
        mask_gray_pil = to_pil_image((mask_np + 1.0) / 2.0)
        logger.info("Provided mask processed.")


    # --- DensePose Generation ---
    logger.info("Generating DensePose image...")
    # Pass the *original sized* or *cropped* human image before final resize
    # DensePose wrapper should handle internal resizing if needed
    # Original code used 384x512 input for DensePose
    pose_img = densepose_predictor.get_densepose_image(human_img.resize((384,512))) # Use the wrapper
    pose_img = pose_img.resize((768, 1024)) # Ensure final size matches pipeline input
    logger.info("DensePose image generated.")

    # --- Prepare Tensors for Pipeline ---
    pose_tensor = tensor_transform(pose_img).unsqueeze(0).to(DEVICE, DTYPE)
    garm_tensor = tensor_transform(garm_img).unsqueeze(0).to(DEVICE, DTYPE)
    human_tensor = tensor_transform(human_img).unsqueeze(0).to(DEVICE, DTYPE) # Needed? Pipe takes PIL
    mask_tensor = transforms.ToTensor()(mask).unsqueeze(0).to(DEVICE, DTYPE) # Mask should be 0-1

    # --- Run VTON Pipeline ---
    logger.info("Running VTON generation pipeline...")
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE != 'cpu' and DTYPE == torch.float16)):
            # Encode prompts
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality, blurry, noisy"
            prompt_embeds, neg_prompt_embeds, pooled_prompt_embeds, neg_pooled_prompt_embeds = pipe.encode_prompt(
                prompt, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)

            prompt_c = "a photo of " + garment_des
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                prompt_c, num_images_per_prompt=1, do_classifier_free_guidance=False) # No negative prompt needed?

            generator = torch.Generator(DEVICE).manual_seed(seed) if seed is not None else None

            images = pipe(
                prompt_embeds=prompt_embeds.to(DEVICE, DTYPE),
                negative_prompt_embeds=neg_prompt_embeds.to(DEVICE, DTYPE),
                pooled_prompt_embeds=pooled_prompt_embeds.to(DEVICE, DTYPE),
                negative_pooled_prompt_embeds=neg_pooled_prompt_embeds.to(DEVICE, DTYPE),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0, # Check if this is the right param for img2img/inpaint
                pose_img=pose_tensor, # Pass the tensor
                text_embeds_cloth=prompt_embeds_c.to(DEVICE, DTYPE),
                cloth=garm_tensor,
                mask_image=mask, # Pass the PIL mask (0/1 or 0/255, check pipe docs)
                image=human_img, # Pass the PIL image
                height=1024,
                width=768,
                guidance_scale=2.0,
                # ip_adapter_image = garm_img.resize((768,1024)), # Check if pipe supports this directly
            ).images # Access the .images attribute

    logger.info("VTON pipeline finished.")
    result_img = images[0]

    # --- Post-processing ---
    if use_crop:
        # Paste the result back onto the original image
        logger.info("Pasting cropped result back.")
        # Resize result back to the size *before* the 768x1024 resize
        out_img_resized = result_img.resize(crop_size)
        human_img_orig.paste(out_img_resized, (int(left), int(top)))
        final_image = human_img_orig
    else:
        final_image = result_img

    return final_image, mask_gray_pil # Return final image and the visual mask