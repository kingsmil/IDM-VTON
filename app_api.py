from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response # For returning images
from pydantic import BaseModel
import uvicorn
import base64
from PIL import Image
import io
import logging
import os
import torch

# Local imports
from tryon_logic import initialize_models, start_tryon

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Initialize Models ---
# This should run when the module is imported by the server runner (like uvicorn or RunPod handler)
# However, placing it here might cause issues if multiple workers start.
# Best practice: Initialize in a dependency or startup event.
# For RunPod, initialize_models() should ideally be called in the handler's init.
# We'll use FastAPI's dependency injection for a slightly better approach here.

loaded_models = False # Simple flag

def get_models():
    """FastAPI dependency to ensure models are loaded."""
    global loaded_models
    if not loaded_models:
        try:
            logger.info("Dependency: Initializing models...")
            initialize_models()
            loaded_models = True
            logger.info("Dependency: Models initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail="Models are not available")
    # else: logger.debug("Dependency: Models already loaded.") # Optional debug log
    return loaded_models # Or return specific models if needed


app = FastAPI(title="IDM-VTON API")

# --- Request/Response Models ---
class TryonRequest(BaseModel):
    human_image: str # Base64 encoded human image
    garment_image: str # Base64 encoded garment image
    mask_image: str | None = None # Base64 encoded mask image (optional)
    garment_description: str
    use_auto_mask: bool = True
    use_crop: bool = False
    denoise_steps: int = 30
    seed: int = 42

class TryonResponse(BaseModel):
    result_image: str # Base64 encoded result image
    mask_visualization: str # Base64 encoded mask visualization

# --- Helper Functions ---
def decode_base64_image(base64_string: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        raise HTTPException(status_code=400, detail="Invalid base64 image string")

def encode_pil_image(pil_image: Image.Image) -> str:
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG") # Use PNG for lossless saving
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- API Endpoint ---
@app.post("/tryon", response_model=TryonResponse)
async def virtual_tryon(request: TryonRequest, models_loaded: bool = Depends(get_models)):
    """Performs virtual try-on given human and garment images."""
    if not models_loaded: # Redundant check, but safe
        raise HTTPException(status_code=503, detail="Models not ready, please try again later.")

    logger.info(f"Received try-on request: desc='{request.garment_description}', auto_mask={request.use_auto_mask}, crop={request.use_crop}")

    try:
        human_img_pil = decode_base64_image(request.human_image)
        garm_img_pil = decode_base64_image(request.garment_image)
        mask_img_pil = decode_base64_image(request.mask_image) if request.mask_image and not request.use_auto_mask else None

        # --- Execute Try-on Logic ---
        logger.info("Starting try-on process...")
        result_pil, mask_vis_pil = start_tryon(
            human_img_pil=human_img_pil,
            garm_img_pil=garm_img_pil,
            mask_img_pil=mask_img_pil,
            garment_des=request.garment_description,
            use_auto_mask=request.use_auto_mask,
            use_crop=request.use_crop,
            denoise_steps=request.denoise_steps,
            seed=request.seed
        )
        logger.info("Try-on process completed successfully.")

        # --- Encode Results ---
        result_base64 = encode_pil_image(result_pil)
        mask_vis_base64 = encode_pil_image(mask_vis_pil)

        return TryonResponse(result_image=result_base64, mask_visualization=mask_vis_base64)

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception during try-on: {http_exc.detail}")
        raise http_exc # Re-raise FastAPI/validation errors
    except Exception as e:
        logger.error(f"Unexpected error during try-on: {e}", exc_info=True)
        # Clear CUDA cache in case of OOM or other GPU errors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Could add checks for model loading status if needed
    return {"status": "ok"}

# --- For local testing ---
# if __name__ == "__main__":
#     # This part won't run on RunPod Serverless directly, but useful for testing
#     logger.info("Starting FastAPI server locally...")
#     # Make sure models are loaded before starting the server if running locally
#     try:
#         get_models() # Call the dependency func to trigger loading
#     except Exception as e:
#         logger.critical(f"Failed to load models before starting server: {e}", exc_info=True)
#         # Decide whether to exit or start server in a degraded state
#         # exit(1) # Exit if models are critical

#     # Note: RunPod handler will manage the server lifecycle
#     uvicorn.run(app, host="0.0.0.0", port=8000) # Port 8000 is common