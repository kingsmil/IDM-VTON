import torch
import logging
import numpy as np
from PIL import Image

# Detectron2 imports (ensure they are available)
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data.detection_utils import read_image, convert_PIL_to_numpy, _apply_exif_orientation

# DensePose imports
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.densepose_outputs_vertex import DensePoseOutputsTextureVisualizer, DensePoseOutputsVertexVisualizer
from densepose.vis.densepose_results import DensePoseResultsFineSegmentationVisualizer # Using dp_segm
from densepose.vis.extractor import CompoundExtractor, create_extractor

# --- Configuration ---
DENSEPOSE_CONFIG = './configs/densepose_rcnn_R_50_FPN_s1x.yaml'
DENSEPOSE_MODEL = './ckpt/densepose/model_final_162be9.pkl' # Make sure this path is correct in the container
DENSEPOSE_VIS = "dp_segm" # From your original command
MIN_SCORE = 0.8 # Default from original script, adjust if needed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DensePosePredictor:
    def __init__(self):
        logger.info("Initializing DensePose Predictor...")
        self.cfg = self._setup_config()
        self.predictor = DefaultPredictor(self.cfg)
        self.visualizer, self.extractor = self._setup_visualizer_extractor()
        logger.info("DensePose Predictor Initialized.")

    def _setup_config(self):
        opts = []
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(DENSEPOSE_CONFIG)
        cfg.MODEL.WEIGHTS = DENSEPOSE_MODEL
        cfg.MODEL.DEVICE = DEVICE
        # Add score threshold directly if needed, or rely on predictor's internal threshold
        # opts.extend(['MODEL.ROI_HEADS.SCORE_THRESH_TEST', str(MIN_SCORE)]) # Example if needed
        cfg.merge_from_list(opts)
        cfg.freeze()
        return cfg

    def _setup_visualizer_extractor(self):
        # Simplified from ShowAction.create_context
        vis_spec = DENSEPOSE_VIS
        # Using DensePoseResultsFineSegmentationVisualizer based on dp_segm
        vis = DensePoseResultsFineSegmentationVisualizer() # Assuming default cfg is enough
        extractor = create_extractor(vis)
        # If you needed multiple visualizers, you'd use CompoundVisualizer/Extractor
        return vis, extractor # Returning single ones for dp_segm

    def get_densepose_image(self, human_pil_image: Image.Image) -> Image.Image:
        """
        Applies DensePose segmentation to a human image.

        Args:
            human_pil_image: PIL Image of the human (RGB).

        Returns:
            PIL Image of the DensePose segmentation visualization.
        """
        # Prepare image for DensePose predictor (expects BGR numpy)
        # Resize *before* converting if needed, predictor might handle it
        # human_img_resized = human_pil_image.resize((384, 512)) # Match original preprocess size
        human_img_oriented = _apply_exif_orientation(human_pil_image)
        human_img_bgr = convert_PIL_to_numpy(human_img_oriented, format="BGR")

        logger.info("Running DensePose prediction...")
        with torch.no_grad():
            outputs = self.predictor(human_img_bgr)["instances"]

        logger.info("Extracting and visualizing DensePose results...")
        # Prepare a blank or gray background for visualization if needed
        # The original code used a grayscale version of the input
        image_gray_np = cv2.cvtColor(human_img_bgr, cv2.COLOR_BGR2GRAY)
        vis_frame = np.tile(image_gray_np[:, :, np.newaxis], [1, 1, 3])

        data = self.extractor(outputs)
        image_vis_np = self.visualizer.visualize(vis_frame, data) # Use the vis_frame

        # Convert result back to PIL (assuming visualize returns BGR)
        image_vis_rgb = cv2.cvtColor(image_vis_np, cv2.COLOR_BGR2RGB)
        pose_img_pil = Image.fromarray(image_vis_rgb)
        logger.info("DensePose visualization complete.")

        # Resize output to target size (e.g., 768x1024) AFTER visualization
        # pose_img_pil = pose_img_pil.resize((768, 1024)) # Match tryon pipeline size

        return pose_img_pil

# --- Optional: Pre-load the model ---
# densepose_predictor = None
# def get_predictor():
#     global densepose_predictor
#     if densepose_predictor is None:
#         densepose_predictor = DensePosePredictor()
#     return densepose_predictor