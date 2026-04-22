# intent: top-level VTONModel that orchestrates TPS warping + composition
#         for virtual try-on inference; singleton so weights load once
# status: done
# next: add garment_type-specific parsing labels; benchmark latency
# confidence: high

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from .composition import CompositionUNet
from .tps import TPSWarping

logger = logging.getLogger(__name__)

# Preprocessing transforms shared across calls
_IMG_SIZE = (256, 192)
_TO_TENSOR = transforms.Compose([
    transforms.Resize(_IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Parsing label that corresponds to the upper-body garment region
_GARMENT_LABELS: dict[str, list[int]] = {
    "upper": [5, 6, 7],
    "lower": [9, 12],
    "full":  [5, 6, 7, 9, 12],
}


class VTONModel:
    """Singleton wrapper around TPS warping + Composition U-Net."""

    _instance: VTONModel | None = None

    # ------------------------------------------------------------------
    # Singleton helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> VTONModel:
        """Return the singleton, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def is_loaded(cls) -> bool:
        return cls._instance is not None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        model_dir = os.environ.get("MODEL_DIR", "checkpoints")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading VTONModel weights from %s on %s", model_dir, self.device)

        # --- Feature encoders (ResNet-18, shared backbone) ---
        self.person_encoder = self._build_feature_encoder()
        self.garment_encoder = self._build_feature_encoder()

        # --- TPS warping module ---
        # ResNet-18 layer3 outputs 256 channels
        self.tps_module = TPSWarping(in_channels=256)

        # --- Composition U-Net ---
        self.composition_net = CompositionUNet(in_channels=24)

        # Try loading from combined checkpoint (produced by training script)
        # Check both MODEL_DIR/final.pt and MODEL_DIR/vton/final.pt
        combined_path = os.path.join(model_dir, "final.pt")
        vton_subdir_path = os.path.join(model_dir, "vton", "final.pt")
        if not os.path.isfile(combined_path) and os.path.isfile(vton_subdir_path):
            combined_path = vton_subdir_path
        if os.path.isfile(combined_path):
            ckpt = torch.load(combined_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            # Map TrainableVTON keys to VTONModel attributes
            self._load_combined_state_dict(state_dict)
            logger.info("Loaded combined checkpoint from %s", combined_path)
        else:
            # Fallback: load individual component weights
            tps_path = os.path.join(model_dir, "tps_module.pth")
            comp_path = os.path.join(model_dir, "composition_net.pth")
            person_enc_path = os.path.join(model_dir, "person_encoder.pth")
            garment_enc_path = os.path.join(model_dir, "garment_encoder.pth")

            if os.path.isfile(tps_path):
                self.tps_module.load_state_dict(
                    torch.load(tps_path, map_location=self.device, weights_only=True)
                )
            if os.path.isfile(comp_path):
                self.composition_net.load_state_dict(
                    torch.load(comp_path, map_location=self.device, weights_only=True)
                )
            if os.path.isfile(person_enc_path):
                self.person_encoder.load_state_dict(
                    torch.load(person_enc_path, map_location=self.device, weights_only=True)
                )
            if os.path.isfile(garment_enc_path):
                self.garment_encoder.load_state_dict(
                    torch.load(garment_enc_path, map_location=self.device, weights_only=True)
                )

        # Move everything to device and set eval mode
        self.person_encoder.to(self.device).eval()
        self.garment_encoder.to(self.device).eval()
        self.tps_module.to(self.device).eval()
        self.composition_net.to(self.device).eval()

        logger.info("VTONModel ready")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_combined_state_dict(self, state_dict: dict) -> None:
        """Load state dict from TrainableVTON combined checkpoint.

        Maps keys like 'person_encoder.0.weight' → self.person_encoder[0].weight
        """
        for name, module in [
            ("person_encoder", self.person_encoder),
            ("garment_encoder", self.garment_encoder),
            ("tps_module", self.tps_module),
            ("composition_net", self.composition_net),
        ]:
            prefix = f"{name}."
            sub_dict = {
                k[len(prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix)
            }
            if sub_dict:
                module.load_state_dict(sub_dict, strict=False)

    @staticmethod
    def _build_feature_encoder() -> torch.nn.Module:
        """ResNet-18 truncated after layer3 (output: 256-ch feature map)."""
        backbone = models.resnet18(weights=None)
        # Keep everything up to and including layer3
        encoder = torch.nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        )
        return encoder

    @staticmethod
    def _generate_pose_heatmaps(
        pose: np.ndarray,
        height: int,
        width: int,
        n_joints: int = 18,
        sigma: float = 6.0,
    ) -> torch.Tensor:
        """Create (n_joints, H, W) Gaussian heatmaps from pose keypoints.

        Parameters
        ----------
        pose : (n_joints, 2|3) array — x, y[, confidence]
        """
        heatmaps = torch.zeros(n_joints, height, width)
        yy = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)
        xx = torch.arange(0, width, dtype=torch.float32).unsqueeze(0)

        for j in range(min(n_joints, len(pose))):
            x, y = float(pose[j][0]), float(pose[j][1])
            if x < 0 or y < 0:
                continue
            heatmaps[j] = torch.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma ** 2)
            )
        return heatmaps

    @staticmethod
    def _generate_agnostic(
        person_tensor: torch.Tensor,
        parsing: np.ndarray,
        garment_type: str,
    ) -> torch.Tensor:
        """Zero-out garment region in the person image using parsing map.

        Parameters
        ----------
        person_tensor : (3, H, W) normalised tensor
        parsing       : (H_p, W_p) int array with segmentation labels
        garment_type  : one of "upper", "lower", "full"

        Returns
        -------
        agnostic : (3, H, W)
        """
        labels = _GARMENT_LABELS.get(garment_type, _GARMENT_LABELS["upper"])
        H, W = person_tensor.shape[1], person_tensor.shape[2]

        # Resize parsing to match tensor spatial dims
        parsing_resized = (
            F.interpolate(
                torch.from_numpy(parsing).float().unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="nearest",
            )
            .squeeze()
            .long()
        )

        mask = torch.zeros(H, W, dtype=torch.bool)
        for lbl in labels:
            mask |= parsing_resized == lbl

        agnostic = person_tensor.clone()
        agnostic[:, mask] = 0.0
        return agnostic

    # ------------------------------------------------------------------
    # Public inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def infer(
        self,
        person_img: Image.Image,
        garment_img: Image.Image,
        pose: np.ndarray,
        parsing: np.ndarray,
        garment_type: str = "upper",
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Run full VTON pipeline on a single person + garment pair.

        Parameters
        ----------
        person_img   : PIL RGB image of the person
        garment_img  : PIL RGB image of the garment (flat-lay)
        pose         : (J, 2|3) keypoint array
        parsing      : (H, W) segmentation label map
        garment_type : "upper" | "lower" | "full"

        Returns
        -------
        result_pil : PIL Image with garment applied
        metadata   : dict with diagnostic info
        """
        # 1. Preprocess
        person_t = _TO_TENSOR(person_img).unsqueeze(0).to(self.device)   # (1,3,H,W)
        garment_t = _TO_TENSOR(garment_img).unsqueeze(0).to(self.device)

        H, W = _IMG_SIZE

        # 2. Encode person + garment with ResNet-18 feature encoders
        f_person = self.person_encoder(person_t)    # (1, 256, H', W')
        f_garment = self.garment_encoder(garment_t)

        # 3. Predict TPS control points and warp garment
        warped_garment = self.tps_module(f_garment, f_person, garment_t)  # (1,3,H,W)

        # 4. Generate agnostic person image
        agnostic = self._generate_agnostic(
            person_t.squeeze(0).cpu(), parsing, garment_type
        ).unsqueeze(0).to(self.device)

        # 5. Build pose heatmaps
        pose_hm = self._generate_pose_heatmaps(pose, H, W).unsqueeze(0).to(self.device)

        # 6. Composition U-Net
        result, comp_mask = self.composition_net(agnostic, warped_garment, pose_hm)

        # 7. Convert back to PIL
        result_np = (
            result.squeeze(0)
            .cpu()
            .clamp(-1.0, 1.0)
            .add(1.0)
            .div(2.0)
            .mul(255.0)
            .byte()
            .permute(1, 2, 0)
            .numpy()
        )
        result_pil = Image.fromarray(result_np, mode="RGB")

        metadata: dict[str, Any] = {
            "garment_type": garment_type,
            "output_size": _IMG_SIZE,
            "device": str(self.device),
            "mask_mean": float(comp_mask.mean().item()),
        }

        return result_pil, metadata
