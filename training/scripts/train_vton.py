"""Train the VTON warping + composition model.

Curriculum learning in 3 stages:
  Stage 1: Warping only (freeze composition) — 30 epochs
  Stage 2: Joint warping + composition — 50 epochs
  Stage 3: Add PatchGAN discriminator — 40 epochs

Usage:
    python training/scripts/train_vton.py --config training/configs/vton_baseline.yaml
    python training/scripts/train_vton.py --config training/configs/vton_baseline.yaml --stage 1
"""

# intent: curriculum training script for TPS warping + composition U-Net
# status: done
# next: run after downloading VITON-HD and extracting features
# confidence: high

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)

# Add service paths
sys.path.insert(0, str(Path("services/vton-service")))


class VTONDataset(Dataset):
    """Dataset for VTON training pairs."""

    def __init__(self, pairs_file: str, data_dir: str, input_size: tuple[int, int], max_samples: int | None = None):
        import json
        import random
        with open(pairs_file) as f:
            self.pairs = json.load(f)
        if max_samples and max_samples < len(self.pairs):
            random.seed(42)
            self.pairs = random.sample(self.pairs, max_samples)
        self.data_dir = Path(data_dir)
        self.input_size = input_size  # (W, H)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        person = self._load_image(pair["person"]["image"])
        garment = self._load_image(pair["garment"]["image"])

        # Load precomputed features (stored alongside person/garment dirs)
        pair_id = pair["id"]
        heatmaps = self._load_npy_path(self.data_dir / "heatmaps" / f"{pair_id}.npy")
        parsing = self._load_parsing(self.data_dir / "parsing" / f"{pair_id}.png")

        return {
            "person": person,
            "garment": garment,
            "heatmaps": heatmaps if heatmaps is not None else torch.zeros(18, self.input_size[1], self.input_size[0]),
            "parsing": parsing if parsing is not None else torch.zeros(self.input_size[1], self.input_size[0]),
            "ground_truth": person,  # paired: GT is the original person image
        }

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(self.data_dir / path).convert("RGB")
        img = img.resize(self.input_size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_npy_path(self, path: Path) -> torch.Tensor | None:
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        return torch.from_numpy(arr)

    def _load_parsing(self, path: Path) -> torch.Tensor | None:
        if not path.exists():
            return None
        img = Image.open(path).convert("L")
        img = img.resize(self.input_size, Image.NEAREST)
        return torch.from_numpy(np.array(img, dtype=np.float32))


class PerceptualLoss(nn.Module):
    """VGG-19 perceptual loss."""

    def __init__(self):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:18], # relu3_4
            vgg[18:27], # relu4_4
        ])
        for param in self.parameters():
            param.requires_grad = False

        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for block, weight in zip(self.blocks, self.weights):
            x = block(x)
            y = block(y)
            loss = loss + weight * F.mse_loss(x, y)
        return loss


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for adversarial training."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_test_grid(model, dataset, device, output_path: Path, n_samples: int = 20):
    """Generate a visual grid of test outputs for user evaluation."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    indices = list(range(min(n_samples, len(dataset))))
    images = []

    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            person = batch["person"].unsqueeze(0).to(device)
            garment = batch["garment"].unsqueeze(0).to(device)
            heatmaps = batch["heatmaps"].unsqueeze(0).to(device)

            warped = model.warp(person, garment)
            result, _ = model(person, garment, heatmaps)

            # Stack: person | garment | warped | result
            row = torch.cat([person, garment, warped, result], dim=3).squeeze(0)
            images.append(row)

    grid = torch.cat(images, dim=1)  # (3, N*H, 4*W)
    grid_np = (grid.cpu().clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()
    Image.fromarray(grid_np).save(output_path)
    logger.info("Test grid saved to %s (%d samples)", output_path, len(indices))


def train(config_path: str, stage: int | None = None, device_override: str | None = None):
    """Run curriculum training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        ))
    logger.info("Training on device: %s", device)

    # Load trainable model
    from src.warping.trainable import TrainableVTON

    model_cfg = config.get("model", {})
    n_side = int(model_cfg.get("tps_control_points", 25) ** 0.5)
    model = TrainableVTON(
        n_control_points_per_side=n_side,
        composition_channels=model_cfg.get("composition_channels", 24),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.2fM", param_count / 1e6)

    # Load dataset — data_dir is the parent directory of the pairs file
    pairs_path = Path(config["data"]["pairs_file"])
    data_dir = str(pairs_path.parent)
    max_samples = config["data"].get("max_samples", None)
    dataset = VTONDataset(
        pairs_file=config["data"]["pairs_file"],
        data_dir=data_dir,
        input_size=tuple(config["data"]["input_size"]),
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"].get("num_workers", 0),
        pin_memory=device.type != "cpu",
    )

    logger.info("Dataset: %d pairs, batch_size: %d", len(dataset), config["training"]["batch_size"])

    perceptual_loss = PerceptualLoss().to(device)
    checkpoint_dir = Path(config["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if exists
    resume_path = checkpoint_dir / "latest.pt"
    start_stage = 1
    if resume_path.exists() and stage is None:
        logger.info("Resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_stage = ckpt.get("stage", 1)

    stages_to_run = [stage] if stage else list(range(start_stage, 4))

    for current_stage in stages_to_run:
        if current_stage == 1:
            _train_stage1(model, loader, config, device, checkpoint_dir)
        elif current_stage == 2:
            _train_stage2(model, loader, config, device, checkpoint_dir, perceptual_loss)
        elif current_stage == 3:
            _train_stage3(model, loader, config, device, checkpoint_dir, perceptual_loss)

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": stages_to_run[-1],
        "config": config,
    }, checkpoint_dir / "final.pt")

    # Generate test grid for visual evaluation
    generate_test_grid(
        model, dataset, device,
        checkpoint_dir / "test_grid.png",
        n_samples=20,
    )

    logger.info("Training complete. Final model saved to %s", checkpoint_dir / "final.pt")


def _train_stage1(model, loader, config, device, checkpoint_dir):
    """Stage 1: Warping only — freeze composition, train encoders + TPS."""
    logger.info("=== Stage 1: Warping only ===")
    stage1 = config["training"]["stage1"]

    # Freeze composition net
    for param in model.composition_net.parameters():
        param.requires_grad = False

    trainable_params = list(model.person_encoder.parameters()) + \
                       list(model.garment_encoder.parameters()) + \
                       list(model.tps_module.parameters())

    optimizer = torch.optim.Adam(trainable_params, lr=stage1["lr"])
    grad_clip = config["training"].get("gradient_clip", 1.0)

    for epoch in range(1, stage1["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            gt = batch["ground_truth"].to(device)

            warped = model.warp(person, garment)
            loss = F.l1_loss(warped, gt) * stage1["losses"]["l1"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Stage 1 Epoch %d/%d — Loss: %.4f", epoch, stage1["epochs"], avg_loss)

    # Unfreeze composition for later stages
    for param in model.composition_net.parameters():
        param.requires_grad = True

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": 1,
    }, checkpoint_dir / "latest.pt")
    torch.save(model.state_dict(), checkpoint_dir / "stage1_final.pt")
    logger.info("Stage 1 complete")


def _train_stage2(model, loader, config, device, checkpoint_dir, perceptual_loss):
    """Stage 2: Joint warping + composition — L1 + perceptual loss."""
    logger.info("=== Stage 2: Joint warping + composition ===")
    stage2 = config["training"]["stage2"]

    optimizer = torch.optim.Adam(model.parameters(), lr=stage2["lr"])
    grad_clip = config["training"].get("gradient_clip", 1.0)
    save_every = config["training"].get("save_every", 10)

    for epoch in range(1, stage2["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in loader:
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            gt = batch["ground_truth"].to(device)

            result, mask = model(person, garment, heatmaps)
            l1 = F.l1_loss(result, gt)
            perc = perceptual_loss(result, gt)
            loss = l1 * stage2["losses"]["l1"] + perc * stage2["losses"]["perceptual"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Stage 2 Epoch %d/%d — Loss: %.4f", epoch, stage2["epochs"], avg_loss)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"stage2_epoch{epoch}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": 2,
    }, checkpoint_dir / "latest.pt")
    logger.info("Stage 2 complete")


def _train_stage3(model, loader, config, device, checkpoint_dir, perceptual_loss):
    """Stage 3: Adversarial training with PatchGAN discriminator."""
    logger.info("=== Stage 3: Adversarial training ===")
    stage3 = config["training"]["stage3"]

    discriminator = PatchGANDiscriminator().to(device)
    opt_g = torch.optim.Adam(model.parameters(), lr=stage3["lr_generator"])
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=stage3["lr_discriminator"])
    grad_clip = config["training"].get("gradient_clip", 1.0)
    save_every = config["training"].get("save_every", 10)

    for epoch in range(1, stage3["epochs"] + 1):
        model.train()
        discriminator.train()
        total_g_loss = 0.0

        for batch in loader:
            person = batch["person"].to(device)
            garment = batch["garment"].to(device)
            heatmaps = batch["heatmaps"].to(device)
            gt = batch["ground_truth"].to(device)

            # Generator step
            result, mask = model(person, garment, heatmaps)
            l1 = F.l1_loss(result, gt)
            perc = perceptual_loss(result, gt)
            d_fake = discriminator(result)
            g_adv = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            g_loss = (l1 * stage3["losses"]["l1"] +
                      perc * stage3["losses"]["perceptual"] +
                      g_adv * stage3["losses"]["adversarial"])

            opt_g.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt_g.step()

            # Discriminator step
            d_real = discriminator(gt.detach())
            d_fake = discriminator(result.detach())
            d_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) +
                      F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))) / 2

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            total_g_loss += g_loss.item()

        avg_loss = total_g_loss / len(loader)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("Stage 3 Epoch %d/%d — G Loss: %.4f", epoch, stage3["epochs"], avg_loss)

        if epoch % save_every == 0:
            torch.save(model.state_dict(), checkpoint_dir / f"stage3_epoch{epoch}.pt")

    torch.save({
        "model_state_dict": model.state_dict(),
        "stage": 3,
    }, checkpoint_dir / "latest.pt")
    logger.info("Stage 3 complete")


def main():
    parser = argparse.ArgumentParser(description="Train VTON model")
    parser.add_argument("--config", type=str, default="training/configs/vton_baseline.yaml")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], default=None,
                        help="Run only a specific stage (default: all)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu, cuda, mps). Default: auto-detect")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args.config, stage=args.stage, device_override=args.device)


if __name__ == "__main__":
    main()
