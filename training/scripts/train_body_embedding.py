"""Train the body embedding CNN with contrastive learning.

Learns a 128-d body shape descriptor from pose heatmaps (18ch) + parsing
maps (20ch one-hot) using triplet loss with online semi-hard mining.

Positive pairs: same person with augmented heatmaps/parsing.
Negative pairs: different persons.

Usage:
    python training/scripts/train_body_embedding.py --config training/configs/body_embedding.yaml
"""

# intent: contrastive training for body embedding CNN
# status: done
# next: run after feature extraction on VITON-HD
# confidence: high

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path("services/feature-service")))

NUM_POSE_CHANNELS = 18
NUM_PARSING_CLASSES = 20
SPATIAL_H = 64
SPATIAL_W = 48


class BodyFeatureDataset(Dataset):
    """Dataset of (pose_heatmaps, parsing_map) pairs for contrastive learning."""

    def __init__(
        self,
        pairs_file: str,
        data_dir: str,
        noise_std: float = 0.05,
        parsing_dropout: float = 0.1,
    ):
        with open(pairs_file) as f:
            self.pairs = json.load(f)
        self.data_dir = Path(data_dir)
        self.noise_std = noise_std
        self.parsing_dropout = parsing_dropout

        # Filter to pairs that have extracted features
        self.valid_pairs = []
        for p in self.pairs:
            pair_id = p["id"]
            hm_path = self.data_dir / "heatmaps" / f"{pair_id}.npy"
            parse_path = self.data_dir / "parsing" / f"{pair_id}.png"
            if hm_path.exists() and parse_path.exists():
                self.valid_pairs.append(p)

        logger.info("Found %d/%d pairs with extracted features", len(self.valid_pairs), len(self.pairs))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        pair = self.valid_pairs[idx]
        pair_id = pair["id"]

        # Load heatmaps (18, 64, 48)
        heatmaps = np.load(self.data_dir / "heatmaps" / f"{pair_id}.npy").astype(np.float32)
        if heatmaps.shape != (NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W):
            heatmaps = np.zeros((NUM_POSE_CHANNELS, SPATIAL_H, SPATIAL_W), dtype=np.float32)

        # Load parsing map and convert to one-hot (20, H, W)
        parsing_img = Image.open(self.data_dir / "parsing" / f"{pair_id}.png").convert("L")
        parsing_img = parsing_img.resize((SPATIAL_W, SPATIAL_H), Image.NEAREST)
        parsing = np.array(parsing_img, dtype=np.uint8)
        parsing_onehot = np.zeros((NUM_PARSING_CLASSES, SPATIAL_H, SPATIAL_W), dtype=np.float32)
        for c in range(NUM_PARSING_CLASSES):
            parsing_onehot[c] = (parsing == c).astype(np.float32)

        # Anchor: original features
        anchor = np.concatenate([heatmaps, parsing_onehot], axis=0)  # (38, 64, 48)

        # Positive: augmented version of same features
        aug_heatmaps = heatmaps + np.random.randn(*heatmaps.shape).astype(np.float32) * self.noise_std
        aug_parsing = parsing_onehot.copy()
        if np.random.random() < self.parsing_dropout:
            # Randomly drop a parsing channel
            drop_ch = np.random.randint(0, NUM_PARSING_CLASSES)
            aug_parsing[drop_ch] = 0.0
        positive = np.concatenate([aug_heatmaps, aug_parsing], axis=0)

        return {
            "anchor": torch.from_numpy(anchor),
            "positive": torch.from_numpy(positive),
            "idx": idx,
        }


class OnlineTripletLoss(nn.Module):
    """Triplet loss with online semi-hard negative mining."""

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor_emb: torch.Tensor, positive_emb: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss using in-batch negatives.

        For each anchor, the positive is the augmented version. Negatives
        are all other anchors in the batch (different persons).

        Args:
            anchor_emb:   (B, 128) L2-normalized embeddings
            positive_emb: (B, 128) L2-normalized embeddings
        """
        B = anchor_emb.size(0)
        if B < 2:
            return torch.tensor(0.0, device=anchor_emb.device, requires_grad=True)

        # Positive distances: anchor[i] vs positive[i]
        pos_dist = F.pairwise_distance(anchor_emb, positive_emb)  # (B,)

        # All pairwise distances between anchors (for negative mining)
        # dist_matrix[i, j] = distance between anchor_emb[i] and anchor_emb[j]
        dist_matrix = torch.cdist(anchor_emb, anchor_emb)  # (B, B)

        # Semi-hard mining: find negatives where pos_dist < neg_dist < pos_dist + margin
        # Mask out self-distances without in-place modification
        self_mask = torch.eye(B, device=anchor_emb.device, dtype=torch.bool)
        dist_matrix = dist_matrix.masked_fill(self_mask, float("inf"))

        losses = []
        for i in range(B):
            neg_dists = dist_matrix[i]

            # Semi-hard: neg_dist > pos_dist and neg_dist < pos_dist + margin
            semi_hard_mask = (neg_dists > pos_dist[i]) & (neg_dists < pos_dist[i] + self.margin)

            if semi_hard_mask.any():
                hardest_neg = neg_dists[semi_hard_mask].min()
            else:
                # Fallback to hardest negative
                hardest_neg = neg_dists.min()

            triplet_loss = F.relu(pos_dist[i] - hardest_neg + self.margin)
            losses.append(triplet_loss)

        return torch.stack(losses).mean()


def train(config_path: str):
    """Run body embedding training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    ))
    logger.info("Training on device: %s", device)

    # Build model
    from src.body_embedding import _build_network
    BodyEmbeddingCNN = _build_network()
    model = BodyEmbeddingCNN().to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %.1fK", param_count / 1e3)

    # Dataset
    aug_cfg = config["training"].get("augmentation", {})
    dataset = BodyFeatureDataset(
        pairs_file=config["data"]["pairs_file"],
        data_dir=config["data"]["dataset_dir"],
        noise_std=aug_cfg.get("heatmap_noise_std", 0.05),
        parsing_dropout=aug_cfg.get("parsing_dropout", 0.1),
    )

    if len(dataset) == 0:
        logger.error("No valid pairs found. Run feature extraction first.")
        return

    # Train/test split
    n_test = max(1, int(len(dataset) * config["data"].get("test_split", 0.1)))
    n_train = len(dataset) - n_test
    train_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_test])

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # Loss + optimizer
    loss_cfg = config["training"].get("loss", {})
    criterion = OnlineTripletLoss(margin=loss_cfg.get("margin", 0.3))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"],
    )

    checkpoint_dir = Path(config["checkpoints"]["dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)

            loss = criterion(anchor_emb, positive_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / max(len(train_loader), 1)

        # Evaluate
        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    anchor = batch["anchor"].to(device)
                    positive = batch["positive"].to(device)
                    anchor_emb = model(anchor)
                    positive_emb = model(positive)
                    test_loss += criterion(anchor_emb, positive_emb).item()

            avg_test = test_loss / max(len(test_loader), 1)
            logger.info(
                "Epoch %d/%d — Train Loss: %.4f, Test Loss: %.4f",
                epoch, config["training"]["epochs"], avg_loss, avg_test,
            )

            if avg_test < best_loss:
                best_loss = avg_test
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")

    # Save final
    torch.save(model.state_dict(), checkpoint_dir / "final.pt")
    logger.info("Training complete. Best test loss: %.4f", best_loss)
    logger.info("Checkpoint: %s", checkpoint_dir / "best.pt")


def main():
    parser = argparse.ArgumentParser(description="Train Body Embedding CNN")
    parser.add_argument("--config", type=str, default="training/configs/body_embedding.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
