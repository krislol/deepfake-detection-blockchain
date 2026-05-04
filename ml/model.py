"""
EfficientNet-B0 deepfake detection model.
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import List, Tuple, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")


def get_data_transforms(input_size: int = 224) -> Dict:
    return {
        'train': transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'eval': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }


class FaceDataset(Dataset):
    def __init__(self, face_images, labels, transform=None):
        self.face_images = face_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        image = self.face_images[idx]
        label = self.labels[idx]
        if isinstance(image, np.ndarray):
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def _build_efficientnet() -> nn.Module:
    """Build EfficientNet-B0 with custom classifier head."""
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    num_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 2)
    )
    return backbone


class DeepfakeDetector:
    """EfficientNet-B0 fine-tuned on Celeb-DF v2 for binary real/fake classification."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or str(DEVICE)
        self.transforms = get_data_transforms()

        self.model = _build_efficientnet().to(self.device)

        if model_path and os.path.exists(model_path):
            self._load_weights(model_path)
        else:
            logger.info("No trained weights found — using ImageNet pretrained weights")

        self.model.eval()

    def _load_weights(self, model_path: str):
        """Load saved weights, handling backbone-wrapper and architecture mismatches."""
        raw = torch.load(model_path, map_location=self.device)

        # Unwrap checkpoint dicts (e.g. {'model_state_dict': ..., 'epoch': ...})
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            state_dict = raw['model_state_dict']
        elif hasattr(raw, 'state_dict'):
            state_dict = raw.state_dict()
        else:
            state_dict = raw

        # strip 'backbone.' prefix if the model was saved inside a wrapper class
        if state_dict and all(k.startswith('backbone.') for k in state_dict.keys()):
            state_dict = {k[len('backbone.'):]: v for k, v in state_dict.items()}

        try:
            self.model.load_state_dict(state_dict, strict=True)
            logger.info(f"✓ Loaded trained weights from: {model_path}")
        except RuntimeError as e:
            logger.warning(f"Strict load failed, trying partial load: {e}")
            model_dict = self.model.state_dict()
            matched = {k: v for k, v in state_dict.items()
                       if k in model_dict and v.shape == model_dict[k].shape}
            model_dict.update(matched)
            self.model.load_state_dict(model_dict)
            logger.info(f"Partial load: {len(matched)}/{len(model_dict)} layers loaded")

    def predict_single_face(self, face_image: np.ndarray) -> Tuple[float, Dict]:
        if face_image.dtype != np.float32:
            face_image = face_image.astype(np.float32) / 255.0
        pil_img = Image.fromarray((face_image * 255).astype(np.uint8))
        tensor = self.transforms['eval'](pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fp = torch.softmax(self.model(tensor), dim=1)[0, 1].item()
        return fp, {'fake_prob': fp, 'real_prob': 1-fp, 'confidence': max(fp, 1-fp)}

    def predict_batch(self, face_images: np.ndarray, batch_size: int = 32) -> Tuple[List[float], List[Dict]]:
        predictions, metadata = [], []
        dataset = FaceDataset(list(face_images), [0]*len(face_images), self.transforms['eval'])
        for batch_imgs, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            with torch.no_grad():
                probs = torch.softmax(self.model(batch_imgs.to(self.device)), dim=1)
            for fp in probs[:, 1].cpu().numpy():
                predictions.append(float(fp))
                metadata.append({'fake_prob': float(fp), 'real_prob': 1-float(fp),
                                  'confidence': max(float(fp), 1-float(fp))})
        return predictions, metadata

    def load_model(self, model_path: str):
        self._load_weights(model_path)
        self.model.eval()

    def save_model(self, model_path: str):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")


DeepfakeDetectionModel = DeepfakeDetector
