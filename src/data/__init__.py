# src/data/__init__.py
from .coco_dataset import COCODataset
from .crowdpose_dataset import CrowdPoseDataset
from .transforms import PoseTransform

def build_dataset(cfg, split):
    name = cfg['dataset']['name'].lower()
    if name == 'coco':
        return COCODataset(cfg, split)
    elif name == 'crowdpose':
        return CrowdPoseDataset(cfg, split)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose 'coco' or 'crowdpose'.")
