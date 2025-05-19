import torch
from torch.utils.data import Dataset
import albumentations as A
import cv2
import numpy as np

class KuzushijiDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        df: Parsed DataFrame (from exploration notebook)
        img_dir: Path to train_images/
        transform: Albumentations augmentations
        """
        self.df = df.groupby('image_id').agg(list)  # Group chars per image
        self.img_dir = img_dir
        self.transform = transform or A.Compose([
            A.Resize(512, 512),  # Standardize size
            A.Normalize()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.index[idx]
        img = cv2.imread(str(self.img_dir / f"{img_id}.jpg"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get all boxes/chars for this image
        boxes = np.array([
            [x, y, x+w, y+h]  # Convert to Pascal VOC format
            for x, y, w, h in zip(
                self.df.iloc[idx]['x_min'],
                self.df.iloc[idx]['y_min'],
                self.df.iloc[idx]['width'],
                self.df.iloc[idx]['height']
            )
        ])
        labels = self.df.iloc[idx]['unicode']  # For class indices
        
        # Apply augmentations
        transformed = self.transform(
            image=img,
            bboxes=boxes,
            labels=labels
        )
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).float()
        boxes_tensor = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        labels_tensor = torch.tensor([int(lbl[2:], 16) for lbl in transformed['labels']])  # Unicode → int
        
        return {
            'image': img_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor
        }