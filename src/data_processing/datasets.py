import torch
from PIL import Image
from pathlib import Path

from . import IMAGE_EXTENSIONS


def get_img_files(path: str):
    return sorted([str(file) for file in Path(path).glob('*') if
                   file.suffix in IMAGE_EXTENSIONS])


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms=None):
        self.files = get_img_files(path)
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, path
