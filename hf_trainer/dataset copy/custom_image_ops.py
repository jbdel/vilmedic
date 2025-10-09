from torchvision import transforms
from torchvision.transforms import InterpolationMode
import numpy as np
from PIL import Image

maira2_tf = transforms.Compose([
    transforms.Resize(518, interpolation=InterpolationMode.BICUBIC, antialias=True),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5307]*3, std=[0.2583]*3),
])