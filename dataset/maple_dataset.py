from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import numpy as np
from PIL import Image
import random
import cv2

def make_random_background(size=(224, 224)):
    bg = cv2.imread("dataset/augmentation_asset.png")
    bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)).resize(size)
    return bg

def composite_foreground_with_background(foreground: Image.Image, background: Image.Image):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA")
    
    bg_w, bg_h = background.size
    fg_w, fg_h = foreground.size
    a = (224*min(fg_h,fg_w)/max(fg_w,fg_h))/((fg_h*fg_w)**(1/2))
    new_w = int(fg_w * a)
    new_h = int(fg_h * a)
    resized_foreground = foreground.resize((new_w, new_h), resample=Image.BICUBIC)

    scale = random.uniform(0.55,0.75)
    new_w = int(new_w * scale)
    new_h = int(new_h * scale)
    resized_foreground = foreground.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # 랜덤 위치 계산
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x_offset = random.randint(int(max_x*2/5), int(max_x*3/5))
    y_offset = random.randint(int(max_y*2/8), int(max_y*3/8))
    
    canvas = Image.new("RGBA", background.size, (0, 0, 0, 0))
    canvas.paste(resized_foreground, (x_offset, y_offset), resized_foreground)

    # 전경의 알파 채널을 기준으로 합성
    composite = Image.alpha_composite(background, canvas)
    return composite.convert("RGB")


# ---------------------
# 1. 데이터셋 정의
# ---------------------
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomAffine(degrees=7, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.evel_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGBA")
        bg = make_random_background((224, 224))
        composite = composite_foreground_with_background(img, bg)
        tensor_img = self.transform(composite)
        label = idx  # 이미지마다 고유 ID
        return tensor_img, label