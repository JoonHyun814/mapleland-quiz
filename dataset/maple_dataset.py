from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import numpy as np
from PIL import Image
import random

def make_random_background(size=(224, 224)):
    w, h = size

    # 배경 기본 색상: 랜덤하게 선택
    base_color = np.array([
        random.randint(50, 230),  # R
        random.randint(50, 230),  # G
        random.randint(50, 230)   # B
    ], dtype=np.uint8)

    # 전체 배경 배열 생성 (색상 단일톤)
    bg = np.ones((h, w, 3), dtype=np.uint8) * base_color

    # 줄무늬 추가
    if random.random() < 0.5:
        stripe_color = np.random.randint(0, 255, size=3)
        for i in range(0, h, random.randint(10, 30)):
            bg[i:i+2, :] = stripe_color

    # 점 추가
    if random.random() < 0.5:
        for _ in range(random.randint(50, 150)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            bg[y, x] = np.random.randint(0, 255, size=3)

    # 약한 노이즈 섞기
    if random.random() < 0.5:
        noise = np.random.randint(0, 30, (h, w, 3), dtype=np.uint8)
        bg = np.clip(bg + noise, 0, 255)

    return Image.fromarray(bg)

def composite_foreground_with_background(foreground: Image.Image, background: Image.Image):
    foreground = foreground.convert("RGBA")
    background = background.convert("RGBA").resize(foreground.size)
    
    bg_w, bg_h = background.size
    scale = random.uniform(0.6,0.9)
    new_w = int(bg_w * scale)
    new_h = int(foreground.height * (new_w / foreground.width))  # 비율 유지
    resized_foreground = foreground.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # 랜덤 위치 계산
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x_offset = random.randint(0, max_x)
    y_offset = random.randint(0, max_y)
    
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
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
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