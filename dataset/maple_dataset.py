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


def add_doodle(image, num_lines=5, num_circles=3, num_dots=20):
    img = image.copy()
    h, w = img.shape[:2]

    if random.random()<0.2:
        # 랜덤 선 그리기
        for _ in range(num_lines):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            color = tuple([random.randint(0, 255) for _ in range(3)])
            thickness = random.randint(1, 3)
            cv2.line(img, pt1, pt2, color, thickness)
    if random.random()<0.2:
        # 랜덤 원 그리기
        for _ in range(num_circles):
            center = (random.randint(0, w), random.randint(0, h))
            radius = random.randint(5, 20)
            color = tuple([random.randint(0, 255) for _ in range(3)])
            cv2.circle(img, center, radius, color, -1)
    if random.random()<0.2:
        # 랜덤 점 찍기
        for _ in range(num_dots):
            x = random.randint(0, w-1)
            y = random.randint(0, h-1)
            color = tuple([random.randint(0, 255) for _ in range(3)])
            img[y, x] = color

    return img


class DoodleAugment:
    def __call__(self, img_pil):
        img_cv = np.array(img_pil)[:, :, ::-1]  # PIL → BGR
        img_cv = add_doodle(img_cv)
        img_pil = Image.fromarray(img_cv[:, :, ::-1])  # BGR → RGB → PIL
        return img_pil


# ---------------------
# 1. 데이터셋 정의
# ---------------------
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomAffine(degrees=3, translate=(0.1, 0.1)),
            DoodleAugment(),
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