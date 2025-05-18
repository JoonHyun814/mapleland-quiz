from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import numpy as np
from PIL import Image
import random
import cv2
import torchvision.transforms.functional as F

def make_random_background(size=(224, 224)):
    if random.random() < 0.5:
        w, h = size
        # 배경 기본 색상: 랜덤하게 선택
        base_color = np.array([
            random.randint(50, 230),  # R
            random.randint(50, 230),  # G
            random.randint(50, 230)   # B
        ], dtype=np.uint8)

        # 전체 배경 배열 생성 (색상 단일톤)
        bg = np.ones((h, w, 3), dtype=np.uint8) * base_color
    else:
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

    scale = random.uniform(0.3,0.6)
    new_w = int(new_w * scale)
    new_h = int(new_h * scale)
    resized_foreground = foreground.resize((new_w, new_h), resample=Image.BICUBIC)
    
    # 랜덤 위치 계산
    max_x = bg_w - new_w
    max_y = bg_h - new_h
    x_offset = random.randint(int(max_x*2/5), int(max_x*3/5))
    y_offset = random.randint(int(max_y*4/9), int(max_y*5/9))
    
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

class RandomResize:
    def __init__(self, min_size=128, max_size=224):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img):
        new_width = random.randint(self.min_size, self.max_size)
        new_height = random.randint(self.min_size, self.max_size)
        return img.resize((new_width, new_height), Image.BILINEAR)


class DoodleAugment:
    def __call__(self, img_pil):
        img_cv = np.array(img_pil)[:, :, ::-1]  # PIL → BGR
        img_cv = add_doodle(img_cv)
        img_pil = Image.fromarray(img_cv[:, :, ::-1])  # BGR → RGB → PIL
        return img_pil

class RandomResizeAndRandomPad:
    def __init__(self, min_size=256, max_size=512, target_size=512, fill=0):
        self.min_size = min_size
        self.max_size = max_size
        self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size

        # 1. 랜덤 크기 결정
        new_w = random.randint(self.min_size, self.max_size)
        new_h = random.randint(self.min_size, self.max_size)
        img = F.resize(img, (new_h, new_w))

        # 2. 패딩 사이즈 계산
        pad_w_total = self.target_size - new_w
        pad_h_total = self.target_size - new_h

        # 랜덤하게 padding 분배
        pad_left = random.randint(0, pad_w_total)
        pad_top = random.randint(0, pad_h_total)
        pad_right = pad_w_total - pad_left
        pad_bottom = pad_h_total - pad_top

        # 3. 패딩 적용
        img = F.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=self.fill)

        return img

# ---------------------
# 1. 데이터셋 정의
# ---------------------
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            RandomResizeAndRandomPad(min_size=128, max_size=224, target_size=224, fill=random.randint(0, 255)),
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