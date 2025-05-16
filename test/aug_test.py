import numpy as np
from PIL import Image, ImageDraw
import random

def make_random_background(size=(224, 224)):
    w, h = size
    bg = np.random.randint(180, 255, (h, w, 3), dtype=np.uint8)  # 밝은 노이즈 배경

    # 랜덤한 줄무늬
    if random.random() < 0.5:
        for i in range(0, h, random.randint(5, 20)):
            bg[i:i+2, :] = np.random.randint(100, 200)

    # 랜덤한 점
    if random.random() < 0.5:
        for _ in range(random.randint(50, 150)):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)
            bg[y, x] = np.random.randint(50, 150, size=3)

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


img = Image.open("../train/all_images/Mr 피에로.png").convert("RGBA")

# 배경 생성 & 합성
random_bg = make_random_background(img.size)
composite_img = composite_foreground_with_background(img, random_bg)

composite_img.save("aug_output.png")