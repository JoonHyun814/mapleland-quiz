import sys
sys.path.append("./")
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from mss import mss
import time
import json
import torch
from torchvision import transforms
from models.maple_models import ResNetClassifier, EfficientNetClassifier

# 모델 초기화 및 학습된 가중치 로드
num_classes = 513
model = EfficientNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("ckpt/test11.pt",map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

with open("inference/database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)

# === 마우스로 영역 선택을 위한 변수 ===
selecting = False
start_point = (-1, -1)
end_point = (-1, -1)
region_selected = False

def mouse_callback(event, x, y, flags, param):
    global selecting, start_point, end_point, region_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        start_point = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and selecting:
        end_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting = False
        end_point = (x, y)
        region_selected = True


# === 모니터 선택 ===
sct = mss()
print("사용 가능한 모니터 목록:")
for i, mon in enumerate(sct.monitors):
    print(f"Monitor {i}: {mon}")
selected_monitor_index = int(input("사용할 모니터 번호를 입력하세요: "))
screen = sct.monitors[selected_monitor_index]

# === 선택한 모니터 캡처해서 배경 이미지로 사용 ===
screenshot = sct.grab(screen)
screen_img = np.array(screenshot)[:, :, :3].copy()

# === 마우스로 영역 선택 ===
cv2.namedWindow("Select Region")
cv2.setMouseCallback("Select Region", mouse_callback)

print("📌 화면에서 마우스로 영역을 드래그하고, 's' 키를 눌러 선택을 완료하세요.")

while True:
    temp = screen_img.copy()
    if selecting or region_selected:
        cv2.rectangle(temp, start_point, end_point, (0, 255, 0), 2)
    cv2.imshow("Select Region", temp)
    key = cv2.waitKey(1)
    if key == ord('s') and region_selected:
        break
cv2.destroyWindow("Select Region")

# === 선택한 영역 정보 계산 ===
left = min(start_point[0], end_point[0]) + screen["left"]
top = min(start_point[1], end_point[1]) + screen["top"]
width = abs(end_point[0] - start_point[0])
height = abs(end_point[1] - start_point[1])
monitor = {"top": top, "left": left, "width": width, "height": height}

print(f"✅ 선택된 영역: {monitor}")

# === 실시간 분류 시작 ===
font = ImageFont.truetype("inference/NanumHumanHeavy.ttf", 20)
while True:
    screenshot = sct.grab(monitor)
    img = np.array(screenshot)[:, :, :3].copy()
    
    pil_img = Image.fromarray(img)
    tensor_img = transform(pil_img)
    
    pred = model(tensor_img.unsqueeze(0))
    pred_class = pred.argmax()

    most_similar_index = int(pred_class)
    
    label = f"{database_idx[str(most_similar_index)]}"

    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text((50, 50), label, font=font, fill=(255, 0, 0))
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    cv2.imshow("Live Classification", img)
    time.sleep(1)
    
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
