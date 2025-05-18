import torch
import sys
sys.path.append("./")
from models.maple_models import ResNetClassifier
from torchvision import transforms
from PIL import Image
import glob

# 모델 초기화 및 학습된 가중치 로드
num_classes = 513
model = ResNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("ckpt/test6.pt",map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
])

image_pil = Image.open("dataset/all_images/G팬텀워치.png").convert("RGB")
tensor_img = transform(image_pil)
print(model(tensor_img.unsqueeze(0)).argmax())