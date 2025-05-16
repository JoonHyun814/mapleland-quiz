import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import json

# ONNX 모델 세션 로딩
session = ort.InferenceSession("../models/6/model.onnx")

with open("database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)

# 전처리 함수: NumPy 이미지 입력
def preprocess_numpy_image(image_np):
    # NumPy → PIL 변환
    img = Image.fromarray(image_np.astype(np.uint8)).convert("RGB")

    # torchvision 전처리
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # (3, 224, 224)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
    return tensor.numpy().astype(np.float32)

# 예측 함수: NumPy 이미지 입력
def predict_from_numpy(image_np):
    input_tensor = preprocess_numpy_image(image_np)
    outputs = session.run(None, {"input": input_tensor})
    probs = outputs[0][0]  # (num_classes,)
    pred_class = int(np.argmax(probs))
    return pred_class, probs

# 예시 실행
if __name__ == "__main__":
    import cv2
    image_np = cv2.imread("../train/all_images/G팬텀워치.png")  # BGR (OpenCV 기본)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)  # RGB로 변환

    pred_class, class_probs = predict_from_numpy(image_np)

    print(f"예측된 클래스: {pred_class}")
    top5 = np.argsort(class_probs)[::-1][:5]
    print("상위 5개 클래스 확률:")
    for i in top5:
        print(f" - 클래스 {i}: {class_probs[i]:.4f}")
