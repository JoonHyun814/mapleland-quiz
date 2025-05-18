import sys
sys.path.append("./")
from dataset import maple_dataset
from torch.utils.data import  DataLoader
import glob
from PIL import Image
import torchvision.transforms as T
import json
from models.maple_models import ResNetClassifier, EfficientNetClassifier
import torch
import tqdm

image_paths = sorted(list(glob.glob("dataset/all_images/*")))

dataset = maple_dataset.ImageClassificationDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

num_classes = 513
model = EfficientNetClassifier(num_classes=num_classes)
model.load_state_dict(torch.load("ckpt/test15.pt",map_location=torch.device('cpu')))
model.eval()

with open("inference/database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)

cnt = 0
s_cnt = 0
for x, label in tqdm.tqdm(dataloader):
    x, label = x, label
    transform = T.ToPILImage()
    output = model(x)  # (B, 513)
    pred = output.argmax(dim=1)
    for l,p in zip(label,pred):
        cnt += 1
        if int(l) != int(p):
            print(f"failed! gt:{database_idx[str(int(l))]}, pred:{database_idx[str(int(p))]}")
        else:
            s_cnt += 1

print(f"accuracy={s_cnt/cnt*100}%")