import sys
sys.path.append("./")
from dataset import maple_dataset
from torch.utils.data import  DataLoader
import glob
from PIL import Image
import torchvision.transforms as T
import json

image_paths = sorted(list(glob.glob("dataset/all_images/*")))

dataset = maple_dataset.ImageClassificationDataset(image_paths)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

with open("inference/database_idx.json",encoding="UTF-8") as f:
    database_idx = json.load(f)

for x, label in dataloader:
    x, label = x.cuda(), label.cuda()
    transform = T.ToPILImage()
    for img,most_similar_index in zip(x,label):
        img = transform(img)
        img.save(f"test/test_images/{database_idx[str(int(most_similar_index))]}.png")