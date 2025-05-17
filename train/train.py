import sys
sys.path.append("./")
from dataset import maple_dataset
from models import maple_models
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import glob
from torch.optim.lr_scheduler import StepLR
import tqdm


# ---------------------
# 학습 루프
# ---------------------
def train(model, dataloader, optimizer, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for x, label in tqdm.tqdm(dataloader):
            x, label = x.cuda(), label.cuda()
            output = model(x)  # (B, 513)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%, Best={best_acc:.2f}%")
        
        # ✅ 최고 정확도일 경우 저장
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"ckpt/test11.pt")
            print(f"✔️ Best model saved at epoch {epoch+1} (Acc: {acc:.2f}%)")


if __name__ == "__main__":
    image_paths = sorted(list(glob.glob("dataset/all_images/*")))

    dataset = maple_dataset.ImageClassificationDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = maple_models.EfficientNetClassifier(num_classes=len(image_paths)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.8)

    train(model, dataloader, optimizer, epochs=100)