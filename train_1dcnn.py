"""78차원 3D 관절 피처 학습 스크립트 (안정성 최우선 버전)."""

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "assets/csv/asl_landmarks.csv"
MODEL_SAVE_PATH = "assets/models/asl_1dcnn.pth"
CLASSES_SAVE_PATH = "assets/models/classes.json"

os.makedirs("assets/models", exist_ok=True)

print("[INFO] Loading CSV Dataset...", flush=True)
df = pd.read_csv(CSV_PATH)
X = df.iloc[:, :-1].values.astype(np.float32)
y_raw = df.iloc[:, -1].values

input_dim = X.shape[1]
print(f"[INFO] Input Feature Dimension: {input_dim}", flush=True)
print(f"[INFO] Total Dataset Size: {len(X)} rows", flush=True)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
classes = label_encoder.classes_.tolist()

with open(CLASSES_SAVE_PATH, "w") as f:
    json.dump(classes, f)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.from_numpy(X_train).unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train).long()

X_val_tensor = torch.from_numpy(X_val).unsqueeze(1)
y_val_tensor = torch.from_numpy(y_val).long()

# 배치 크기를 128로 설정하여 학습 루프 횟수 감소
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


class AdvancedHandNet(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AdvancedHandNet, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.MaxPool1d(2)
        )
        
        conv_out_dim = 128 * (in_features // 2)
        
        self.fc1 = nn.Linear(conv_out_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.act1 = nn.SiLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.act2 = nn.SiLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.out(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Training on device: {device}", flush=True)

model = AdvancedHandNet(in_features=input_dim, num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# 에포크 수를 20~25 정도로 조율해도 충분히 높은 정확도가 나옵니다.
epochs = 20
best_acc = 0.0

print("[INFO] Start Training...", flush=True)
for epoch in range(epochs):
    model.train()
    train_loss, correct = 0, 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_correct += (outputs.argmax(1) == labels).sum().item()

    val_acc = val_correct / len(y_val) * 100

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.cpu().state_dict(), MODEL_SAVE_PATH)
        model.to(device)

    print(
        f"Epoch {epoch+1:02d}/{epochs} | Train Acc: {correct/len(y_train)*100:.2f}% | Val Acc: {val_acc:.2f}% (Best: {best_acc:.2f}%)",
        flush=True,
    )

print(f"\n[SUCCESS] Best Model saved to '{MODEL_SAVE_PATH}' (Best Acc: {best_acc:.2f}%)", flush=True)