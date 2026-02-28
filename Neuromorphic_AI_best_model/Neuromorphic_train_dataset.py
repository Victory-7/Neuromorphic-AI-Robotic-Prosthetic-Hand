# ============================================================
# NEUROMORPHIC HAND – FULL TRAINING PIPELINE
# Random Forest + GRU Sequence Model
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================

df = pd.read_csv("neuromorphic_hand_dataset_7400_samples.csv")

feature_cols = [
    "emg_adc","emg_rms",
    "flex_thumb","flex_index","flex_middle","flex_ring","flex_pinky",
    "fsr_thumb","fsr_index","fsr_middle",
    "spike_rate","membrane_potential"
]

X = df[feature_cols].values
y = df["gesture_id"].values - 1  # convert to 0–36

# ============================================================
# 2️⃣ TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3️⃣ SCALE FEATURES
# ============================================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ============================================================
# 4️⃣ RANDOM FOREST BASELINE
# ============================================================

print("\n================ RANDOM FOREST =================")

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

# ============================================================
# 5️⃣ CREATE SEQUENCES FOR GRU
# ============================================================

def create_sequences(X, y, window_size=25):
    X_seq = []
    y_seq = []

    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])

    return np.array(X_seq), np.array(y_seq)

window_size = 25

X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size)
X_test_seq, y_test_seq   = create_sequences(X_test, y_test, window_size)

X_train = torch.tensor(X_train_seq, dtype=torch.float32)
X_test  = torch.tensor(X_test_seq, dtype=torch.float32)
y_train = torch.tensor(y_train_seq, dtype=torch.long)
y_test  = torch.tensor(y_test_seq, dtype=torch.long)

# ============================================================
# 6️⃣ GRU MODEL
# ============================================================

class GRUGestureModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=37):
        super().__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

# ============================================================
# 7️⃣ TRAINING SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUGestureModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)

# ============================================================
# 8️⃣ TRAIN LOOP
# ============================================================

epochs = 100
patience = 15
best_acc = 0
counter = 0

train_acc_history = []
test_acc_history = []

print("\n================ GRU TRAINING =================")

for epoch in range(epochs):

    model.train()
    correct_train = 0
    total_train = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == yb).sum().item()
        total_train += yb.size(0)

    train_acc = correct_train / total_train

    # ---- TEST ----
    model.eval()
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == yb).sum().item()
            total_test += yb.size(0)

    test_acc = correct_test / total_test

    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train: {train_acc*100:.2f}% "
          f"Test: {test_acc*100:.2f}%")

    # Early stopping
    if test_acc > best_acc:
        best_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), "best_gru_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print(f"\nBest GRU Accuracy: {best_acc*100:.2f}%")

# ============================================================
# 9️⃣ FINAL EVALUATION
# ============================================================

model.load_state_dict(torch.load("best_gru_model.pth"))
model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

final_accuracy = accuracy_score(all_true, all_preds)
print(f"\nFinal GRU Accuracy: {final_accuracy*100:.2f}%")

# Classification report
report = classification_report(all_true, all_preds)
print("\nClassification Report:\n")
print(report)

# Confusion matrix
cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("gru_confusion_matrix.png")
plt.show()

print("\nTraining Complete 🚀")