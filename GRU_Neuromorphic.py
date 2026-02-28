# ============================================================
# NEUROMORPHIC HAND – CORRECT SEQUENCE TRAINING PIPELINE
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

window_size = 20

# ============================================================
# 2️⃣ CREATE SEQUENCES PER GESTURE (IMPORTANT FIX)
# ============================================================

X_sequences = []
y_sequences = []

for gesture in sorted(df["gesture_id"].unique()):

    gesture_df = df[df["gesture_id"] == gesture]

    X_g = gesture_df[feature_cols].values
    y_g = gesture_df["gesture_id"].values - 1

    for i in range(len(X_g) - window_size):
        X_sequences.append(X_g[i:i+window_size])
        y_sequences.append(y_g[i])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("Total sequences:", X_sequences.shape[0])

# ============================================================
# 3️⃣ TRAIN TEST SPLIT (NOW SAFE)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    stratify=y_sequences
)

# ============================================================
# 4️⃣ SCALE FEATURES CORRECTLY
# ============================================================

num_samples, time_steps, num_features = X_train.shape

X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped  = X_test.reshape(-1, num_features)

scaler = StandardScaler()
X_train_reshaped = scaler.fit_transform(X_train_reshaped)
X_test_reshaped  = scaler.transform(X_test_reshaped)

X_train = X_train_reshaped.reshape(X_train.shape[0], time_steps, num_features)
X_test  = X_test_reshaped.reshape(X_test.shape[0], time_steps, num_features)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# ============================================================
# 5️⃣ GRU MODEL
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
# 6️⃣ TRAINING SETUP
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
# 7️⃣ TRAIN LOOP
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
# 8️⃣ FINAL EVALUATION
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
print("\nClassification Report:\n")
print(classification_report(all_true, all_preds, zero_division=0))

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