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
from sklearn.utils.class_weight import compute_class_weight

# ==========================================
# 1️⃣ Load Dataset
# ==========================================

df = pd.read_csv("neuromorphic_hand_dataset_7400_samples.csv")

feature_cols = [
    "emg_adc","emg_rms",
    "flex_thumb","flex_index","flex_middle","flex_ring","flex_pinky",
    "fsr_thumb","fsr_index","fsr_middle",
    "spike_rate","membrane_potential","servo_angle"
]

import seaborn as sns
import matplotlib.pyplot as plt

# Sample to avoid huge plot
sample_df = df.sample(500, random_state=42)

sns.pairplot(sample_df, hue="gesture_id")
plt.show()

X = df[feature_cols].values
y = df["gesture_id"].values - 1  # convert to 0–36

# Train-test split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit scaler ONLY on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_test  = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test, dtype=torch.long)

# ==========================================
# 2️⃣ LIF Spiking Layer
# ==========================================

class LIFLayer(nn.Module):
    def __init__(self, size, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.size = size

    def forward(self, input_current):
        batch_size = input_current.size(0)
        membrane = torch.zeros(batch_size, self.size).to(input_current.device)
        spikes = torch.zeros_like(membrane)

        time_steps = 10

        for _ in range(time_steps):
            membrane = self.decay * membrane + input_current
            spike = (membrane >= self.threshold).float()
            membrane = membrane * (1 - spike)
            spikes += spike

        return spikes / time_steps

# ==========================================
# 3️⃣ Hybrid CNN + SNN Model
# ==========================================

class HybridCNNSNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Flatten(),
            nn.Linear(64 * 13, 64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.lif = LIFLayer(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        # spikes = self.lif(features)
        # return self.fc(spikes)

        return self.fc(features)   # 🔥 bypass LIF

# ==========================================
# 4️⃣ Setup
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNSNN().to(device)

# Class Weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy()
)

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

# DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset  = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64)

# ==========================================
# 5️⃣ Training
# ==========================================

epochs = 150
patience = 20
best_acc = 0
counter = 0

train_acc_history = []
test_acc_history = []

for epoch in range(epochs):

    # ---- TRAIN ----
    model.train()
    correct_train = 0
    total_train = 0
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
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
          f"Loss: {total_loss:.4f} "
          f"Train Acc: {train_acc*100:.2f}% "
          f"Test Acc: {test_acc*100:.2f}%")

    # ---- Early Stopping ----
    if test_acc > best_acc:
        best_acc = test_acc
        counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

print("\nTraining Complete 🚀")

# ==========================================
# 6️⃣ Epoch vs Accuracy Graph
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(train_acc_history, label="Train Accuracy")
plt.plot(test_acc_history, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Epoch vs Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("epoch_vs_accuracy.png")
plt.show()

# ==========================================
# 7️⃣ Final Evaluation
# ==========================================

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

# Accuracy
final_accuracy = accuracy_score(all_true, all_preds)
print(f"\nFinal Accuracy: {final_accuracy*100:.2f}%")

# Classification Report
report_dict = classification_report(
    all_true,
    all_preds,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()
report_df.to_excel("classification_report.xlsx", index=True)
print("Classification report saved.")

# Confusion Matrix
cm = confusion_matrix(all_true, all_preds)
cm_df = pd.DataFrame(cm)
cm_df.to_excel("confusion_matrix.xlsx", index=False)

plt.figure(figsize=(10,8))
sns.heatmap(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Macro Metrics
macro_precision = report_dict["macro avg"]["precision"]
macro_recall    = report_dict["macro avg"]["recall"]
macro_f1        = report_dict["macro avg"]["f1-score"]

print("\n📌 Macro Metrics:")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall:    {macro_recall:.4f}")
print(f"F1 Score:  {macro_f1:.4f}")