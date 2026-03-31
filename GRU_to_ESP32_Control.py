# ============================================================
# NEUROMORPHIC HAND – FINAL DEPLOYMENT (DATASET FIXED)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import serial
import time
import joblib
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler

# ============================================================
# ⚙️ SETTINGS
# ============================================================

MODEL_PATH = "best_gru_model.pth"
SCALER_PATH = "scaler.save"

SERIAL_PORT = "COM4"   # 🔁 CHANGE THIS
BAUD_RATE = 115200

WINDOW_SIZE = 20
CONFIDENCE_THRESHOLD = 0.90

# ============================================================
# 📊 FEATURE COLUMNS (MATCHED TO YOUR DATASET)
# ============================================================

FEATURE_COLS = [
    "emg_adc","emg_rms",
    "flex_thumb","flex_index","flex_middle","flex_ring","flex_pinky",
    "fsr_thumb","fsr_index","fsr_middle",
    "spike_rate","membrane_potential"
]

NUM_FEATURES = len(FEATURE_COLS)

# ============================================================
# 🧠 MODEL (SAME AS TRAINING)
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
# 📦 LOAD MODEL + SCALER
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GRUGestureModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

df = pd.read_csv("neuromorphic_hand_dataset_7400_samples.csv")
scaler = StandardScaler()
scaler.fit(df[FEATURE_COLS].values)

print("✅ Model + scaler loaded")

# ============================================================
# 🔌 CONNECT TO ESP32
# ============================================================

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

print("✅ ESP32 Connected")

# ============================================================
# 🧠 BUFFER
# ============================================================

buffer = deque(maxlen=WINDOW_SIZE)
last_sent = -1

# ============================================================
# 📡 REAL SENSOR INPUT (CSV STREAM VERSION FOR TESTING)
# ============================================================

# 🔁 TEMP: simulate real-time from your dataset
df = pd.read_csv("neuromorphic_hand_dataset_7400_samples.csv")
data_stream = df[FEATURE_COLS].values

index = 0

def get_sensor_data():
    global index

    if index >= len(data_stream):
        index = 0  # loop for testing

    row = data_stream[index]
    index += 1

    return row

# ============================================================
# ⚡ REAL-TIME LOOP
# ============================================================

print("\n🚀 Running real-time control...\n")

while True:
    try:
        data = get_sensor_data()
        buffer.append(data)

        if len(buffer) < WINDOW_SIZE:
            continue

        seq = np.array(buffer)

        # SCALE (VERY IMPORTANT)
        seq = scaler.transform(seq)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        gesture = pred.item()
        confidence = confidence.item()

        print(f"Gesture: {gesture} | Confidence: {confidence:.2f}")

        # Send to ESP32
        if confidence > CONFIDENCE_THRESHOLD and gesture != last_sent:
            message = f"G:{gesture}\n"
            ser.write(message.encode())

            print(f"📤 Sent → {message.strip()}")
            last_sent = gesture

        time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Stopped")
        break

ser.close()