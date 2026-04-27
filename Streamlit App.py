import streamlit as st
import serial
import time
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

# =========================
# PAGE
# =========================
st.set_page_config(page_title="Neuromorphic Hand", layout="wide")
st.title("🤖 Neuromorphic Hand Control Dashboard")

# =========================
# DEVICE
# =========================
def get_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")

DEVICE = get_device()
st.write(f"Device: {DEVICE}")

# =========================
# MODEL
# =========================
class GRUGestureModel(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=37):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=0.3)

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

@st.cache_resource
def load_model():
    model = GRUGestureModel().to(DEVICE)
    model.load_state_dict(torch.load("best_gru_model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()
st.success("Model loaded")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("neuromorphic_hand_dataset_7400_samples.csv")

    feature_cols = [
        "emg_adc","emg_rms",
        "flex_thumb","flex_index","flex_middle","flex_ring","flex_pinky",
        "fsr_thumb","fsr_index","fsr_middle",
        "spike_rate","membrane_potential"
    ]

    window_size = 20
    X_sequences = []

    for gesture in sorted(df["gesture_id"].unique()):
        g_df = df[df["gesture_id"] == gesture]
        X_g = g_df[feature_cols].values

        for i in range(len(X_g) - window_size):
            X_sequences.append(X_g[i:i+window_size])

    X = np.array(X_sequences)

    # SCALE
    num_samples, time_steps, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)

    scaler = StandardScaler()
    X_reshaped = scaler.fit_transform(X_reshaped)

    X = X_reshaped.reshape(num_samples, time_steps, num_features)

    return X

X = load_data()
st.success(f"Dataset ready: {X.shape}")

# =========================
# GESTURE NAMES
# =========================
gesture_names = [
    "Fist", "Open", "C", "E", "OK", "I", "L", "O", "S", "U",
    "V", "W", "Y", "D", "G", "H", "K", "P", "R", "J",
    "M", "Gesture21", "N", "Gesture23", "Q", "Gesture25",
    "T", "X", "Z", "Neutral", "Thumbs Up", "Index Point",
    "OK Sign", "Peace", "Open Hand", "Fist Alt", "Light Grip"
]

# =========================
# SERIAL (FIXED)
# =========================
if "ser" not in st.session_state:
    try:
        st.session_state.ser = serial.Serial("COM3", 115200, timeout=1)
        time.sleep(2)
        st.session_state.connected = True
    except:
        st.session_state.ser = None
        st.session_state.connected = False

ser = st.session_state.ser

if st.session_state.connected:
    st.success("ESP Connected")
else:
    st.error("Serial not connected")

# =========================
# CONTROLS
# =========================
if "running" not in st.session_state:
    st.session_state.running = False

col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("▶ Start Live"):
    st.session_state.running = True

if col_btn2.button("⏹ Stop"):
    st.session_state.running = False

# =========================
# PLACEHOLDERS
# =========================
col1, col2 = st.columns(2)

text_placeholder = col1.empty()
confidence_placeholder = col1.empty()

graph_col1, graph_col2 = st.columns(2)

emg_chart = graph_col1.empty()
live_chart = graph_col2.empty()

# history persistence
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# LOOP (STREAMLIT SAFE)
# =========================
if st.session_state.running:

    idx = random.randint(0, len(X) - 1)
    sample = X[idx]

    inp = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(inp)

    probs = F.softmax(out, dim=1)
    confidence = float(torch.max(probs).item()) * 100
    pred_class = int(torch.argmax(out, dim=1).item())

    esp_value = pred_class
    if esp_value < 0 or esp_value > 36:
        esp_value = 29

    gesture_name = gesture_names[esp_value]

    # TEXT
    text_placeholder.markdown(f"""
    ### 🎯 Gesture: **{gesture_name}**
    - ESP ID: `{esp_value}`
    - Sample: `{idx}`
    """)

    confidence_placeholder.progress(int(confidence))
    confidence_placeholder.write(f"Confidence: {confidence:.2f}%")

    # EMG GRAPH
    graph_col1.subheader("📈 EMG Waveform (emg_adc)")
    emg_chart.line_chart(sample[:, 0])

    # LIVE GRAPH
    st.session_state.history.append(esp_value)
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

    graph_col2.subheader("📊 Live Gesture Predictions")
    live_chart.line_chart(st.session_state.history)

    # SEND TO ESP
    if ser:
        ser.write((str(esp_value) + "\n").encode())

    time.sleep(1)
    st.rerun()