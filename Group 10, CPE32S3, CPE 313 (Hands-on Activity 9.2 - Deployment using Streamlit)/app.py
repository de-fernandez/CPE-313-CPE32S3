import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# CNN-BiGRU Model
class CNN_BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.conv = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.bigru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, h = self.bigru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        h = self.dropout(h)
        return self.fc(h)

NUM_FEATURES = 51
NUM_CLASSES  = 2
HIDDEN_SIZE  = 64
CLASS_LABELS = ['Attack', 'Normal']

# Load Model
@st.cache_resource
def load_model():
    model = CNN_BiGRU(
        input_size=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load('cnn_bigru_state_dict.pth', map_location='cpu'))
    model.eval()
    return model

# Streamlit UI
st.set_page_config(page_title="CNN-BiGRU Network Intrusion Detection", layout="wide")
st.title("CNN-BiGRU Intrusion Detection System")
st.write("Upload a CSV file with traffic to classify as **Attack** or **Normal**.")

try:
    model = load_model()
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file 'cnn_bigru_state_dict.pth' not found. Please make sure it's in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
else:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # --- Check feature count ---
    if df.shape[1] != NUM_FEATURES:
        st.error(f"Expected {NUM_FEATURES} features, but got {df.shape[1]}. Please check your CSV.")
        st.stop()

    try:
        # --- Preprocess ---
        data = df.values.astype(np.float32)
        data_tensor = torch.tensor(data).unsqueeze(1)

        # --- Predict ---
        with torch.no_grad():
            outputs = model(data_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()
            predictions = np.argmax(probabilities, axis=1)

        # --- results ---
        predicted_labels = [CLASS_LABELS[p] for p in predictions]
        attack_probs = probabilities[:, 0] * 100
        normal_probs = probabilities[:, 1] * 100

        results_df = df.copy()
        results_df['Prediction']       = predicted_labels
        results_df['Attack Prob (%)']  = attack_probs.round(2)
        results_df['Normal Prob (%)']  = normal_probs.round(2)

        st.write("### Prediction Results")
        st.dataframe(results_df[['Prediction', 'Attack Prob (%)', 'Normal Prob (%)']])

        attack_count = predicted_labels.count('Attack')
        normal_count = predicted_labels.count('Normal')

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(predicted_labels))
        col2.metric("Attack", attack_count)
        col3.metric("Normal", normal_count)

        if attack_count > 0:
            st.warning(f"{attack_count} attack(s) detected in the uploaded data!")
        else:
            st.success("No attacks detected. All traffic appears normal.")

    except Exception as e:
        st.error(f"Prediction error: {e}")