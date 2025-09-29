"""
Run with:
    streamlit run app.py
Make sure to install the requirements:
    pip install streamlit webrtcvad pyaudio numpy pillow matplotlib
"""

import numpy as np
import webrtcvad
import pyaudio
from PIL import Image
import streamlit as st
import time
import os

# -----------------------------------------------------
# Settings
# -----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_TALKING = os.path.join(BASE_DIR, "Talking.jpg")
IMG_NOTTALKING = os.path.join(BASE_DIR, "NotTalking.jpg")

RATE = 16000          # 16 kHz sample rate
FRAME_DURATION = 30   # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)
CHANNELS = 1
FORMAT = pyaudio.paInt16

# -----------------------------------------------------
# Streamlit Layout
# -----------------------------------------------------
st.set_page_config(page_title="Live Voice Activity Detection", layout="centered")
st.title("🎙️ Real-Time Speech Detection")

# Image placeholders
img_placeholder = st.empty()
status_placeholder = st.empty()
plot_placeholder = st.empty()

# Load images once
talking_img = Image.open(IMG_TALKING)
not_talking_img = Image.open(IMG_NOTTALKING)

# -----------------------------------------------------
# WebRTC VAD + PyAudio Setup
# -----------------------------------------------------
vad = webrtcvad.Vad(3)
pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 frames_per_buffer=FRAME_SIZE)

st.info("Click the **Stop** button in the top-right of the page to end the demo.")

# -----------------------------------------------------
# Main Loop
# -----------------------------------------------------
while True:
    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
    samples = np.frombuffer(data, dtype=np.int16)
    is_speech = vad.is_speech(data, RATE)

    # Update UI
    img_placeholder.image(talking_img if is_speech else not_talking_img,
                          use_container_width=True)
    status_placeholder.markdown(
        f"<h3 style='text-align:center; color:black;'>"
        f"{'Speech Detected!' if is_speech else 'No Speech Detected'}"
        f"</h3>", unsafe_allow_html=True)

    # Simple amplitude plot (tiny waveform)
    plot_placeholder.line_chart(samples)

    time.sleep(0.05)  # 50 ms update
