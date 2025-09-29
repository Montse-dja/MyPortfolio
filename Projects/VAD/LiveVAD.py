"""
Run with: python vad_app.py
Requires: Python 3.9+ and the packages listed below.

Recommended installation:
    pip install -r requirements.txt

Requirements:
    pyaudio
    numpy
    pydub

    soundfile
    pillow
    opencv-python==4.5.5.64
    matplotlib
    tk   # usually comes with Python on Windows/Mac
"""
"""

import os
import av
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# -------------------------------------------------------
# Page Setup
# -------------------------------------------------------
st.set_page_config(page_title="Live Voice Activity Detection", layout="centered")
st.title("🎙️ Real-Time Speech Detection (Browser)")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
talking_img = Image.open(os.path.join(BASE_DIR, "Talking.jpg"))
not_talking_img = Image.open(os.path.join(BASE_DIR, "NotTalking.jpg"))

img_placeholder = st.empty()
text_placeholder = st.empty()

# -------------------------------------------------------
# Audio Processor
# -------------------------------------------------------
class VADProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.energy = 0.0
        self.is_speech = False
        self.threshold = 0.015  # tweak this if too sensitive/insensitive

    def recv_audio_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono float32 numpy array
        audio = frame.to_ndarray().astype(np.float32).mean(axis=0)
        # Compute root-mean-square energy
        rms = np.sqrt(np.mean(np.square(audio)))
        self.energy = rms
        self.is_speech = rms > self.threshold
        return frame  # pass through unchanged

# -------------------------------------------------------
# WebRTC Streamer
# -------------------------------------------------------
webrtc_ctx = webrtc_streamer(
    key="vad-demo",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,  # small buffer for low latency
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    audio_processor_factory=VADProcessor,
)

# -------------------------------------------------------
# UI Update Loop
# -------------------------------------------------------
if webrtc_ctx and webrtc_ctx.audio_processor:
    processor: VADProcessor = webrtc_ctx.audio_processor
    st.info("Give microphone permission when asked.")
    while True:
        if processor.is_speech:
            img_placeholder.image(talking_img, use_container_width=True)
            text_placeholder.markdown(
                "<h3 style='text-align:center;color:black;'>Speech Detected!</h3>",
                unsafe_allow_html=True,
            )
        else:
            img_placeholder.image(not_talking_img, use_container_width=True)
            text_placeholder.markdown(
                "<h3 style='text-align:center;color:black;'>No Speech Detected</h3>",
                unsafe_allow_html=True,
            )
        st.sleep(0.1)

