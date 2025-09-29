"""
Run with: python vad_app.py
Requires: Python 3.9+ and the packages listed below.

Recommended installation:
    pip install -r requirements.txt

Requirements:
    pyaudio
    numpy
    pydub
    webrtcvad
    soundfile
    pillow
    opencv-python==4.5.5.64
    matplotlib
    tk   # usually comes with Python on Windows/Mac
"""

import os
import tkinter as tk
import threading
import numpy as np
import webrtcvad
import pyaudio
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Paths ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "Projects", "VAD")

# ---- Tkinter Main Window ----
root = tk.Tk()
root.title("Real-Time Speech Detection")
root.geometry("870x600")

def load_image(file_name):
    return Image.open(os.path.join(IMG_DIR, file_name))

# Load and resize images
bg_speech = load_image("Talking.png").resize((870, 600))
bg_no_speech = load_image("NotTalking.png").resize((870, 600))

bg_speech_tk = ImageTk.PhotoImage(bg_speech)
bg_no_speech_tk = ImageTk.PhotoImage(bg_no_speech)

video_label = tk.Label(root, image=bg_no_speech_tk)
video_label.place(x=0, y=0, relwidth=1, relheight=1)
video_label.image = bg_no_speech_tk  # Keep reference

# Graph area
fig, ax = plt.subplots(figsize=(3, 1.2), facecolor="white")
canvas_plot = FigureCanvasTkAgg(fig, master=root)
canvas_plot.get_tk_widget().place(x=280, y=392)

label = tk.Label(root, text="No speech detected",
                 font=("Montserrat", 16), fg="black", bg="white")
label.place(x=350, y=375)

# ---- Audio / VAD Setup ----
vad = webrtcvad.Vad(3)  # 0–3 (3 = most aggressive)

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FRAME_DURATION = 30  # ms
FRAME_SIZE = int(RATE * FRAME_DURATION / 1000)

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=FRAME_SIZE)

frame_counter = 0
frame_update_interval = 4

def update_background(is_speech):
    current_bg = bg_speech_tk if is_speech else bg_no_speech_tk
    video_label.config(image=current_bg)
    video_label.image = current_bg

def update_signal_plot(audio_samples):
    ax.clear()
    ax.plot(audio_samples, color="black", linewidth=1)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xticks([])
    ax.set_yticks([])
    canvas_plot.draw()

def update_ui(is_speech, audio_samples):
    update_background(is_speech)
    label.config(text="Speech Detected!" if is_speech else "No Speech Detected", fg="black")
    update_signal_plot(audio_samples)

def process_audio():
    global frame_counter
    while True:
        data = stream.read(FRAME_SIZE, exception_on_overflow=False)
        audio_samples = np.frombuffer(data, dtype=np.int16)
        is_speech = vad.is_speech(data, RATE)

        frame_counter += 1
        if frame_counter >= frame_update_interval:
            frame_counter = 0
            root.after(0, update_ui, is_speech, audio_samples)

threading.Thread(target=process_audio, daemon=True).start()
update_background(False)
root.mainloop()

