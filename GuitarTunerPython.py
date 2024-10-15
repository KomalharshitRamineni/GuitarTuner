import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time
import threading
import tkinter as tk
from tkinter import ttk


# Reference: Guitar Tuner Algorithm Explanation
# Source: https://www.chciken.com/digital/signal/processing/2020/05/13/guitar-tuner.html


# General settings that can be changed by the user
SAMPLE_FREQ = 48000  # sample frequency in Hz
WINDOW_SIZE = 48000   # window size of the DFT in samples
WINDOW_STEP = 12000   # step size of window
NUM_HPS = 5           # max number of harmonic product spectrums
POWER_THRESH = 1e-5   # tuning is activated if the signal power exceeds this threshold
REFRENCE_NOTE = 440   # defining A4
WHITE_NOISE_THRESH = 0.4  # Everything under WHITE_NOISE_THRESH*avg_energy_per_freq is cut off, Increase and decrease as required

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  # length of the window in seconds
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ        # length between two samples in seconds
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE  # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

def find_closest_note(pitch):
    """
    This function finds the closest note for a given pitch
    Parameters:
        pitch (float): pitch given in hertz
    Returns:
        closest_note (str): e.g. A4, G#3, ..
        closest_pitch (float): pitch of the closest note in hertz
    """
    if pitch == 0:
        return "...", 0.0
    i = int(np.round(np.log2(pitch / REFRENCE_NOTE) * 12))
    closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
    closest_pitch = REFRENCE_NOTE * 2 ** (i / 12)
    return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)

class GuitarTunerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Guitar Tuner")
        self.root.geometry("400x400")
        self.create_widgets()
        self.current_note = "..."
        self.current_freq = 0.0
        self.target_freq = REFRENCE_NOTE

        self.current_angle = 0  # Initial angle of the needle (degrees)
        self.target_angle = 0   # The angle we want to reach
        self.needle_speed = 1   # Speed of the needle movement (degrees per update)

        self.running = True
        self.update_gui()

    def create_widgets(self):
        # Note Label
        self.note_label = ttk.Label(self.root, text="Closest Note: ...", font=("Helvetica", 16))
        self.note_label.pack(pady=10)

        # Frequency Label
        self.freq_label = ttk.Label(self.root, text="Frequency: 0.0 Hz / 440.0 Hz", font=("Helvetica", 12))
        self.freq_label.pack(pady=5)

        # Canvas for Needle
        self.canvas_width = 400
        self.canvas_height = 200
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)

        # Draw Gauge
        self.draw_gauge()

    def draw_gauge(self):


        # Draw colored zones
        self.canvas.create_arc(10, 10, self.canvas_width-10, self.canvas_height*2-10, start=70, extent=15, style='pieslice', outline='yellow', fill='yellow', width=2)

        self.canvas.create_arc(10, 10, self.canvas_width-10, self.canvas_height*2-10, start=95, extent=15, style='pieslice', outline='yellow', fill='yellow', width=2)

        self.canvas.create_arc(10, 10, self.canvas_width-10, self.canvas_height*2-10, start=85, extent=10, style='pieslice', outline='green', fill='green', width=2)


        # Draw semicircle
        self.canvas.create_arc(10, 10, self.canvas_width-10, self.canvas_height*2-10, start=0, extent=180, style='pieslice', width=2)



        # Needle initialization
        self.needle = self.canvas.create_line(self.canvas_width//2, self.canvas_height, self.canvas_width//2, 20, fill='gray', width=2)

    def update_needle(self, deviation_ratio):
        # Remove existing needle
        self.canvas.delete(self.needle)
        angle = deviation_ratio
        # The angle is in refrence to east increasing anti-clockwise
            
        radius = self.canvas_width//2 - 20


        # Calculation of where the line should be drawn based on the angle

        if angle > 90:

            angle_rad = np.radians(180 - (180-angle))

            height = radius * np.sin(angle_rad)
            width = radius * np.cos(angle_rad)

            end_x = self.canvas_width//2  - width
            end_y = self.canvas_height - height
        else:

            angle_rad = np.radians(180 - angle) 

            height = radius * np.sin(angle_rad)
            width = radius * np.cos(angle_rad)

            end_x = self.canvas_width//2  + width
            end_y = self.canvas_height - height

        self.needle = self.canvas.create_line(self.canvas_width//2, self.canvas_height, end_x, end_y, fill='gray', width=2)



    def update_needle_position(self):
        # Smoothly move the needle towards the target angle
        if abs(self.current_angle - self.target_angle) > self.needle_speed:
            if self.current_angle < self.target_angle:
                self.current_angle += self.needle_speed
            else:
                self.current_angle -= self.needle_speed

            # Calculate the new needle position
            self.update_needle(self.current_angle)

            # Continue updating after 30 ms for smooth animation
            self.root.after(30, self.update_needle_position)
        else:
            # Snap to the exact target angle if close enough
            self.current_angle = self.target_angle
            self.update_needle(self.current_angle)


    def set_needle_target(self,angle):
        self.target_angle = angle
        self.update_needle_position()


    def update_gui(self):
        # Update Labels
        self.note_label.config(text=f"Note: {self.current_note}")
        self.freq_label.config(text=f"Frequency: {self.current_freq} Hz / {self.target_freq} Hz")

        if self.target_freq != 0:

            semitoneNext = int(np.round(np.log2(self.target_freq/REFRENCE_NOTE)*12)) + 1
            semitonePrevious = int(np.round(np.log2(self.target_freq/REFRENCE_NOTE)*12)) - 1

            nextNoteFreq = REFRENCE_NOTE * 2 ** (semitoneNext / 12)
            previousNoteFreq = REFRENCE_NOTE * 2 ** (semitonePrevious / 12)
            
            
            frequencyRange = abs(previousNoteFreq-nextNoteFreq)


            hertzAngleRatio = 180/frequencyRange
            angleupdate = hertzAngleRatio * (previousNoteFreq-self.current_freq)
            angleupdate = abs(angleupdate)

            if self.current_freq==0:
                self.set_needle_target(0)
                
            elif self.current_note == "...":
                self.set_needle_target(0)

            else:
                self.set_needle_target(angleupdate)


        # Schedule next update
        if self.running:
            self.root.after(100, self.update_gui)

    def set_note(self, note, freq, target_freq):
        self.current_note = note
        self.current_freq = freq
        self.target_freq = target_freq

    def stop(self):
        self.running = False

def audio_callback(indata, frames, time_info, status, tuner_gui):
    """
    Callback function for processing audio input.
    """
    # Define static variables
    if not hasattr(audio_callback, "window_samples"):
        audio_callback.window_samples = np.zeros(WINDOW_SIZE)
    if not hasattr(audio_callback, "noteBuffer"):
        audio_callback.noteBuffer = ["1", "2"]

    if status:
        print(status)
        return

    if any(indata):
        # Append new samples and remove old ones
        audio_callback.window_samples = np.concatenate((audio_callback.window_samples, indata[:, 0]))
        audio_callback.window_samples = audio_callback.window_samples[len(indata[:, 0]):]

        # Calculate signal power
        signal_power = (np.linalg.norm(audio_callback.window_samples, ord=2)**2) / len(audio_callback.window_samples)
        if signal_power < POWER_THRESH:
            # Clear console output (optional)
            os.system('cls' if os.name == 'nt' else 'clear')
            # Update GUI with no note detected
            tuner_gui.set_note("...", 0.0, REFRENCE_NOTE)
            return

        # Apply Hann window to reduce spectral leakage
        hann_samples = audio_callback.window_samples * HANN_WINDOW
        # Perform FFT and get magnitude spectrum
        magnitude_spec = np.abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

        # Suppress mains hum below 62Hz
        for i in range(int(62 / DELTA_FREQ)):
            magnitude_spec[i] = 0

        # Apply octave band filtering to suppress noise
        for j in range(len(OCTAVE_BANDS) - 1):
            ind_start = int(OCTAVE_BANDS[j] / DELTA_FREQ)
            ind_end = int(OCTAVE_BANDS[j + 1] / DELTA_FREQ)
            ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
            avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end - ind_start)
            avg_energy_per_freq = avg_energy_per_freq ** 0.5
            for i in range(ind_start, ind_end):
                if magnitude_spec[i] < WHITE_NOISE_THRESH * avg_energy_per_freq:
                    magnitude_spec[i] = 0

        # Interpolate spectrum for HPS
        mag_spec_ipol = np.interp(
            np.arange(0, len(magnitude_spec), 1 / NUM_HPS),
            np.arange(0, len(magnitude_spec)),
            magnitude_spec
        )
        mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2)  # Normalize

        # Copy for HPS calculation
        hps_spec = copy.deepcopy(mag_spec_ipol)

        # Calculate HPS
        for i in range(NUM_HPS):
            tmp_hps_spec = np.multiply(
                hps_spec[:int(np.ceil(len(mag_spec_ipol) / (i + 1)))],
                mag_spec_ipol[::(i + 1)]
            )
            if not any(tmp_hps_spec):
                break
            hps_spec = tmp_hps_spec

        # Find the peak frequency in HPS spectrum
        if len(hps_spec) == 0:
            max_freq = 0.0
        else:
            max_ind = np.argmax(hps_spec)
            max_freq = max_ind * (SAMPLE_FREQ / WINDOW_SIZE) / NUM_HPS

        # Find the closest musical note
        closest_note, closest_pitch = find_closest_note(max_freq)
        max_freq = round(max_freq, 1)
        closest_pitch = round(closest_pitch, 1)

        # Update note buffer for stability
        audio_callback.noteBuffer.insert(0, closest_note)
        audio_callback.noteBuffer.pop()

        # Check if the detected note is stable
        if audio_callback.noteBuffer.count(audio_callback.noteBuffer[0]) == len(audio_callback.noteBuffer):
            tuner_gui.set_note(closest_note, max_freq, closest_pitch)
        else:
            tuner_gui.set_note("...", 0.0, REFRENCE_NOTE)
    else:
        tuner_gui.set_note("...", 0.0, REFRENCE_NOTE)

def start_audio_stream(tuner_gui):
    """
    Starts the audio input stream and processes audio in real-time.
    """
    try:
        print("Starting HPS guitar tuner...")
        with sd.InputStream(
            channels=1,
            callback=lambda indata, frames, time_info, status: audio_callback(indata, frames, time_info, status, tuner_gui),
            blocksize=WINDOW_STEP,
            samplerate=SAMPLE_FREQ
        ):
            while tuner_gui.running:
                time.sleep(0.1)
    except Exception as exc:
        print(str(exc))

def main():
    # Initialize Tkinter
    root = tk.Tk()
    tuner_gui = GuitarTunerGUI(root)

    # Start audio processing in a separate thread
    audio_thread = threading.Thread(target=start_audio_stream, args=(tuner_gui,), daemon=True)
    audio_thread.start()

    # Handle window close
    def on_closing():
        tuner_gui.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Start Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
