import math
import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import pandas as pd

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize a DataFrame to store the values
columns = [
    'Time', 'Avg Red Intensity', 'Avg Green Intensity', 'Avg Blue Intensity',
    'Quan Red Intensity', 'Quan Green Intensity', 'Quan Blue Intensity',
    'Avg Intensity', 'Quan Intensity', 'Red Difference', 'Green Difference', 'Blue Difference', 'Avg Difference']

intensity_data = pd.DataFrame(columns=columns)

# Initialize a global DataFrame to store heart rate data
heart_rate_data = pd.DataFrame(columns=['Timestamp', 'Average Heart Rate', 'Average Quaternion Heart Rate'])

# Initialize lists to store the signal values and timestamps
red_intensities = []
green_intensities = []
blue_intensities = []
quan_red_intensities = []
quan_green_intensities = []
quan_blue_intensities = []
timestamps = []
start_time = time.time()

# Flag to control the capture thread
capture_running = True

# Initialize variable to store the latest heart rate calculated from each channel
latest_heart_rate_red = 0.0
latest_heart_rate_green = 0.0
latest_heart_rate_blue = 0.0
latest_heart_rate_quan_red = 0.0
latest_heart_rate_quan_green = 0.0
latest_heart_rate_quan_blue = 0.0

# Initialize variables to default values
last_hr_red = last_hr_green = last_hr_blue = last_avg_hr = 0
last_hr_quan_red = last_hr_quan_green = last_hr_quan_blue = last_avg_quan_hr = 0

# Global lists to store heart rate values over time
hr_red_times, hr_green_times, hr_blue_times = [], [], []
hr_red_values, hr_green_values, hr_blue_values = [], [], []
hr_quan_red_times, hr_quan_green_times, hr_quan_blue_times = [], [], []
hr_quan_red_values, hr_quan_green_values, hr_quan_blue_values = [], [], []
hr_avg_times, hr_quan_times = [], []
hr_avg_values, hr_quan_values = [], []

current_second_hr_red = []
current_second_hr_green = []
current_second_hr_blue = []
current_second_hr_quan_red = []
current_second_hr_quan_green = []
current_second_hr_quan_blue = []
current_second_timestamps = []

# These should be initialized somewhere earlier in your code
latest_heart_rate_quan_red_values = []
latest_heart_rate_quan_green_values = []
latest_heart_rate_quan_blue_values = []
heart_rate_quan_means = []


# Global lists to store average heart rate values over time
max_hr_avg_times, max_hr_quan_avg_times = [], []
max_hr_avg_values, max_hr_quan_avg_values = [], []

# Alpha values for smoothing
alpha_rgb = 0.14
alpha_hr = 0.1

# Initialize previous intensities to None (or zero if you prefer)
previous_avg_red_intensity = None
previous_avg_green_intensity = None
previous_avg_blue_intensity = None

previous_quan_red_intensity = None
previous_quan_green_intensity = None
previous_quan_blue_intensity = None

previous_heart_rate_red = 0
previous_heart_rate_green = 0
previous_heart_rate_blue = 0

previous_heart_rate_quan_red = 0
previous_heart_rate_quan_green = 0
previous_heart_rate_quan_blue = 0

# Initialize the previous heart rate
previous_heart_rate = 0

# Butterworth Bandpass Filter Functions
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_frame(frame):
    global previous_avg_red_intensity, previous_avg_green_intensity, previous_avg_blue_intensity
    global previous_quan_red_intensity, previous_quan_green_intensity, previous_quan_blue_intensity


    # Initialize variables to default values before processing
    avg_red_intensity = avg_green_intensity = avg_blue_intensity = 0
    quan_red_intensity = quan_green_intensity = quan_blue_intensity = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Define regions based on detected face
        forehead_region = (x + int(w * 0.25), y + int(h * 0.05), int(w * 0.5), int(h * 0.20))
        left_cheek_region = (x + int(w * 0.19), y + int(h * 0.5), int(min(w, h) * 0.20), int(min(w, h) * 0.20))
        right_cheek_region = (x + w - int(min(w, h) * 0.25) - int(w * 0.15), y + int(h * 0.5), int(min(w, h) * 0.20), int(min(w, h) * 0.20))

        def analyze_region(frame, region):
            x, y, w, h = region
            region = frame[y:y + h, x:x + w]
            avg_red = np.mean(region[:, :, 2])
            avg_green = np.mean(region[:, :, 1])
            avg_blue = np.mean(region[:, :, 0])
            return avg_red, avg_green, avg_blue

        # Analyze each region
        avg_red_forehead, avg_green_forehead, avg_blue_forehead = analyze_region(frame, forehead_region)
        avg_red_left_cheek, avg_green_left_cheek, avg_blue_left_cheek = analyze_region(frame, left_cheek_region)
        avg_red_right_cheek, avg_green_right_cheek, avg_blue_right_cheek = analyze_region(frame, right_cheek_region)

        # Calculate the average intensity across all regions
        avg_red_intensity = (avg_red_forehead + avg_red_left_cheek + avg_red_right_cheek) / 3
        avg_green_intensity = (avg_green_forehead + avg_green_left_cheek + avg_green_right_cheek) / 3
        avg_blue_intensity = (avg_blue_forehead + avg_blue_left_cheek + avg_blue_right_cheek) / 3

        # Apply EMA smoothing for each RGB intensity
        if previous_avg_red_intensity is not None:  # Check to ensure the previous value exists
            avg_red_intensity = (alpha_rgb * avg_red_intensity) + ((1 - alpha_rgb) * previous_avg_red_intensity)
        if previous_avg_green_intensity is not None:
            avg_green_intensity = (alpha_rgb * avg_green_intensity) + ((1 - alpha_rgb) * previous_avg_green_intensity)
        if previous_avg_blue_intensity is not None:
            avg_blue_intensity = (alpha_rgb * avg_blue_intensity) + ((1 - alpha_rgb) * previous_avg_blue_intensity)

        # Calculate the average intensity across all regions using quanternion
        quan_red_intensity = (avg_red_forehead ** 2)/255 + (avg_red_left_cheek ** 2)/255 + (avg_red_right_cheek ** 2)/255
        quan_green_intensity = (avg_green_forehead ** 2)/255 + (avg_green_left_cheek ** 2)/255 + (avg_green_right_cheek ** 2)/255
        quan_blue_intensity = (avg_blue_forehead ** 2)/255 + (avg_blue_left_cheek ** 2)/255 + (avg_blue_right_cheek ** 2)/255

        # Apply EMA smoothing for each RGB intensity
        if previous_quan_red_intensity is not None:  # Check to ensure the previous value exists
            quan_red_intensity = (quan_red_intensity ** (1 - alpha_rgb)) + (previous_quan_red_intensity ** (1 - alpha_rgb))
        if previous_quan_green_intensity is not None:
            quan_green_intensity = (quan_green_intensity ** (1 - alpha_rgb)) + (previous_quan_green_intensity ** (1 - alpha_rgb))
        if previous_quan_blue_intensity is not None:
            quan_blue_intensity = (quan_blue_intensity ** (1 - alpha_rgb)) + (previous_quan_blue_intensity ** (1 - alpha_rgb))

        # Update the previous intensities
        previous_avg_red_intensity = avg_red_intensity
        previous_avg_green_intensity = avg_green_intensity
        previous_avg_blue_intensity = avg_blue_intensity

        previous_quan_red_intensity = quan_red_intensity
        previous_quan_green_intensity = quan_green_intensity
        previous_quan_blue_intensity = quan_blue_intensity

        # Draw rectangles for visualization
        cv2.rectangle(frame, forehead_region[:2], (forehead_region[0] + forehead_region[2], forehead_region[1] + forehead_region[3]), (255, 255, 0), 2)
        cv2.rectangle(frame, left_cheek_region[:2], (left_cheek_region[0] + left_cheek_region[2], left_cheek_region[1] + left_cheek_region[3]), (255, 255, 0), 2)
        cv2.rectangle(frame, right_cheek_region[:2], (right_cheek_region[0] + right_cheek_region[2], right_cheek_region[1] + right_cheek_region[3]), (255, 255, 0), 2)

    return avg_red_intensity, avg_green_intensity, avg_blue_intensity, quan_red_intensity, quan_green_intensity, quan_blue_intensity, frame

def calculate_fft(signal):
    """
    Calculate the Discrete Fourier Transform (DFT) of the given signal.

    Parameters:
    signal (np.array): Input time domain signal.

    Returns:
    np.array: Magnitude spectrum of the DFT.
    """
    # N = len(signal)
    # n = np.arange(N)
    # k = n.reshape((N, 1))
    # e = np.exp(-2j * np.pi * k * n / N)
    # dft = np.dot(e, signal)
    # return np.abs(dft)

    fft_result = np.fft.fft(signal)
    return np.abs(fft_result)

def calculate_heart_rate(signal, fs, min_bpm=40, max_bpm=180):
    """
    Calculate the heart rate from a signal by finding the peak in its DFT.

    Parameters:
    signal (np.array): The input signal.
    fs (int): The sampling frequency of the signal.
    min_bpm (int): The minimum expected heart rate, in BPM.
    max_bpm (int): The maximum expected heart rate, in BPM.

    Returns:
    float: The estimated heart rate, in BPM.
    """
    window = np.hanning(len(signal))

    windowed_signal = signal * window

    # Compute the DFT of the windowed signal
    fft = calculate_fft(windowed_signal)

    # Compute the frequency axis
    freqs = np.fft.fftfreq(len(fft), 1/fs)

    # Find the magnitude spectrum
    magnitude = np.abs(fft)

    # Ignore negative frequencies and frequencies outside the expected BPM range
    min_freq, max_freq = min_bpm / 60, max_bpm / 60
    valid_mask = (freqs > min_freq) & (freqs < max_freq)

    # Ensure there are valid frequencies before finding the peak
    if valid_mask.sum() > 0:  # Check if there are any true values in valid_mask
        peak_freq = freqs[valid_mask][np.argmax(magnitude[valid_mask])]
        heart_rate = 60 * peak_freq
    else:
        heart_rate = None  # or set to a default value, or handle this case as needed

    return heart_rate

# Main function to capture video and analyze heart rate
def capture_and_analyze_heart_rate():
    global latest_heart_rate_red, latest_heart_rate_green, latest_heart_rate_blue, latest_heart_rate_quan_red, latest_heart_rate_quan_green, latest_heart_rate_quan_blue
    global last_hr_red, last_hr_green, last_hr_blue, last_avg_hr, last_hr_quan_red, last_hr_quan_green, last_hr_quan_blue, last_avg_quan_hr
    global previous_heart_rate_red, previous_heart_rate_green, previous_heart_rate_blue
    global previous_heart_rate_quan_red, previous_heart_rate_quan_green, previous_heart_rate_quan_blue

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    # Define capture_running properly if not already defined
    global capture_running
    capture_running = True

    global intensity_data

    while capture_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame to get average intensities
        avg_red_intensity, avg_green_intensity, avg_blue_intensity, quan_red_intensity, quan_green_intensity, quan_blue_intensity, frame = process_frame(frame)
        current_time = time.time() - start_time

        # Append intensities and timestamp
        red_intensities.append(avg_red_intensity)
        green_intensities.append(avg_green_intensity)
        blue_intensities.append(avg_blue_intensity)
        quan_red_intensities.append(quan_red_intensity)
        quan_green_intensities.append(quan_green_intensity)
        quan_blue_intensities.append(quan_blue_intensity)
        timestamps.append(current_time)
        current_second_timestamps.append(current_time)

        new_row = {
            'Time': current_time,
            # 'Avg Red Intensity': latest_heart_rate_red,
            # 'Avg Green Intensity': latest_heart_rate_green,
            # 'Avg Blue Intensity': latest_heart_rate_blue,
            'Avg Intensity' : (latest_heart_rate_red + latest_heart_rate_green + latest_heart_rate_blue)/3,
            # 'Quan Red Intensity': latest_heart_rate_quan_red,
            # 'Quan Green Intensity': latest_heart_rate_quan_green,
            # 'Quan Blue Intensity': latest_heart_rate_quan_blue,
            'Quan Intensity': (latest_heart_rate_quan_red + latest_heart_rate_quan_green + latest_heart_rate_quan_blue)/3
            # 'Red Difference': latest_heart_rate_red - latest_heart_rate_quan_red,
            # 'Green Difference': latest_heart_rate_green - latest_heart_rate_quan_green,
            # 'Blue Difference': latest_heart_rate_blue - latest_heart_rate_quan_blue,
            # 'Avg Difference': (latest_heart_rate_red + latest_heart_rate_green + latest_heart_rate_blue)/3 - (latest_heart_rate_quan_red + latest_heart_rate_quan_green + latest_heart_rate_quan_blue)/3
        }

        # Use loc to append data
        intensity_data.loc[len(intensity_data.index)] = new_row

        # Update heart rate calculation periodically for each channel
        if len(timestamps) % 5 == 0:
            # Safety check for fs calculation
            if len(timestamps) > 1 and timestamps[-1] != timestamps[0]:
                fs = float(len(timestamps)) / (timestamps[-1] - timestamps[0])
                # print(f"Sampling frequency (fs): {fs}")
                # Ensure fs is a valid value
                if fs > 0:
                    # Process each color channel
                    try:
                        filtered_red = butter_bandpass_filter(red_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_red.size > 0:  # Ensure filtered signal is not empty
                            # Apply EMA smoothing to the heart rate calculation
                            current_heart_rate_red = calculate_heart_rate(filtered_red, fs)
                            smoothed_heart_rate_red = (current_heart_rate_red ** (1 - alpha_rgb)) + (previous_heart_rate_red ** (1 - alpha_rgb))
                            previous_heart_rate_red = smoothed_heart_rate_red
                            latest_heart_rate_red = smoothed_heart_rate_red

                        filtered_green = butter_bandpass_filter(green_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_green.size > 0:  # Ensure filtered signal is not empty
                            current_heart_rate_green = calculate_heart_rate(filtered_green, fs)
                            smoothed_heart_rate_green = (current_heart_rate_green ** (1 - alpha_rgb)) + (previous_heart_rate_green ** (1 - alpha_rgb))
                            previous_heart_rate_green = smoothed_heart_rate_green
                            latest_heart_rate_green = smoothed_heart_rate_green

                        filtered_blue = butter_bandpass_filter(blue_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_blue.size > 0:  # Ensure filtered signal is not empty
                            current_heart_rate_blue = calculate_heart_rate(filtered_blue, fs)
                            smoothed_heart_rate_blue = (current_heart_rate_blue ** (1 - alpha_rgb)) + (previous_heart_rate_blue ** (1 - alpha_rgb))
                            previous_heart_rate_blue = smoothed_heart_rate_blue
                            latest_heart_rate_blue = smoothed_heart_rate_blue

                        filtered_quan_red = butter_bandpass_filter(quan_red_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_red.size > 0:
                            current_heart_rate_quan_red = calculate_heart_rate(filtered_quan_red, fs)
                            smoothed_heart_rate_quan_red = (current_heart_rate_quan_red ** (1 - alpha_rgb)) + (previous_heart_rate_quan_red ** (1 - alpha_rgb))
                            previous_heart_rate_quan_red = smoothed_heart_rate_quan_red
                            latest_heart_rate_quan_red = smoothed_heart_rate_quan_red
                            latest_heart_rate_quan_red_values.append(latest_heart_rate_quan_red)

                        filtered_quan_green = butter_bandpass_filter(quan_green_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_green.size > 0:
                            current_heart_rate_quan_green = calculate_heart_rate(filtered_quan_green, fs)
                            smoothed_heart_rate_quan_green = (current_heart_rate_quan_green ** (1 - alpha_rgb)) + (previous_heart_rate_quan_green ** (1 - alpha_rgb))
                            previous_heart_rate_quan_green = smoothed_heart_rate_quan_green
                            latest_heart_rate_quan_green = smoothed_heart_rate_quan_green
                            latest_heart_rate_quan_green_values.append(latest_heart_rate_quan_green)

                        filtered_quan_blue = butter_bandpass_filter(quan_blue_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_blue.size > 0:
                            current_heart_rate_quan_blue = calculate_heart_rate(filtered_quan_blue, fs)
                            smoothed_heart_rate_quan_blue = (current_heart_rate_quan_blue ** (1 - alpha_rgb)) + (previous_heart_rate_quan_blue ** (1 - alpha_rgb))
                            previous_heart_rate_quan_blue = smoothed_heart_rate_quan_blue
                            latest_heart_rate_quan_blue = smoothed_heart_rate_quan_blue
                            latest_heart_rate_quan_blue_values.append(latest_heart_rate_quan_blue)

                        hr_quan_times.append(current_time)

                    except Exception as e:
                        print(f"Error processing blue channel: {e}")
                else:
                    print("Invalid sampling frequency (fs)")
            else:
                print("Insufficient data for fs calculation")

            if len(current_second_timestamps) > 1 and current_second_timestamps[-1] - current_second_timestamps[0] >= 1:
                # Assuming you have already appended heart rate values to the lists
                # Find the maximum heart rate value for each channel within the last second
                max_hr_red = max(current_second_hr_red) if current_second_hr_red else None
                print(max_hr_red, current_second_hr_red)
                max_hr_green = max(current_second_hr_green) if current_second_hr_green else None
                print(max_hr_green, current_second_hr_green)
                max_hr_blue = max(current_second_hr_blue) if current_second_hr_blue else None
                print(max_hr_blue, current_second_hr_blue)
                max_hr_quan_red = max(current_second_hr_quan_red) if current_second_hr_quan_red else None
                print(max_hr_quan_red, current_second_hr_quan_red)
                max_hr_quan_green = max(current_second_hr_quan_green) if current_second_hr_quan_green else None
                print(max_hr_quan_green, current_second_hr_quan_green)
                max_hr_quan_blue = max(current_second_hr_quan_blue) if current_second_hr_quan_blue else None
                print(max_hr_quan_blue, current_second_hr_quan_blue)

                # Calculate the average of the maximum heart rates, if applicable
                max_hrs = [hr for hr in [max_hr_red, max_hr_green, max_hr_blue] if hr is not None]
                max_quan_hrs = [hr for hr in [max_hr_quan_red, max_hr_quan_green, max_hr_quan_blue] if hr is not None]
                max_hr_avg = np.mean(max_hrs) if max_hrs else None  # or use np.mean(max_hrs) for average
                max_hr_quan_avg = np.mean(max_quan_hrs) if max_quan_hrs else None

                max_hr_avg_times.append(current_time)
                max_hr_avg_values.append(max_hr_avg)

                max_hr_quan_avg_times.append(current_time)
                max_hr_quan_avg_values.append(max_hr_quan_avg)

                # Output the maximum heart rates
                # print(f"Maximum HR (Red) in the last second: {max_hr_red} BPM") if max_hr_red else print(
                #     "No HR detected for Red in the last second")
                # print(f"Maximum HR (Green) in the last second: {max_hr_green} BPM") if max_hr_green else print(
                #     "No HR detected for Green in the last second")
                # print(f"Maximum HR (Blue) in the last second: {max_hr_blue} BPM") if max_hr_blue else print(
                #     "No HR detected for Blue in the last second")
                # print(f"Maximum Quan HR (Red) in the last second: {max_hr_quan_red} BPM") if max_hr_quan_red else print(
                #     "No HR detected for Red in the last second")
                # print(f"Maximum Quan HR (Green) in the last second: {max_hr_quan_green} BPM") if max_hr_quan_green else print(
                #     "No HR detected for Green in the last second")
                # print(f"Maximum Quan HR (Blue) in the last second: {max_hr_quan_blue} BPM") if max_hr_quan_blue else print(
                #     "No HR detected for Blue in the last second")
                # print(f"Maximum Average HR in the last second: {max_hr_avg} BPM") if max_hr_avg else print(
                #     "No average HR detected in the last second")
                # print(f"Maximum Average Quan HR in the last second: {max_hr_quan_avg} BPM") if max_hr_quan_avg else print(
                #     "No average HR detected in the last second")

            # # Append the latest heart rate values to the global lists
            # hr_red_times.append(current_time)
            # hr_red_values.append(latest_heart_rate_red)
            #
            # hr_green_times.append(current_time)
            # hr_green_values.append(latest_heart_rate_green)
            #
            # hr_blue_times.append(current_time)
            # hr_blue_values.append(latest_heart_rate_blue)
            #
            # hr_quan_red_times.append(current_time)
            # hr_quan_red_values.append(latest_heart_rate_quan_red)
            #
            # hr_quan_green_times.append(current_time)
            # hr_quan_green_values.append(latest_heart_rate_quan_green)
            #
            # hr_quan_blue_times.append(current_time)
            # hr_quan_blue_values.append(latest_heart_rate_quan_blue)
            #
            # if hr_red_values and hr_green_values and hr_blue_values:  # Ensure there is data to calculate an average
            #     latest_avg_hr = np.mean([
            #         hr_red_values[-1],
            #         hr_green_values[-1],
            #         hr_blue_values[-1]
            #     ])
            #     hr_avg_values.append(latest_avg_hr)
            #     hr_avg_times.append(current_time)
            #
            # if hr_quan_red_values and hr_quan_green_values and hr_quan_blue_values:
            #     latest_avg_quan_hr = np.mean([
            #         hr_quan_red_values[-1],
            #         hr_quan_green_values[-1],
            #         hr_quan_blue_values[-1]
            #     ])
            #     hr_quan_values.append(latest_avg_quan_hr)
            #     hr_quan_times.append(current_time)

        # Calculate the average heart rate from all three channels
        avg_heart_rate = np.mean([latest_heart_rate_red, latest_heart_rate_green, latest_heart_rate_blue])
        avg_quan_heart_rate = np.mean([latest_heart_rate_quan_red, latest_heart_rate_quan_green, latest_heart_rate_quan_blue])

        if len(current_second_hr_red) > 0 and len(current_second_hr_green) > 0 and len(current_second_hr_blue) > 0:
            current_avg_heart_rate = (current_second_hr_red[-1] + current_second_hr_green[-1] + current_second_hr_blue[-1]) / 3
        if len(current_second_hr_quan_red) > 0 and len(current_second_hr_quan_green) > 0 and len(current_second_hr_quan_blue) > 0:
            current_avg_quan_heart_rate = (current_second_hr_quan_red[-1] + current_second_hr_quan_green[-1] + current_second_hr_quan_blue[-1]) / 3

        # Display the latest heart rate from each channel on every frame
        if current_second_hr_red and current_second_hr_green and current_second_hr_blue:
            last_hr_red = current_second_hr_red[-1]
            last_hr_green = current_second_hr_green[-1]
            last_hr_blue = current_second_hr_blue[-1]
            last_avg_hr = (current_second_hr_red[-1] + current_second_hr_green[-1] + current_second_hr_blue[-1]) / 3

        if current_second_hr_quan_red and current_second_hr_quan_green and current_second_hr_quan_blue:
            last_hr_quan_red = current_second_hr_quan_red[-1]
            last_hr_quan_green = current_second_hr_quan_green[-1]
            last_hr_quan_blue = current_second_hr_quan_blue[-1]
            last_avg_quan_hr = (current_second_hr_quan_red[-1] + current_second_hr_quan_green[-1] + current_second_hr_quan_blue[-1]) / 3

        # cv2.putText(frame, f"{latest_heart_rate_red:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),2)
        # cv2.putText(frame, f"{latest_heart_rate_green:.2f} BPM", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,(0, 100, 0), 2)
        # cv2.putText(frame, f"{latest_heart_rate_blue:.2f} BPM", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0),2)
        cv2.putText(frame, f"Avg: {avg_heart_rate:.2f} BPM", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255),2)
        # cv2.putText(frame, f"{latest_heart_rate_quan_red:.2f} BPM", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255),2)
        # cv2.putText(frame, f"{latest_heart_rate_quan_green:.2f} BPM", (450, 60), cv2.FONT_HERSHEY_SIMPLEX, .5,(0, 100, 0), 2)
        # cv2.putText(frame, f"{latest_heart_rate_quan_blue:.2f} BPM", (450, 90), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0),2)
        cv2.putText(frame, f"Quan: {avg_quan_heart_rate:.2f} BPM", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255),2)


        # Clear the lists for the next second's data collection
        current_second_hr_red.clear()
        current_second_hr_green.clear()
        current_second_hr_blue.clear()
        current_second_hr_quan_red.clear()
        current_second_hr_quan_green.clear()
        current_second_hr_quan_blue.clear()
        current_second_timestamps.clear()

        # Display the frame
        cv2.imshow('Frame', frame)
        # print(intensity_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_text_table():
    if not intensity_data.empty:
        display_text = "Seconds | Avg Intensity | Quan Intensity\n"
        display_text += "-" * 100 + "\n"
        for index, row in intensity_data.tail(5).iterrows():
            display_text += (f"{row['Time']:.2f} | "
                             f"{row['Avg Intensity']:.2f} | {row['Quan Intensity']:.2f}\n"
                             f"{'-' * 100}\n")
        return display_text
    else:
        return "Waiting for data..."

def update_figure(frame, ax9):
    ax9.clear()  # Clear previous content
    display_text = update_text_table()  # Get the latest text to display
    ax9.text(0.5, 0.5, display_text, transform=ax9.transAxes, fontsize=10, ha='center', va='center')

def save_to_excel(df, file_path):
    # Ensure you have pandas and openpyxl installed
    try:
        df.to_excel(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

def update_combined_plots(frame):
    global latest_heart_rate_quan_red_values, latest_heart_rate_quan_green_values, latest_heart_rate_quan_blue_values
    global current_second_timestamps, current_second_hr_red, current_second_hr_green, current_second_hr_blue, current_second_hr_quan_red, current_second_hr_quan_green, current_second_hr_quan_blue
    global hr_quan_times

    ax1.clear()  # Clear previous plots to prevent overlap
    ax2.clear()
    ax2.axis('off')

    global start_time
    elapsed_time = time.time() - start_time  # This is the current time in your relative timeline

    # Calculate the lower bound for the last 15 seconds
    fifteen_seconds_ago = elapsed_time - 15
    if fifteen_seconds_ago < 0:
        fifteen_seconds_ago = 0  # Ensure it doesn't go negative

    # Filter to get the last 15 seconds of data
    avg_indices = [i for i, t in enumerate(hr_avg_times) if t >= fifteen_seconds_ago]
    quan_indices = [i for i, t in enumerate(hr_quan_times) if t >= fifteen_seconds_ago]

    min_length = min(len(latest_heart_rate_quan_red_values), len(latest_heart_rate_quan_green_values), len(latest_heart_rate_quan_blue_values))
    trimmed_red = latest_heart_rate_quan_red_values[:min_length]
    trimmed_green = latest_heart_rate_quan_green_values[:min_length]
    trimmed_blue = latest_heart_rate_quan_blue_values[:min_length]

    heart_rate_quan_means = np.mean([trimmed_red, trimmed_green, trimmed_blue], axis=0)

    # # Plot the average heart rate for the last 15 seconds
    # if avg_indices:
    #     ax1.plot([hr_avg_times[i] for i in avg_indices], [hr_avg_values[i] for i in avg_indices], label='Avg HR', color='black', linestyle='--')
    #
    # # Plot the quaternion heart rate for the last 15 seconds
    # if quan_indices:
    #     ax1.plot([hr_quan_times[i] for i in quan_indices], [hr_quan_values[i] for i in quan_indices], label='Quan HR', color='orange')

    print("Timestamps length:", len([hr_quan_times[i] for i in quan_indices]))
    print("Heart rate values length:", len([latest_heart_rate_quan_red_values[i] for i in quan_indices]))

    x_values = [hr_quan_times[i] for i in quan_indices]
    red = [latest_heart_rate_quan_red_values[i] for i in quan_indices]
    green = [latest_heart_rate_quan_green_values[i] for i in quan_indices]
    blue = [latest_heart_rate_quan_blue_values[i] for i in quan_indices]
    Quan = [heart_rate_quan_means[i] for i in quan_indices]


    # Ensure x and y lengths match before plotting
    min_length = min(len(x_values), len(red), len(green), len(blue), len(Quan))
    x_values = x_values[:min_length]
    red = red[:min_length]
    green = green[:min_length]
    blue = blue[:min_length]
    Quan = Quan[:min_length]

    # # Configure the plot
    # ax1.set_title('Heart Rate Over Time')
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Heart Rate (BPM)')
    # ax1.legend(loc='upper left')


    # Optionally, set the x-axis to focus on the most recent data
    if hr_quan_times:
        ax1.set_xlim([max(0, hr_quan_times[-1] - 15), hr_quan_times[-1]])  # Adjust as needed
        ax2.set_xlim([max(0, hr_quan_times[-1] - 15), hr_quan_times[-1]])
        ax3.set_xlim([max(0, hr_quan_times[-1] - 15), hr_quan_times[-1]])
        ax4.set_xlim([max(0, hr_quan_times[-1] - 15), hr_quan_times[-1]])

    # Update table text with the latest averages
    # text_str = update_text_table()
    # ax2.text(0, 1, text_str, ha='left', va='top', fontsize='small')

def update_plot_red(frame):
    global red_intensities, green_intensities, blue_intensities, quan_red_intensities, quan_green_intensities, quan_blue_intensities, timestamps
    global latest_heart_rate_quan_red_values, latest_heart_rate_quan_green_values, latest_heart_rate_quan_blue_values
    global current_second_timestamps, current_second_hr_red, current_second_hr_green, current_second_hr_blue, current_second_hr_quan_red, current_second_hr_quan_green, current_second_hr_quan_blue
    global hr_quan_times

    ax1.clear()  # Clear previous plots to prevent overlap
    ax5.clear()

    global start_time
    elapsed_time = time.time() - start_time  # This is the current time in your relative timeline

    # Calculate the lower bound for the last 20 seconds
    fifteen_seconds_ago = elapsed_time - 20
    if fifteen_seconds_ago < 0:
        fifteen_seconds_ago = 0  # Ensure it doesn't go negative

    # Filter to get the last 15 seconds of data
    quan_indices = [i for i, t in enumerate(hr_quan_times) if t >= fifteen_seconds_ago]

    x_values = [hr_quan_times[i] for i in quan_indices]
    red = [latest_heart_rate_quan_red_values[i] for i in quan_indices]

    if len(x_values) > 0 and len(red) > 0:
        ax1.clear()
        ax1.plot(x_values, red, label='Red', color='red')
        ax1.set_title('Red Channel Heart Rate')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Heart Rate (BPM)')

    if len(timestamps) > 0 and len(red_intensities) > 0:
        ax5.clear()
        ax5.plot(timestamps, red_intensities, label='Red', color='red')
        ax5.set_title('Red Channel Intensity')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Intensity')

    # Optionally, set the x-axis to focus on the most recent data
    if current_second_timestamps:
        ax1.set_xlim([max(0, current_second_timestamps[-1] - 15), current_second_timestamps[-1]])  # Adjust as needed

def update_plot_green(frame):
    global red_intensities, green_intensities, blue_intensities, quan_red_intensities, quan_green_intensities, quan_blue_intensities, timestamps
    global latest_heart_rate_quan_red_values, latest_heart_rate_quan_green_values, latest_heart_rate_quan_blue_values
    global current_second_timestamps, current_second_hr_red, current_second_hr_green, current_second_hr_blue, current_second_hr_quan_red, current_second_hr_quan_green, current_second_hr_quan_blue
    global hr_quan_times

    ax2.clear()  # Clear previous plots to prevent overlap
    ax6.clear()

    global start_time
    elapsed_time = time.time() - start_time  # This is the current time in your relative timeline

    # Calculate the lower bound for the last 60 seconds
    fifteen_seconds_ago = elapsed_time - 20
    if fifteen_seconds_ago < 0:
        fifteen_seconds_ago = 0  # Ensure it doesn't go negative

    # Filter to get the last 15 seconds of data
    quan_indices = [i for i, t in enumerate(hr_quan_times) if t >= fifteen_seconds_ago]

    x_values = [hr_quan_times[i] for i in quan_indices]
    green = [latest_heart_rate_quan_green_values[i] for i in quan_indices]

    # Example update function for the Green channel
    if len(x_values) > 0 and len(green) > 0:
        ax2.clear()
        ax2.plot(x_values, green, label='Green', color='green')
        ax2.set_title('Green Channel Heart Rate')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Heart Rate (BPM)')

    if len(timestamps) > 0 and len(green_intensities) > 0:
        ax6.clear()
        ax6.plot(timestamps, green_intensities, label='Green', color='green')
        ax6.set_title('Green Channel Intensity')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Intensity')

    # Optionally, set the x-axis to focus on the most recent data
    if current_second_timestamps:
        ax2.set_xlim([max(0, current_second_timestamps[-1] - 15), current_second_timestamps[-1]])  # Adjust as needed

def update_plot_blue(frame):
    global red_intensities, green_intensities, blue_intensities, quan_red_intensities, quan_green_intensities, quan_blue_intensities, timestamps
    global latest_heart_rate_quan_red_values, latest_heart_rate_quan_green_values, latest_heart_rate_quan_blue_values
    global current_second_timestamps, current_second_hr_red, current_second_hr_green, current_second_hr_blue, current_second_hr_quan_red, current_second_hr_quan_green, current_second_hr_quan_blue
    global hr_quan_times

    ax1.clear()  # Clear previous plots to prevent overlap
    ax2.clear()
    ax2.axis('off')

    global start_time
    elapsed_time = time.time() - start_time  # This is the current time in your relative timeline

    # Calculate the lower bound for the last 20 seconds
    fifteen_seconds_ago = elapsed_time - 20
    if fifteen_seconds_ago < 0:
        fifteen_seconds_ago = 0  # Ensure it doesn't go negative

    # Filter to get the last 15 seconds of data
    quan_indices = [i for i, t in enumerate(hr_quan_times) if t >= fifteen_seconds_ago]

    x_values = [hr_quan_times[i] for i in quan_indices]
    blue = [latest_heart_rate_quan_blue_values[i] for i in quan_indices]

    # Example update function for the Blue channel
    if len(x_values) > 0 and len(blue) > 0:
        ax3.clear()
        ax3.plot(x_values, blue, label='Blue', color='blue')
        ax3.set_title('Blue Channel Heart Rate')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Heart Rate (BPM)')
        ax3.legend()

    if len(timestamps) > 0 and len(blue_intensities) > 0:
        ax7.clear()
        ax7.plot(timestamps, blue_intensities, label='Blue', color='blue')
        ax7.set_title('Blue Channel Intensity')
        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Intensity')

    # Optionally, set the x-axis to focus on the most recent data
    if current_second_timestamps:
        ax3.set_xlim([max(0, current_second_timestamps[-1] - 15), current_second_timestamps[-1]])  # Adjust as needed

def update_plot_quan(frame):
    global red_intensities, green_intensities, blue_intensities, quan_red_intensities, quan_green_intensities, quan_blue_intensities, timestamps
    global latest_heart_rate_quan_red_values, latest_heart_rate_quan_green_values, latest_heart_rate_quan_blue_values
    global current_second_timestamps, current_second_hr_red, current_second_hr_green, current_second_hr_blue, current_second_hr_quan_red, current_second_hr_quan_green, current_second_hr_quan_blue
    global hr_quan_times

    ax1.clear()  # Clear previous plots to prevent overlap
    ax2.clear()
    ax2.axis('off')

    global start_time
    elapsed_time = time.time() - start_time  # This is the current time in your relative timeline

    # Calculate the lower bound for the last 20 seconds
    fifteen_seconds_ago = elapsed_time - 20
    if fifteen_seconds_ago < 0:
        fifteen_seconds_ago = 0  # Ensure it doesn't go negative

    # Filter to get the last 15 seconds of data
    quan_indices = [i for i, t in enumerate(hr_quan_times) if t >= fifteen_seconds_ago]

    min_length = min(len(latest_heart_rate_quan_red_values), len(latest_heart_rate_quan_green_values), len(latest_heart_rate_quan_blue_values))
    trimmed_red = latest_heart_rate_quan_red_values[:min_length]
    trimmed_green = latest_heart_rate_quan_green_values[:min_length]
    trimmed_blue = latest_heart_rate_quan_blue_values[:min_length]

    heart_rate_quan_means = np.mean([trimmed_red, trimmed_green, trimmed_blue], axis=0)

    x_values = [hr_quan_times[i] for i in quan_indices]
    quan = [heart_rate_quan_means[i] for i in quan_indices]

    # Example update function for the Blue channel
    if len(x_values) > 0 and len(quan) > 0:
        ax4.clear()
        ax4.plot(x_values, quan, label='Quan', color='orange')
        ax4.set_title('Quan Channel Heart Rate')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Heart Rate (BPM)')

    # Optionally, set the x-axis to focus on the most recent data
    if current_second_timestamps:
        ax4.set_xlim([max(0, current_second_timestamps[-1] - 15), current_second_timestamps[-1]])  # Adjust as needed

# Integration into main execution
def main():
    global ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, update_counter

    update_counter = 0  # Initialize update counter

    # Initialize figures for each channel
    fig_red, (ax1, ax5) = plt.subplots(2,1)
    fig_green, (ax2, ax6) = plt.subplots(2, 1)
    fig_blue, (ax3, ax7) = plt.subplots(2, 1)
    fig_quan, ax4 = plt.subplots()
    fig, ax9 = plt.subplots()  # Create a figure and a single subplot

    ax9.axis('off')  # Turn off axis

    # Create animations for each figure
    ani_red = FuncAnimation(fig_red, update_plot_red, interval=500)
    ani_green = FuncAnimation(fig_green, update_plot_green, interval=500)
    ani_blue = FuncAnimation(fig_blue, update_plot_blue, interval=500)
    ani_quan = FuncAnimation(fig_quan, update_plot_quan, interval=500)
    ani = FuncAnimation(fig, update_figure, fargs=(ax9,), interval=1000)  # Update every second


    capture_thread = threading.Thread(target=capture_and_analyze_heart_rate)
    capture_thread.start()

    plt.show()

    capture_thread.join()
    file_path = 'C:/Users/maxim/PycharmProjects/RBGHeartBeatSensor/intensity_data.xlsx'
    intensity_data.to_excel(file_path, engine='openpyxl')

if __name__ == "__main__":
    main()