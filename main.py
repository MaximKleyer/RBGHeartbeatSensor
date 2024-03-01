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

# Global lists to store heart rate values over time
hr_red_times, hr_green_times, hr_blue_times = [], [], []
hr_red_values, hr_green_values, hr_blue_values = [], [], []
hr_quan_red_times, hr_quan_green_times, hr_quan_blue_times = [], [], []
hr_quan_red_values, hr_quan_green_values, hr_quan_blue_values = [], [], []
hr_avg_times, hr_quan_times = [], []
hr_avg_values, hr_quan_values = [], []

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Define regions based on detected face
        forehead_region = (x + int(w * 0.25), y, int(w * 0.5), int(h * 0.25))
        left_cheek_region = (x + int(w * 0.12), y + int(h * 0.5), int(min(w, h) * 0.25), int(min(w, h) * 0.25))
        right_cheek_region = (x + w - int(min(w, h) * 0.25) - int(w * 0.12), y + int(h * 0.5), int(min(w, h) * 0.25), int(min(w, h) * 0.25))

        def analyze_region(region):
            x, y, w, h = region
            region_rgb = frame[y:y + h, x:x + w]
            avg_red = np.mean(region_rgb[:, :, 2])
            avg_green = np.mean(region_rgb[:, :, 1])
            avg_blue = np.mean(region_rgb[:, :, 0])
            return avg_red, avg_green, avg_blue

        # Analyze each region
        avg_red_forehead, avg_green_forehead, avg_blue_forehead = analyze_region(forehead_region)
        avg_red_left_cheek, avg_green_left_cheek, avg_blue_left_cheek = analyze_region(left_cheek_region)
        avg_red_right_cheek, avg_green_right_cheek, avg_blue_right_cheek = analyze_region(right_cheek_region)

        # Calculate the average intensity across all regions
        avg_red_intensity = (avg_red_forehead + avg_red_left_cheek + avg_red_right_cheek) / 3
        avg_green_intensity = (avg_green_forehead + avg_green_left_cheek + avg_green_right_cheek) / 3
        avg_blue_intensity = (avg_blue_forehead + avg_blue_left_cheek + avg_blue_right_cheek) / 3

        # Calculate the average intensity across all regions using quanternion
        quan_red_intensity = math.sqrt(avg_red_forehead**2 + avg_red_left_cheek**2 + avg_red_right_cheek**2)
        quan_green_intensity = math.sqrt(avg_green_forehead**2 + avg_green_left_cheek**2 + avg_green_right_cheek**2)
        quan_blue_intensity = math.sqrt(avg_blue_forehead**2 + avg_blue_left_cheek**2 + avg_blue_right_cheek**2)

        # Draw rectangles for visualization
        cv2.rectangle(frame, forehead_region[:2], (forehead_region[0] + forehead_region[2], forehead_region[1] + forehead_region[3]), (255, 255, 0), 2)
        cv2.rectangle(frame, left_cheek_region[:2], (left_cheek_region[0] + left_cheek_region[2], left_cheek_region[1] + left_cheek_region[3]), (255, 255, 0), 2)
        cv2.rectangle(frame, right_cheek_region[:2], (right_cheek_region[0] + right_cheek_region[2], right_cheek_region[1] + right_cheek_region[3]), (255, 255, 0), 2)

    return avg_red_intensity, avg_green_intensity, avg_blue_intensity, quan_red_intensity, quan_green_intensity, quan_blue_intensity, frame

def calculate_heart_rate(peaks, timestamps):
    if len(peaks) > 1:
        peak_times = np.diff(timestamps[peaks])
        avg_peak_time = np.mean(peak_times) if len(peak_times) else 0
        heart_rate = 60.0 / avg_peak_time if avg_peak_time else 0
        return heart_rate
    else:
        return 0

# Main function to capture video and analyze heart rate
def capture_and_analyze_heart_rate():
    global latest_heart_rate_red, latest_heart_rate_green, latest_heart_rate_blue, latest_heart_rate_quan_red, latest_heart_rate_quan_green, latest_heart_rate_quan_blue
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

        new_row = {
            'Time': current_time,
            'Avg Red Intensity': latest_heart_rate_red,
            'Avg Green Intensity': latest_heart_rate_green,
            'Avg Blue Intensity': latest_heart_rate_blue,
            'Avg Intensity' : (latest_heart_rate_red + latest_heart_rate_green + latest_heart_rate_blue)/3,
            'Quan Red Intensity': latest_heart_rate_quan_red,
            'Quan Green Intensity': latest_heart_rate_quan_green,
            'Quan Blue Intensity': latest_heart_rate_quan_blue,
            'Quan Intensity': (latest_heart_rate_quan_red + latest_heart_rate_quan_green + latest_heart_rate_quan_blue)/3,
            'Red Difference': latest_heart_rate_red - latest_heart_rate_quan_red,
            'Green Difference': latest_heart_rate_green - latest_heart_rate_quan_green,
            'Blue Difference': latest_heart_rate_blue - latest_heart_rate_quan_blue,
            'Avg Difference': (latest_heart_rate_red + latest_heart_rate_green + latest_heart_rate_blue)/3 - (latest_heart_rate_quan_red + latest_heart_rate_quan_green + latest_heart_rate_quan_blue)/3
        }

        # Use loc to append data
        intensity_data.loc[len(intensity_data.index)] = new_row

        # Update heart rate calculation periodically for each channel
        if len(timestamps) % 5 == 0:
            # Safety check for fs calculation
            if len(timestamps) > 1 and timestamps[-1] != timestamps[0]:
                fs = float(len(timestamps)) / (timestamps[-1] - timestamps[0])
                # Ensure fs is a valid value
                if fs > 0:
                    # Process each color channel
                    # Example for the red channel
                    try:
                        filtered_red = butter_bandpass_filter(red_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_red.size > 0:  # Ensure filtered signal is not empty
                            peaks_red, _ = find_peaks(filtered_red, distance=fs * 0.5)
                            latest_heart_rate_red = calculate_heart_rate(peaks_red, np.array(timestamps))

                        filtered_green = butter_bandpass_filter(green_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_green.size > 0:  # Ensure filtered signal is not empty
                            peaks_green, _ = find_peaks(filtered_green, distance=fs * 0.5)
                            latest_heart_rate_green = calculate_heart_rate(peaks_green, np.array(timestamps))

                        filtered_blue = butter_bandpass_filter(blue_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_blue.size > 0:  # Ensure filtered signal is not empty
                            peaks_blue, _ = find_peaks(filtered_blue, distance=fs * 0.5)
                            latest_heart_rate_blue = calculate_heart_rate(peaks_blue, np.array(timestamps))

                        filtered_quan_red = butter_bandpass_filter(quan_red_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_red.size > 0:
                            peaks_quan_red, _ = find_peaks(filtered_quan_red, distance=fs * 0.5)
                            latest_heart_rate_quan_red = calculate_heart_rate(peaks_quan_red, np.array(timestamps))

                        filtered_quan_green = butter_bandpass_filter(quan_green_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_green.size > 0:
                            peaks_quan_green, _ = find_peaks(filtered_quan_green, distance=fs * 0.5)
                            latest_heart_rate_quan_green = calculate_heart_rate(peaks_quan_green, np.array(timestamps))

                        filtered_quan_blue = butter_bandpass_filter(quan_blue_intensities, 0.7, 2.5, fs, order=5)
                        if filtered_quan_blue.size > 0:
                            peaks_quan_blue, _ = find_peaks(filtered_quan_blue, distance=fs * 0.5)
                            latest_heart_rate_quan_blue = calculate_heart_rate(peaks_quan_blue, np.array(timestamps))
                    except Exception as e:
                        print(f"Error processing blue channel: {e}")
                else:
                    print("Invalid sampling frequency (fs)")
            else:
                print("Insufficient data for fs calculation")

            # Append the latest heart rate values to the global lists
            hr_red_times.append(current_time)
            hr_red_values.append(latest_heart_rate_red)

            hr_green_times.append(current_time)
            hr_green_values.append(latest_heart_rate_green)

            hr_blue_times.append(current_time)
            hr_blue_values.append(latest_heart_rate_blue)

            hr_quan_red_times.append(current_time)
            hr_quan_red_values.append(latest_heart_rate_quan_red)

            hr_quan_green_times.append(current_time)
            hr_quan_green_values.append(latest_heart_rate_quan_green)

            hr_quan_blue_times.append(current_time)
            hr_quan_blue_values.append(latest_heart_rate_quan_blue)

            if hr_red_values and hr_green_values and hr_blue_values:  # Ensure there is data to calculate an average
                latest_avg_hr = np.mean([
                    hr_red_values[-1],
                    hr_green_values[-1],
                    hr_blue_values[-1]
                ])
                hr_avg_values.append(latest_avg_hr)
                hr_avg_times.append(current_time)

            if hr_quan_red_values and hr_quan_green_values and hr_quan_blue_values:
                latest_avg_quan_hr = np.mean([
                    hr_quan_red_values[-1],
                    hr_quan_green_values[-1],
                    hr_quan_blue_values[-1]
                ])
                hr_quan_values.append(latest_avg_quan_hr)
                hr_quan_times.append(current_time)

        # Calculate the average heart rate from all three channels
        avg_heart_rate = np.mean([latest_heart_rate_red, latest_heart_rate_green, latest_heart_rate_blue])
        avg_quan_heart_rate = np.mean([latest_heart_rate_quan_red, latest_heart_rate_quan_green, latest_heart_rate_quan_blue])

        # Display the latest heart rate from each channel on every frame
        cv2.putText(frame, f"{latest_heart_rate_red:.2f} BPM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"{latest_heart_rate_green:.2f} BPM", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
        cv2.putText(frame, f"{latest_heart_rate_blue:.2f} BPM", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Avg: {avg_heart_rate:.2f} BPM", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"{latest_heart_rate_quan_red:.2f} BPM", (400, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, f"{latest_heart_rate_quan_green:.2f} BPM", (400, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 2)
        cv2.putText(frame, f"{latest_heart_rate_quan_blue:.2f} BPM", (400, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, f"Quan: {avg_quan_heart_rate:.2f} BPM", (400, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # # Display the graph of the red, blue. and green channel intensity
        # for i in range(1, len(red_intensities)):
        #     cv2.line(frame, (i - 1, 200 - int(red_intensities[i - 1])), (i, 200 - int(red_intensities[i])), (0, 0, 255), 2)
        #     cv2.line(frame, (i - 1, 400 - int(green_intensities[i - 1])), (i, 400 - int(green_intensities[i])), (0, 255, 0), 2)
        #     cv2.line(frame, (i - 1, 600 - int(blue_intensities[i - 1])), (i, 600 - int(blue_intensities[i])), (255, 0, 0), 2)

        # Display the frame
        cv2.imshow('Frame', frame)
        # print(intensity_data)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_text_table():
    if not intensity_data.empty:
        display_text = "Seconds | Avg Red | Avg Green | Avg Blue | Quan Red | Quan Green | Quan Blue | Avg Intensity | Quan Intensity\n"
        display_text += "-" * 100 + "\n"
        for index, row in intensity_data.tail(5).iterrows():
            display_text += (f"{row['Time']:.2f} | {row['Avg Red Intensity']:.2f} | {row['Avg Green Intensity']:.2f} | "
                             f"{row['Avg Blue Intensity']:.2f} | {row['Quan Red Intensity']:.2f} | "
                             f"{row['Quan Green Intensity']:.2f} | {row['Quan Blue Intensity']:.2f} | "
                             f"{row['Avg Intensity']:.2f} | {row['Quan Intensity']:.2f}\n"
                             f"{'-' * 100}\n")
        return display_text
    else:
        return "Waiting for data..."

def save_to_excel(df, file_path):
    # Ensure you have pandas and openpyxl installed
    try:
        df.to_excel(file_path, index=False)
        print(f"Data successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving to Excel: {e}")

def update_combined_plots(frame):
    global hr_red_times, hr_green_times, hr_blue_times, hr_avg_times, hr_quan_times
    global hr_red_values, hr_green_values, hr_blue_values, hr_avg_values, hr_quan_values

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

    ax1.clear()  # Clear previous plots to prevent overlap

    # Plot the average heart rate for the last 15 seconds
    if avg_indices:
        ax1.plot([hr_avg_times[i] for i in avg_indices], [hr_avg_values[i] for i in avg_indices], label='Avg HR', color='black', linestyle='--')

    # Plot the quaternion heart rate for the last 15 seconds
    if quan_indices:
        ax1.plot([hr_quan_times[i] for i in quan_indices], [hr_quan_values[i] for i in quan_indices], label='Quan HR', color='orange')

    # Set the x-axis to only show the last 15 seconds
    ax1.set_xlim([fifteen_seconds_ago, fifteen_seconds_ago + 15])

    ax1.legend(loc='upper left')
    ax1.set_title('Heart Rate Over Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Heart Rate (BPM)')

    # Update table text with the latest averages
    text_str = update_text_table()
    ax2.text(0, 1, text_str, ha='left', va='top', fontsize='small')

# Integration into main execution
def main():
    global ax1, ax2, update_counter
    update_counter = 0  # Initialize update counter

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)

    ani = FuncAnimation(fig, update_combined_plots, interval=1000)  # Update every second

    capture_thread = threading.Thread(target=capture_and_analyze_heart_rate)
    capture_thread.start()

    plt.show()

    capture_thread.join()
    file_path = 'C:/Users/nycdoe/PycharmProjects/RBGHeartRateSensor/intensity_data.xlsx'
    intensity_data.to_excel(file_path, engine='openpyxl')

if __name__ == "__main__":
    main()