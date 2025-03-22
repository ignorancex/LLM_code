import cv2
import numpy as np


# Function to calculate noise intensity per channel
def calculate_noise_intensity(current_frame, total_frames):
    fraction_through_video = current_frame / total_frames
    # Example: linear increase in noise intensity
    # You can implement a different function to vary the noise over time.
    intensity_blue = np.rint(fraction_through_video * 10)  # Adjust these as needed
    intensity_green = np.rint(fraction_through_video * 20)
    intensity_red = np.rint(fraction_through_video * 30)
    return intensity_blue, intensity_green, intensity_red


def glitch_effect(frame, current_frame, total_frames):
    # Get the intensity for the current frame
    intensity_blue, intensity_green, intensity_red = calculate_noise_intensity(current_frame, total_frames)
    print(f"Frame {current_frame+1}/{total_frames} - Intensity: B={intensity_blue}, G={intensity_green}, R={intensity_red}")

    # Generate noise for each channel with the respective intensity
    noise_blue = np.random.randint(0, intensity_blue, (frame.shape[0], frame.shape[1]), dtype='uint8') if intensity_blue > 0 else 0
    noise_green = np.random.randint(0, intensity_green, (frame.shape[0], frame.shape[1]), dtype='uint8') if intensity_green > 0 else 0
    noise_red = np.random.randint(0, intensity_red, (frame.shape[0], frame.shape[1]), dtype='uint8') if intensity_red > 0 else 0

    # Channel shifting
    blue, green, red = cv2.split(frame)
    blue_shifted = np.roll(blue, int(intensity_blue), axis=1)
    green_shifted = np.roll(green, -int(intensity_green), axis=1)
    red_shifted = np.roll(red, int(intensity_red), axis=0)

    # Reconstruct the frame with noise added per channel
    noisy_frame = cv2.merge((
        cv2.add(blue_shifted, noise_blue),
        cv2.add(green_shifted, noise_green),
        cv2.add(red_shifted, noise_red)
    ))

    return noisy_frame

def adjusted_sine_helper(x, A, B, C, D):
    """
    A helper function that returns:
      - 0 from x = 0 to x = A
      - A half sine wave from x = A to x = B (0 to 1)
      - 1 from x = B to x = C
      - A half sine wave from x = C to x = D (1 to 0)
      - 0 from x = D onwards
    """
    if x < A or x > D:
        return 0
    elif A <= x < B:
        # Half sine wave from 0 to 1
        return 0.5 * (1 + np.sin(np.pi * ((x - A) / (B - A)) - np.pi / 2))
    elif B <= x <= C:
        return 1
    elif C < x <= D:
        # Half sine wave from 1 to 0
        return 0.5 * (1 + np.sin(np.pi * ((x - C) / (D - C)) + np.pi / 2))


def adjusted_sine_helper(x, alpha: float, T_peak: float, T: float):
    """
    A helper function that returns:
        - 0 from x = 0 to x = T_0 = alpha * T (check alpha in [0, 1])
        - A half sine wave from x = T_0 to x = (T - T_peak) / 2 (0 to 1)
        - 1 from x = (T - T_peak) / 2 to x = (T + T_peak) / 2
        - A half sine wave from x = (T + T_peak) / 2 to x = T - T_0 = (1 - alpha) * T (1 to 0)
        - 0 from x = T - T_0 to x = T
    We must check that alpha is in the range [0, 1] and T_peak < T.
    """
    T_0 = alpha * T
    if x < T_0 or x > T - T_0:
        return 0
    elif T_0 <= x < (T - T_peak) / 2:
        return 0.5 * (1 + np.sin(np.pi * ((x - T_0) / ((T - T_peak) / 2 - T_0)) - np.pi / 2))
    elif (T - T_peak) / 2 <= x <= (T + T_peak) / 2:
        return 1
    elif (T + T_peak) / 2 < x <= T - T_0:
        return 0.5 * (1 + np.sin(np.pi * ((x - (T + T_peak) / 2) / (T - T_0 - (T + T_peak) / 2)) + np.pi / 2))



# Example usage of the function
A = 0
B = 1
C = 2
D = 3
x_values = np.linspace(-1, 4, 500)  # Generates 500 points from -1 to 4
y_values = [adjusted_sine_helper(x, A, B, C, D) for x in x_values]

# Plotting the function to visualize it
import matplotlib.pyplot as plt
plt.plot(x_values, y_values)
plt.title("Adjusted Sine Wave Helper Function")
plt.xlabel("x")
plt.ylabel("Value")
plt.grid(True)
plt.show()


# Dictionary mapping video file extensions to FOURCC codes
fourcc_dict = {
    'mp4': 'mp4v',  # H.264 codec
    'avi': 'XVID',  # XVID MPEG-4 codec
    'mov': 'mp4v',  # H.264 codec for QuickTime
    # Add more mappings as necessary
}

video_path = './out/video/00005-random-video-4xslowdown/1x1-slerp-4xslowdown-compressed.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_fps = cap.get(cv2.CAP_PROP_FPS)

# Start the glitch effect for each channel at different timesteps
start_blue = np.rint(total_frames // 3, total_frames // 2)

# Generate output video path with '-glitched' appended before the file extension
video_name, video_ext = video_path.rsplit('.', 1)
video_ext = video_ext.lower()
output_video_path = f"{video_name}-glitched.{video_ext}"

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*fourcc_dict.get(video_ext, 'mp4v')), video_fps, size)

current_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the glitch effect
    glitched_frame = glitch_effect(frame, current_frame, total_frames)

    # Write the frame into the file
    out.write(glitched_frame)

    current_frame += 1

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

