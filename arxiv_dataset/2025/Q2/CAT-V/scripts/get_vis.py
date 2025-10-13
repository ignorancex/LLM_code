import cv2
import json
import sys
from tqdm import tqdm

def add_captions_to_video(video_input_path, json_path, video_output_path):
    # Load JSON data
    with open(json_path, 'r') as f:
        captions_data = json.load(f)

    # Open the input video
    cap = cv2.VideoCapture(video_input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    # Helper function: Check if a frame is within a time segment
    def is_frame_in_segment(frame_idx, start, end, fps):
        timestamp = frame_idx / fps
        return start <= timestamp <= end

    # Helper function: Wrap text to fit within the video width
    def wrap_text(text, font, font_scale, thickness, max_width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines

    # Dynamic scaling based on video resolution
    def get_font_scale_and_thickness(width, height):
        base_width = 1280.0  # Reference width for scaling
        scale_factor = width / base_width
        font_scale = max(0.5 * scale_factor, 0.4)  # Reduced default font size
        thickness = max(int(1.5 * scale_factor), 1)  # Slightly thinner font
        return font_scale, thickness

    # Process video frames
    frame_idx = 0
    with tqdm(total=frame_count, desc="Processing video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Find captions for the current frame
            for caption in captions_data:
                start_time = float(caption['segment'][0])
                end_time = float(caption['segment'][1])
                text = caption['model_answer']

                if is_frame_in_segment(frame_idx, start_time, end_time, fps):
                    # Get font scale and thickness dynamically
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale, font_thickness = get_font_scale_and_thickness(width, height)
                    text_color = (255, 255, 255)  # White
                    bg_color = (0, 0, 0, 150)  # Black with alpha for transparency
                    margin = int(10 * (height / 720))  # Adjust margin proportionally
                    max_width = int(width * 0.95)  # Wrap text at 85% of video width

                    # Wrap text into multiple lines
                    lines = wrap_text(text, font, font_scale, font_thickness, max_width)
                    line_height = cv2.getTextSize("Test", font, font_scale, font_thickness)[0][1] + margin

                    # Determine the text box position
                    total_text_height = len(lines) * line_height
                    text_x = (width - max_width) // 2
                    text_y = height - margin - total_text_height

                    # Create a transparent overlay
                    overlay = frame.copy()
                    cv2.rectangle(
                        overlay,
                        (text_x - margin, text_y - margin),
                        (text_x + max_width + margin, text_y + total_text_height + margin),
                        (0, 0, 0),  # Black background
                        -1
                    )

                    # Add the transparent overlay to the frame
                    alpha = 0.6  # Transparency factor for the background
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                    # Draw each line of text
                    for i, line in enumerate(lines):
                        line_y = text_y + (i * line_height) + line_height
                        cv2.putText(
                            frame,
                            line,
                            (text_x, line_y),
                            font,
                            font_scale,
                            text_color,
                            font_thickness,
                            lineType=cv2.LINE_AA
                        )

            out.write(frame)
            frame_idx += 1
            pbar.update(1)

    # Release resources
    cap.release()
    out.release()
    print(f"Captioned video saved to: {video_output_path}")


if __name__ == "__main__":
    # Read arguments
    video_input_path = sys.argv[1]
    json_path = sys.argv[2]
    video_output_path = sys.argv[3]

    add_captions_to_video(video_input_path, json_path, video_output_path)
