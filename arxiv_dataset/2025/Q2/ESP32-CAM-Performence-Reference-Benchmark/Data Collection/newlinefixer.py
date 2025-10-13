import os

# Define the file paths
input_file = r"C:\\Users\\tnowr\\OneDrive\\Documents\\GitHub\\Safe-Streaming-Time-ESP32-CAM-Paper-Data\\Data Collection\\Power_ver\\320x240_3v0_proper.txt"
output_file = r"C:\\Users\\tnowr\\OneDrive\\Documents\\GitHub\\Safe-Streaming-Time-ESP32-CAM-Paper-Data\\Data Collection\\Cleaned_data\\cleaned_320x240_3v0_proper.txt"

# Ensure the output directory exists
output_dir = os.path.dirname(output_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the file, replace '\r\n' with '\n', and save the cleaned content
with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile:
    content = infile.read()

cleaned_content = content.replace('\r\n', '\n')

# Write the cleaned content, ensuring newlines are correctly handled
with open(output_file, 'w', encoding='utf-8', newline='\n') as outfile:
    outfile.write(cleaned_content)

print(f"File: {os.path.basename(input_file)} has been cleaned and saved as {os.path.basename(output_file)}")
