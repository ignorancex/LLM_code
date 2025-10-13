import re
import csv
import os

# Define the regex pattern to match the required information from the log lines
# This pattern captures:
# 1. Runtime
# 2. Bytes Transmitted
# 3. Instant Frame Time (ms)
# 4. Instant FPS
# 5. Average Frame Time (ms)
# 6. Average FPS
# 7. Temperature (C)
pattern = re.compile(r'\[\s*(\d+)\].*?MJPG: (\d+)B (\d+)ms \((\d+\.\d+)fps\), AVG: (\d+)ms \((\d+\.\d+)fps\), Temperature: (\d+\.\d+)')

# Define the input and output directories
dir_ = r"C:\\Users\\tnowr\\OneDrive\\Documents\\GitHub\\Safe-Streaming-Time-ESP32-CAM-Paper-Data\\Data Collection\\Framesize_variation\\"
output_dir = dir_ + r'Cleaned_data\\'

# Ensure the output directory exists, create it if it doesn't
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each .txt file in the directory
for filename in os.listdir(dir_):
    if filename.endswith('.txt'):
        input_file = os.path.join(dir_, filename)
        output_file = os.path.join(output_dir, f"cleaned_{filename[0:-4]}.csv")

        # Extract framesize and voltage from filename
        parts = filename.replace('.txt', '').split('_')
        framesize = parts[0] if len(parts) > 0 else 'Unknown'
        voltage = parts[1] if len(parts) > 1 else 'Unknown'

        # Open the input file for reading and output file for writing
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)

            # Write the CSV header to the output file
            writer.writerow([
                'Runtime', 
                'Bytes Transmitted', 
                'Instant Frame Time (ms)', 
                'Instant FPS', 
                'Avg Frame Time (ms)', 
                'Avg FPS', 
                'Temperature (C)',
                'Framesize',
                'Voltage'
            ])

            total_lines = 0  # Count total lines processed
            cleaned_lines = 0  # Count lines matching the pattern
            first_runtime = None  # Store the first runtime to calculate duration
            last_runtime = None  # Store the last runtime to calculate duration

            # Iterate over each line in the input file
            for line in infile:
                total_lines += 1  # Increment the total line count
                match = pattern.search(line)  # Search for the pattern in the line
                if match:
                    cleaned_lines += 1  # Increment the cleaned line count if pattern matches
                    runtime = int(match.group(1))  # Extract runtime from the match
                    
                    # Set the first runtime if not already set
                    if first_runtime is None:
                        first_runtime = runtime
                    last_runtime = runtime  # Update the last runtime
                    
                    # Write the matched groups along with framesize and voltage to the output CSV file
                    writer.writerow(list(match.groups()) + [framesize, voltage])

            # Calculate the duration based on the first and last runtime
            duration = (last_runtime - first_runtime) / 1000 /60 if first_runtime and last_runtime else 0
            
            # Print summary of processing for the current file
            print(f"File: {filename} - Total Lines: {total_lines}, Cleaned Lines: {cleaned_lines}, Removed Lines: {total_lines - cleaned_lines}, Duration: {duration:.2f} Minutes, Framesize: {framesize}, Voltage: {voltage}")

# Final message indicating all files have been processed
print("Processing complete. All files have been cleaned and saved in the 'Cleaned_data' directory.")
