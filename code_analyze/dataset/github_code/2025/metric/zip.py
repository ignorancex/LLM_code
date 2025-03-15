import os
import zipfile

def zip_files_in_current_directory(file_list, output_filename):
    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_filename, 'w') as zipf:
        for file in file_list:
            # Check if the file exists and is within the current directory or subdirectories
            if os.path.exists(file) and os.path.isfile(file):
                # Add the file to the zip archive, using the relative path
                zipf.write(file, os.path.relpath(file, start=os.getcwd()))
            else:
                print(f"File {file} not found or is not a valid file.")

# Example usage
if __name__ == "__main__":
    # List of files to be zipped
    files_to_zip = [
        "active_learning/main.py", 
        "d_optimal/main.py",
        "lpme/main.py",
        "npme/main.py",
        "npme/elliptical_sampler.py",
        "uncertainty_quantification/ece.py",
        "uncertainty_quantification/hmc_nn.py",
        "uncertainty_quantification/metropolis.py",
    ]
    # Output zip file name
    output_zip_filename = "cs329h-pset2-submission.zip"

    zip_files_in_current_directory(files_to_zip, output_zip_filename)
    print(f"Files have been zipped into {output_zip_filename}")
