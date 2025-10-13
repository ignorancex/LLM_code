import os
import numpy as np

def merge_npy_files(folder1, folder2, folder3, folder4, folder5, output_folder):
    # Make sure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get the list of .npy files in the input folders
    files1 = sorted([f for f in os.listdir(folder1) if f.endswith('.npy')])
    files2 = sorted([f for f in os.listdir(folder2) if f.endswith('.npy')])
    files3 = sorted([f for f in os.listdir(folder3) if f.endswith('.npy')])
    files4 = sorted([f for f in os.listdir(folder4) if f.endswith('.npy')])
    files5 = sorted([f for f in os.listdir(folder5) if f.endswith('.npy')])

    # Make sure all folders contain the same number of .npy files
    assert len(files1) == len(files2) == len(files3) == len(files4) == len(files5), "All folders must contain the same number of .npy files"
    print(len(files1))
    
    for file1, file2, file3, file4, file5 in zip(files1, files2, files3, files4, files5):
        # Set file path
        file_path1 = os.path.join(folder1, file1)
        file_path2 = os.path.join(folder2, file2)
        file_path3 = os.path.join(folder3, file3)
        file_path4 = os.path.join(folder4, file4)
        file_path5 = os.path.join(folder5, file5)
        
        try:
            # Load data and preprocess data
            data1 = np.load(file_path1)  # P_iso
            data2 = np.load(file_path2)  # Building
            data3 = np.load(file_path3)  # Sparse Tput
            data3 = data3 / 10           # Preprocess Sparse_TP data
            data4 = np.load(file_path4)  # Sparse RI
            data4 = data4 * 10           # Preprocess Sparse RI data
            data5 = np.load(file_path5)  # Sparse CQI
            data5 = data5 * 10           # Preprocess Sparse CQI data  
            

            # Merge the data along the last axis
            merged_data = np.stack((data1, data2, data3, data4, data5), axis=-1)
            
            # Save the merged data to a new .npy file
            output_file_path = os.path.join(output_folder, file1)
            np.save(output_file_path, merged_data)
            print(f"Processed and saved {file1} as {output_file_path}")

        except Exception as e:
            print(f"Error processing {file1}, {file2}, {file3}, {file4}, {file5}: {e}")
            continue

# Change the folder names to match your data
folder1 = 'Data_all/PGIso'
folder2 = 'Data_all/Building'
folder3 = 'Data_all/Sparse_TP'
folder4 = 'Data_all/Sparse_RI'
folder5 = 'Data_all/Sparse_CQI'
output_folder = 'Training_data/Input'
merge_npy_files(folder1, folder2, folder3, folder4, folder5, output_folder)
