import torch
import os

# Define the directory where you want to save the files
directory = '/scratch/qilong3/transferattack/message/'

# Ensure the directory exists
os.makedirs(directory, exist_ok=True)

# Generate and save 100 .pth files
for k in range(1, 101):
    # Generate a 100x30 tensor of random 0s and 1s
    tensor = torch.randint(0, 2, (100, 30), dtype=torch.uint8)

    # Define the file name
    file_name = f'30bits_message_{k}.pth'
    file_path = os.path.join(directory, file_name)

    # Save the tensor to the file
    torch.save(tensor, file_path)

    print(f'Saved {file_path}')
