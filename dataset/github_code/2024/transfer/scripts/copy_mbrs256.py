import shutil

# Base paths
base_path = "/scratch/qilong3/transferattack/results/0.25/"
sub_dir_template = "hidden_to_mbrs_cnn_30_to_256bitsAT_{models}models/"
source_filename = "results.log"
destination_template = "/scratch/qilong3/transferattack/results/hidden_to_mbrs_cnn_30_to_256bitsAT_{models}models.log"

# Values for the models
model_values = [1, 2, 5, 10, 20, 30, 40, 50]

# Copy files for each model value
for model in model_values:
    # Generate source and destination paths
    source_path = f"{base_path}{sub_dir_template.format(models=model)}{source_filename}"
    destination_path = destination_template.format(models=model)

    try:
        # Perform the copy operation
        shutil.copyfile(source_path, destination_path)
        print(f"Copied: {source_path} -> {destination_path}")
    except FileNotFoundError:
        print(f"File not found: {source_path}")
    except Exception as e:
        print(f"Error copying {source_path} to {destination_path}: {e}")
