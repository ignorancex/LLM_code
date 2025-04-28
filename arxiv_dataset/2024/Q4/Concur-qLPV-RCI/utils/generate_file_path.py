def generate_file_path(folder_name, file_name, current_directory):
    file_name = f"{file_name}.pkl"
    folder_path = current_directory / "identification_results" / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)  
    file_path = folder_path / file_name  
    return file_path