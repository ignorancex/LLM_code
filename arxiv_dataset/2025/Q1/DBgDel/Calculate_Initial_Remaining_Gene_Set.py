import os
import pandas as pd

def read_csv_files(folder_path):
    # Define the subdirectory
    subfolder_path = os.path.join(folder_path, "Example_gene_deletion_data")
    
    # Ensure the directory exists
    if not os.path.exists(subfolder_path):
        print(f"Warning: The folder '{subfolder_path}' does not exist.")
        return []

    # Get a list of all CSV files in the subfolder
    csv_files = [f for f in os.listdir(subfolder_path) if f.endswith('.csv')]
    
    # Print the number of CSV files found
    print(f"Found {len(csv_files)} CSV files in '{subfolder_path}'.")

    # Initialize a list to hold sets of data from each CSV file
    data_sets = []
    
    for csv_file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(os.path.join(subfolder_path, csv_file))
        
        # Ensure there's only one column and get the data as a set (excluding the title)
        if len(df.columns) == 1:
            data_set = set(df.iloc[1:, 0])  # Exclude the title row
            data_sets.append(data_set)
        else:
            print(f"Warning: {csv_file} has more than one column.")
    
    return data_sets

def calculate_intersection(data_sets):
    # Calculate the intersection of all sets
    if data_sets:
        intersection_set = set.intersection(*data_sets)
        print(f"Initial remaining gene set: {len(intersection_set)} genes.")
        return intersection_set
    else:
        print("Initial remaining gene set: 0 gene.")
        return set()

def save_to_csv(data_set, output_file):
    # Convert the set to a DataFrame
    df = pd.DataFrame(list(data_set), columns=['Remaining_gene'])
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

def main(folder_path, output_file):
    # Read all CSV files and get their remaining gene as sets
    data_sets = read_csv_files(folder_path)
    
    # Calculate the intersection of the remaining gene sets
    common_data = calculate_intersection(data_sets)
    
    # Save the result to a new CSV file
    save_to_csv(common_data, output_file)
    print(f"The initial remaining gene set has been saved to {output_file}")

if __name__ == "__main__":
    # Get the path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set folder path to the script's directory
    folder_path = script_dir
    
    # Set output file path within the script's directory
    output_file = os.path.join(script_dir, "Remaining_gene.csv")

    # # Change to your folder path
    # folder_path = "path"
    
    # # Change to your desired output file path
    # output_file = "path/Remaining_gene.csv" 
    
    main(folder_path, output_file)
