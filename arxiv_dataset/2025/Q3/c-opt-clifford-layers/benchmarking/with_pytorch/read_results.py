import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def parse_results(filename):
    """Parse the benchmark results file and extract timing information."""
    results = defaultdict(list)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    pattern = r"Time for (\d+D), agg=(\w+), backend_act=(\w+), backend_conv=(\w+): (\d+\.\d+) ms"
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            dimension, agg, backend_act, backend_conv, time = match.groups()
            time = float(time)
            
            # Create a key that uniquely identifies this configuration
            key = (dimension, backend_act, backend_conv)
            results[key].append(time)
    
    # Convert to DataFrame for easier analysis
    data = []
    for (dimension, backend_act, backend_conv), times in results.items():
        avg_time = sum(times) / len(times)
        data.append({
            'Dimension': dimension,
            'Backend_Act': backend_act,
            'Backend_Conv': backend_conv,
            'Avg_Time_ms': avg_time
        })
    
    return pd.DataFrame(data)

def main():
    try:
        df = parse_results("results.txt")
        print("Successfully parsed results file.")
        print(f"Found data for {len(df)} configurations.")
        
        # Display the raw data
        print("\nRaw Data:")
        print(df)

    except FileNotFoundError:
        print("Error: Could not find results.txt file.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"Error processing results: {e}")

if __name__ == "__main__":
    main()