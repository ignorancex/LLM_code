def parse_log_file(file_path):
    """
    Parse the log file and extract validation metrics for each epoch.
    Returns a list of dictionaries containing metrics for each epoch.
    """
    epochs = []
    current_metrics = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            if 'Validation Metrics' in line:
                metrics = {}
                # Split the line by comma and extract metrics
                parts = line.strip().split(',')
                for part in parts:
                    if ':' in part:
                        metric, value = part.split(':')
                        # Clean up metric name and convert value to float
                        metric = metric.strip().split(' - ')[-1]
                        metrics[metric] = float(value.strip())
                current_metrics = metrics
            
            elif 'EPOCH:' in line:
                # Extract epoch number
                epoch_num = int(line.split('EPOCH:')[1].split(',')[0].strip())
                # Combine current metrics with epoch number
                epoch_data = {
                    'epoch': epoch_num,
                    **current_metrics
                }
                epochs.append(epoch_data)
    
    return epochs

def find_best_epoch(epochs):
    """
    Find the epoch with the best combination of metrics.
    """
    best_epoch = None
    best_combined_score = float('-inf')
    
    for epoch in epochs:
        if all(key in epoch for key in ['Accuracy', 'Precision', 'Recall']):
            # Calculate combined score (you can modify this formula based on your needs)
            combined_score = epoch['Accuracy'] + epoch['Precision'] + epoch['Recall']
            
            if combined_score > best_combined_score:
                best_combined_score = combined_score
                best_epoch = epoch
    
    return best_epoch

def analyze_log_file(file_path):
    """
    Analyze a log file and print the best metrics.
    """
    print(f"\nAnalyzing file: {file_path}")
    print("-" * 50)
    file_path = "logs/" + file_path
    
    epochs = parse_log_file(file_path)
    best_epoch = find_best_epoch(epochs)
    
    if best_epoch:
        print(f"Best metrics found in epoch {best_epoch['epoch']}:")
        print(f"Accuracy:  {best_epoch['Accuracy']:.4f}")
        print(f"Precision: {best_epoch['Precision']:.4f}")
        print(f"Recall:    {best_epoch['Recall']:.4f}")
        if 'F1' in best_epoch:
            print(f"F1 Score:  {best_epoch['F1']:.4f}")
    else:
        print("No valid metrics found in the log file.")

def main():
    # List of log files to analyze
    log_files = ['moatabularckan.log', 'moatabularcnn.log']
    
    for log_file in log_files:
        try:
            analyze_log_file(log_file)
        except FileNotFoundError:
            print(f"\nError: File '{log_file}' not found")
        except Exception as e:
            print(f"\nError processing '{log_file}': {str(e)}")

if __name__ == "__main__":
    main()
