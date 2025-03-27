import pandas as pd

# Assuming you have your data in a CSV file named 'data.csv'
data = pd.read_csv('../bbq_results/before_finetune/bbq_test_sexual_orientation_model_0.csv')

# Calculating the number of correct predictions
correct_predictions = sum(data['correct'] == data['predicted'])

# Calculating the total number of predictions
total_predictions = len(data)

# Calculating the accuracy
accuracy = correct_predictions / total_predictions

# Print the accuracy
print(f"Accuracy of the model: {accuracy}")
