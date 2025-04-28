from huggingface_hub import HfApi

from huggingface_hub import login

# Replace 'your_token_here' with your actual Hugging Face token
login(token="your_token_here")

# Initialize the HfApi client
api = HfApi()

# Specify the repository ID (e.g., 'your_username/your_repo_name')
repo_id = "benstaf/nasdaq_2013_2023"

# List of files to upload
files_to_upload = [
 #   "trade_data_deepseek_risk_2019_2023.csv",
    "trade_data_deepseek_sentiment_2019_2023.csv",
#    "trade_data_llama_risk_2019_2023.csv",
   # "trade_data_llama_sentiment_2019_2023.csv",
  #  "train_data_deepseek_risk_2013_2018.csv",
    "train_data_deepseek_sentiment_2013_2018.csv",
#    "train_data_llama_risk_2013_2018.csv",
 #   "train_data_llama_sentiment_2013_2018.csv"
]

# Upload each file
for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,  # The path in the repo (same as local path in this case)
        repo_id=repo_id,
        repo_type="dataset"  # Use "model" if uploading to a model repo
    )
