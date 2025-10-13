import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG Experiment")

    parser.add_argument("--seed", type=int, default = 2024, help="seed")

    # Model and GPU
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path for the AI model to be used.")
    
    parser.add_argument('--rerank_model_path', type=str, default='./Rerank_train/Save_model/model_epoch_1.pth', help="Model path for the rerank model.")

    parser.add_argument('--gpu', type=int, default=2,
                        help="GPU device number to use.")
    
    # preprocess
    parser.add_argument("--source", default= "./data/dataset/ModC.pkl.gz", type=str, help="Queries and evidence to run the RAG experiment on")

    parser.add_argument("--n", type=int, default=100, help="number of pieces of evidence to retrieve for each question")

    parser.add_argument("--type", type=str, choices=['sentences', 'chunks'], default='chunks', help="splitting method type")

    parser.add_argument("--chunk_size", type=int, default=256, help="max chunk size if chunk is selected as splitting method")

    # RAG
    parser.add_argument('--k', default=5, type=int, help='Number of relevant sentences to keep for final evaluation')

    parser.add_argument("--with_MediaBG", type=str, choices=['true', 'false'], default='false', help="Whether to consider Media Background")

    parser.add_argument("--method", type=str, choices=["DirectAnswer", "Explain", "CoT", "MajorityVoting","AgentBased", "Filter", "RerankSoft", "RerankHard"], default="DirectAnswer", help="processing method")

    parser.add_argument("--media_data", type=str, choices=['mbfc', 'all'], default='all', help="Whether to consider media credibility from mbfc dataset only")

    return parser.parse_args()