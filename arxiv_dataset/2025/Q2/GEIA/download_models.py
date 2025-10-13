import sys
from transformers import AutoModel, AutoTokenizer

cache_directory = sys.argv[1] if len(sys.argv) > 1 else None

# Load models with specified cache directory (or default)
sroberta = AutoModel.from_pretrained("sentence-transformers/all-distilroberta-v1", cache_dir=cache_directory)
simcse_bert = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased", cache_dir=cache_directory)
simcse_roberta = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base", cache_dir=cache_directory)
st5 = AutoModel.from_pretrained("sentence-transformers/sentence-t5-base", cache_dir=cache_directory)
mpnet = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2", cache_dir=cache_directory)

