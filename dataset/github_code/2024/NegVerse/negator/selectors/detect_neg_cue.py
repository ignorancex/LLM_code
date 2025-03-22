import torch
from transformers import BertTokenizer, BertForTokenClassification

# Load the fine-tuned model and tokenizer from the specified directory
model_dir = "negator/selectors/NegBERT"
CUE_MODEL = 'bert-base-uncased'
model = BertForTokenClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(CUE_MODEL, do_lower_case=True)

# Label mapping dictionary
label_map = {
    0: "Affix",
    1: "Normal Cue",
    2: "Part of a multiword cue",
    3: "Not a Cue"
}

# Set the model to evaluation mode
model.eval()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict labels for a given sentence
def predict_batch(sentences):
    # Tokenize the input sentence
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Run the model on the batch of tokenized sentences
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    # Get the logits and find the predicted labels for each token
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).cpu().numpy()

    # Convert token IDs back to words and get the corresponding labels
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids.cpu().numpy()]
    
    # Format the results
    results = []
    for sentence_tokens, sentence_labels in zip(tokens, predictions):
        labeled_tokens = [(token, label) for token, label in zip(sentence_tokens, sentence_labels)]
        results.append(labeled_tokens)
        #print(results)
    
    
    return tokens, predictions
    
# Function to filter sentences based on labels
def filter_neg_sentences(sentences):
    tokens, predictions = predict_batch(sentences)
    
    filtered_sentences = []
    for i, (sentence_tokens, sentence_labels) in enumerate(zip(tokens, predictions)):
        # Check if any label in the sentence is 0, 1, or 2 (negation cues labels)
        if any(label in {0, 1, 2} for label in sentence_labels):
            filtered_sentences.append(sentences[i])
    
    return filtered_sentences
