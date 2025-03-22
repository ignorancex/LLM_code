import torch
import torch.nn.functional as F

def classify_claim(claim: str, context: str, model, tokenizer, device: torch.device):
    """
    Classifies the given claim based on the provided context using a pre-trained model.
    
    Args:
        claim (str): The claim that needs to be verified.
        context (str): The supporting or opposing context for the claim.
        model: The pre-trained classification model.
        tokenizer: The tokenizer corresponding to the model.
        device (torch.device): The device (CPU or GPU) on which to run the model.
    
    Returns:
        tuple: A tuple containing:
            - prob (torch.Tensor): The probability of the predicted class.
            - pred (int): The predicted class label.
    """
    
    model.to(device)
    model.eval()

    encoding = tokenizer(
        claim,
        context,
        truncation="only_second",
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probabilities = F.softmax(outputs["logits"], dim=1)
    prob, pred = torch.max(probabilities, dim=1)

    return prob, pred.item()
