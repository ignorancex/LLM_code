from unsloth import FastLanguageModel
import torch

# settings
max_seq_length = 2048  
dtype = None  
load_in_4bit = True  

# load model and tokenizer from huggingface hub
model_hf, tokenizer_hf = FastLanguageModel.from_pretrained(
    model_name="hyoo14/Meta-Llama-3.1-8B-Instruct-bnb-4bit_DNA_AMR",  # funetuned model for dna amr drug classification
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

# change model to LoRA based PEFT model
model_hf = FastLanguageModel.get_peft_model(
    model_hf,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# model for inference
model_hf = FastLanguageModel.for_inference(model_hf)

# set prompt
alpaca_prompt = "### Instruction:\n{}\n### Input:\n{}\n### Response:\n{}"

inputs = tokenizer_hf(
    [
        alpaca_prompt.format(
            "Tell me the resistance drug given a DNA sequence and a list of drug classes.",
            "Tell me the resistance drug among drugs (Sulfonamides, Aminoglycosides, betalactams, Glycopeptides, Tetracyclines, Phenicol, Fluoroquinolones, MLS, Multi-drug_resistance) with DNA sequence (ATGAATCCCTATCTACAGTTAGTTAGCAAAGAATTTCCGTTAGAAAAAAACCAAGAACCCCCTCATTTAGTCCTTGCTGCCTTCAGCGAAGACGAGGTTTACTTGCAGCCAGAAGCAGCCAAACAATGGGAACGGCTCGTCAAAGCCTTAAAGCTTGAAAACGAGATCTGCTTGTTAGATGGCTACCGAACCGAAAAGCAGCAACGGTATTTATGGGAATATTCCTTGAAAGAAAATGGGCTAGCCTATACGAAACAATTTGTGGCATTGCCCGGTTGCAGTGAACATCAGTTAGGCTTAGCGATCGATGTTGGTTTAAAGGGCAGCCAGGATGATTTGATCTGCCCACTATTTCGTGACAGCGCAGCCGCTGATTTATTTACCCAGGAAATGATGAACTATGGGTTTATTTTACGCTATCCCGCAGACAAACAGGAGATTACAGGGATTGGCTATGAACCATGGCATTTCCGGTATGTCGGGCTGCCTCATAGTCAAATCATCGCCAGTCAACAGTGGACCTTGGAAGAATACCATCAATACCTTGAACAAACTGCGAGGCAGTTCGCATGA)?",
            ""
        )
    ],
    return_tensors="pt"
)

# get the output 
outputs = model_hf.generate(input_ids=inputs['input_ids'], max_new_tokens=64, use_cache=True)

# decode the output
decoded_outputs = tokenizer_hf.batch_decode(outputs, skip_special_tokens=True)

print(decoded_outputs)
