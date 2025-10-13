import json
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import nltk
from experimental_model_viewer.Spider_utils import format_prompt
from experimental_model_viewer.measurement_utils import load_model
nltk.download('punkt_tab')


def main():
    parser = argparse.ArgumentParser(description='Spider dataset text-to-SQL inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Hugging Face model name or path')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input JSON file (e.g., dev.json)')
    parser.add_argument('--tables', type=str, required=True,
                        help='Path to tables.json file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path for debugging')
    parser.add_argument('--max_length', type=int, default=2048,
                        help='Maximum generation length')
    parser.add_argument("--predictions_filename", type=str, required=True,
                        help='Output file path for predictions')
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--addition_dir", type=str, required=True)
    parser.add_argument("--brainfloat", action="store_true")
    parser.add_argument('--output_savedir', type=str, required=True)
    args = parser.parse_args()
    os.makedirs(args.output_savedir, exist_ok=True)

    model_info = load_model(engine=args.model, checkpoints_dir=args.addition_dir, brainfloat=args.brainfloat)
    model, tokenizer = model_info["model"], model_info["tokenizer"]

    # Load data files
    with open(args.input) as f:
        data = json.load(f)
    with open(args.tables) as f:
        all_tables = json.load(f)

    # Create database ID to schema mapping
    db_schema = {db['db_id']: db for db in all_tables}

    predictions = []
    questions = []
    short_questions = []
    answers = []
    for idx, item in enumerate(data):
        if args.testing and idx > 3:
            break
        # Get database schema information
        db_id = item['db_id']
        db_info = db_schema[db_id]
        
        # Format input text
        input_text = format_prompt(item['question'], db_info)
        # Apply chat template
        inputs_text = tokenizer.apply_chat_template(
            input_text,
            add_generation_prompt=True,
            tokenize=False
        )
        inputs = tokenizer(inputs_text, truncation=False, padding=False, max_length=args.max_length, return_tensors="pt")["input_ids"].to(model.device)
        print("inputs", tokenizer.decode(inputs[0]), "inputs_text", inputs_text, inputs_text == tokenizer.decode(inputs[0]))

        questions.append(input_text)
        short_questions.append(item['question'])
        answers.append(item["query"])
        
        # Generate response
        outputs = model.generate(
            inputs,
            max_new_tokens=args.max_length,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id  # Llama-3 doesn't have pad token
        )
        
        # Decode and store prediction
        pred = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        predictions.append(pred.split(";")[0].strip().replace("\n", " ") + ";")  # Take first SQL statement

    # Save predictions
    with open(args.output, 'w') as f:
        for prediction, question, short_question, answer in zip(predictions, questions, short_questions, answers):
            f.write(f"---\n {question=}\n{prediction=}\n{answer=}\n")
    with open(os.path.join(args.output_savedir, args.predictions_filename), "w") as f:
        f.write("\n".join(predictions))

    print(f"Saved debug to {args.output}; saved predictions to {os.path.join(args.output_savedir, args.predictions_filename)}")

if __name__ == "__main__":
    main()