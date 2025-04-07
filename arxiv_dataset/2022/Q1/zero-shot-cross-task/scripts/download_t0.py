from transformers import T5ForConditionalGeneration, AutoTokenizer

T5ForConditionalGeneration.from_pretrained("bigscience/T0_3B")
AutoTokenizer.from_pretrained("bigscience/T0_3B")
