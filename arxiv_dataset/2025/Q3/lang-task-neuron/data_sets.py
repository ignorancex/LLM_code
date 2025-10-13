class Dataset:
    def __init__(self, dataset_name, ds):
        self.dataset_name = dataset_name
        self.ds = ds
    @staticmethod
    def check_index_prompt(infer_model, when, prompt, model_name): # TODO: method ini pindahin ke class infer_model dan hapus printnya 
        """
        return len tokenized text when it is cut (excluding the chat template), len tokenized text 
        """
        text = ""
        # print(f"model_name: {model_name}")
        if model_name.startswith('Qwen/Qwen2.5'):
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                {"role": "user", "content": prompt}]
            text = infer_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            sentence_tokens = infer_model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            sentence_ids = sentence_tokens["input_ids"]
            sentence_token_texts = infer_model.tokenizer.convert_ids_to_tokens(sentence_ids)[:-5]
            # print(f"{when} sentence_token_texts {sentence_token_texts}. len(sentence_token_texts) {len(sentence_token_texts)}")
            if when == "prompt_before":
                return len(sentence_token_texts)+1, len(sentence_ids)
            return len(sentence_token_texts), len(sentence_ids)
            # print(f"####### {text}")
        elif model_name.startswith('google/gemma'):
            messages = [
                {"role": "user", "content": prompt}
                ]
            text = infer_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            sentence_tokens = infer_model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            sentence_ids = sentence_tokens["input_ids"]
            sentence_token_texts = infer_model.tokenizer.convert_ids_to_tokens(sentence_ids)[:-5]
            # print(f"{when} sentence_token_texts {sentence_token_texts}. len(sentence_token_texts) {len(sentence_token_texts)}")
            if when == "prompt_before":
                return len(sentence_token_texts), len(sentence_ids)
            return len(sentence_token_texts), len(sentence_ids)
        elif model_name.startswith('SeaLLMs'):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = infer_model.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            sentence_tokens = infer_model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            sentence_ids = sentence_tokens["input_ids"]
            sentence_token_texts = infer_model.tokenizer.convert_ids_to_tokens(sentence_ids)[:-5]
            # print(f"{when} sentence_token_texts {sentence_token_texts}. len(sentence_token_texts) {len(sentence_token_texts)}")
            if when == "prompt_before":
                return len(sentence_token_texts)+1, len(sentence_ids)
            return len(sentence_token_texts), len(sentence_ids)
        assert text != "", "Model does not exist!"
        

class Massive(Dataset):
    def __init__(self, dataset_name, ds):
        dataset_name = "AmazonScience/massive"
        super().__init__( dataset_name, ds)
    def get_index_start_end_prompt(self, infer_model, data, model_name, is_predict=True, take_whole=False):
        prompt_before = prompt_after = prompt_whole = ""
        utterance, options = data
        prompt_before = (
            f"""
            Instruction: Classify the intent of the following utterance.
            Utterance:"""
            )
        prompt_after = (
            f"""
            Instruction: Classify the intent of the following utterance.  
            Utterance: {utterance}.  
            """
            )
        prompt_whole = (
            f"""
            Instruction: Classify the intent of the following utterance.  
            Utterance: {utterance}.  
            Options: {options}. 
            Intent: 
            """
            )

        if not is_predict:
            prompt_whole = prompt_after = f"""
            {utterance}"""
            prompt_before = ""
        if is_predict and take_whole:
            prompt_before = prompt_after
        assert prompt_whole != "", "dataset invalid"
        
        id_prompt_start, _ = self.check_index_prompt(infer_model, "prompt_before", prompt_before, model_name)
        id_prompt_end, _ = self.check_index_prompt(infer_model,"prompt_after", prompt_after, model_name)
        _, len_sentence = self.check_index_prompt(infer_model, "prompt_whole", prompt_whole, model_name)
        if is_predict and take_whole:
            id_prompt_start = 0
            id_prompt_end = len_sentence
        return id_prompt_start, id_prompt_end, len_sentence, prompt_whole
    def get_detail_inference(self, data):
        intents = self.ds.features['intent'].names
        scenarios = self.ds.features['scenario'].names
        label_intent = intents[data['intent']]
        label_intent_no = -1
        scenario = scenarios[data['scenario']]
        
        intent_options = [intent[intent.find('_')+1:] for intent in intents if intent.startswith(scenario)]
        utterance = data['utt']
        options = f"{scenario} {', '.join(intent_options)}"
        detail_data = utterance, options
        return detail_data

class Xcopa(Dataset): # only premise for non predict
    def __init__(self, dataset_name, ds):
        dataset_name = "cambridgeltl/xcopa"
        super().__init__( dataset_name, ds)
    def get_index_start_end_prompt(self, infer_model, data, model_name, is_predict=True, take_whole=False):
        premise, choice1, choice2, question_type, label = data
        prompt_before = (
            f"""
            Premise:""")
        prompt_after = (
            f"""
            Premise: {premise}
            """)
        prompt_whole = (
            f"""
            Premise: {premise}
            I'm hesitating between the two options. Help me choose the {question_type}: 
            - {choice1} 
            - {choice2}
            """)
        if not is_predict:
            prompt_whole = prompt_after = (f"""
            {premise}
            """)
            prompt_before = ""
        if is_predict and take_whole:
            prompt_before = prompt_after
        
        assert prompt_whole != "", "dataset invalid"
        
        id_prompt_start, _ = self.check_index_prompt(infer_model, "prompt_before", prompt_before, model_name)
        id_prompt_end, _ = self.check_index_prompt(infer_model,"prompt_after", prompt_after, model_name)
        _, len_sentence = self.check_index_prompt(infer_model, "prompt_whole", prompt_whole, model_name)
        if is_predict and take_whole:
            id_prompt_start = 0
            id_prompt_end = len_sentence
        return id_prompt_start, id_prompt_end, len_sentence, prompt_whole
    def get_detail_inference(self, data):
        question_type = "cause" if data['question'] == "cause" else "effect"
        data_detail = data['premise'], data['choice1'], data['choice2'], question_type, data['label']
        return data_detail


        
class Xwinograd(Dataset):
    def __init__(self, dataset_name, ds):
        dataset_name = "Muennighoff/xwinograd"
        super().__init__( dataset_name, ds)
    def get_index_start_end_prompt(self, infer_model, data, model_name, is_predict=True, take_whole=False):
        sentence, option1, option2, answer = data
        prompt_before = (
            f"""""")
        prompt_after = (
            f"""
            {sentence}""")
        prompt_whole = (
            f"""
            {sentence}
            Who does '_' refer to? The answer should be one of '{option1}' or '{option2}'.\n
            Answer:""")
        replacer = option1 if answer == '1' else option2
        question = sentence.replace('_', replacer)
        if not is_predict:
            prompt_whole = prompt_after = (
            f"""
            {question}""")
            prompt_before = ""
        
        assert prompt_whole != "", "dataset invalid"
        
        id_prompt_start, _ = self.check_index_prompt(infer_model, "prompt_before", prompt_before, model_name)
        id_prompt_end, _ = self.check_index_prompt(infer_model,"prompt_after", prompt_after, model_name)
        _, len_sentence = self.check_index_prompt(infer_model, "prompt_whole", prompt_whole, model_name)
        if is_predict and take_whole:
            id_prompt_start = 0
            id_prompt_end = len_sentence
        return id_prompt_start, id_prompt_end, len_sentence, prompt_whole

    def get_detail_inference(self, data):
        # sentence = data['sentence']
        # option1 = data['option1']
        # option2 = data['option2']
        # answer = data['answer']
        # replacer = option1 if answer == '1' else option2
        # question = sentence.replace('_', replacer)
        data_detail = data['sentence'], data['option1'], data['option2'], data['answer']
        # data_detail = question/
        return data_detail


class Flores(Dataset):
    def __init__(self, dataset_name, ds):
        dataset_name = "Muennighoff/flores200"
        super().__init__( dataset_name, ds)
    def get_index_start_end_prompt(self, infer_model, data, model_name, is_predict=True, take_whole=False):
        sentence = data
        prompt_before = (
            f"""Translate this sentence:""")
        prompt_after = (
            f"""Translate this sentence:
            {sentence}""")
        prompt_whole = (
            f"""Translate this sentence:
            {sentence}""")

        if not is_predict:
            prompt_whole = prompt_after = (f"""
            {sentence}""")
            prompt_before = ""
        if is_predict and take_whole:
            prompt_before = prompt_after
        
        assert prompt_whole != "", "dataset invalid"
        id_prompt_start, _ = self.check_index_prompt(infer_model, "prompt_before", prompt_before, model_name)
        id_prompt_end, _ = self.check_index_prompt(infer_model,"prompt_after", prompt_after, model_name)
        _, len_sentence = self.check_index_prompt(infer_model, "prompt_whole", prompt_whole, model_name)
        if is_predict and take_whole:
            id_prompt_start = 0
            id_prompt_end = len_sentence
        return id_prompt_start, id_prompt_end, len_sentence, prompt_whole

    def get_detail_inference(self, data):
        data_detail = data['sentence']
        return data_detail



class Mlama(Dataset):
    def __init__(self, dataset_name, ds):
        dataset_name = "inayarhmns/MLAMA-dod-185"
        super().__init__( dataset_name, ds)
    def get_index_start_end_prompt(self, infer_model, data, model_name, is_predict=True, take_whole=False):
        template, subjek, objek = data
        question = template.replace("[X]", subjek)
        if not is_predict:
            question = question.replace("[Y]", objek)
        prompt_before = (
            f"""""")
        prompt_after = (
            f"""
            {question}""")
        prompt_whole = (
            f"""
            {question}""")
        if is_predict and take_whole:
            prompt_before = prompt_after
        
        assert prompt_whole != "", "dataset invalid"
        
        id_prompt_start, _ = self.check_index_prompt(infer_model, "prompt_before", prompt_before, model_name)
        id_prompt_end, _ = self.check_index_prompt(infer_model,"prompt_after", prompt_after, model_name)
        _, len_sentence = self.check_index_prompt(infer_model, "prompt_whole", prompt_whole, model_name)
        if is_predict and take_whole:
            id_prompt_start = 0
            id_prompt_end = len_sentence
        return id_prompt_start, id_prompt_end, len_sentence, prompt_whole

    def get_detail_inference(self, data):
        data_detail = data['template'], data['sub_label'] , data['obj_label']
        return data_detail


def map_language(entry, flores_to_text=True):
    langs_mapper = {
        "English": "eng_Latn",
        "Haitian Creole" : "hat_Latn",
        "Quechua": "quy_Latn",
        "Swahili": "swh_Latn",
        "Thai" : "tha_Thai",
        "Dutch": "nld_Latn",
        "Indonesian": "ind_Latn",
        "Malay": "zsm_Latn",
        "Vietnamese": "vie_Latn",
        "Japanese": "jpn_Jpan",
        "Chinese": "zho_Hans",
        "French": "fra_Latn",
        "Portuguese":"por_Latn",
        "Russian": "rus_Cyrl",
        "Estonian":"est_Latn" ,
        "Italian": "ita_Latn",
        "Tamil": "tam_Taml",
        "Turkish": "tur_Latn"
    }
    flores_text = {v:k for k, v in langs_mapper.items()}
    if flores_to_text:
        return flores_text[entry]
        