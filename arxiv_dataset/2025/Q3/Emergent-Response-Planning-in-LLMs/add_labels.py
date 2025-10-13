import argparse
import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, List, Optional, Tuple, Union
import re
from data_utils import DatasetManager
from model_utils import ModelWrapper


class DataProcessor:
    def __init__(self, global_config: Dict, model_name, tokenizer, template_type: str, seed, dataset_name):
        self.global_config = global_config
        self.template_config = global_config['template_types'][template_type]
        self.tokenizer = tokenizer
        self.template_type = template_type
        self.MAX_TOKEN_LENGTH = int(global_config['template_types'][template_type].get('max_token_length_for_label', 1000))
        self.is_chat_model = any([x in model_name.lower() for x in ['chat', 'instruct']])
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.seed = seed

    def process_dataset(self, dataset: pd.DataFrame, augmentation_type: str = None) -> Tuple[pd.DataFrame]:
        """
        Process dataset by labeling and optionally augmenting the data.
        
        Args:
            dataset: Input DataFrame with 'prompt' and 'response' columns
            augmentation_type: Type of augmentation ('', 'balanced', 'linear')
            
        Returns:
            Processed DataFrame and optional list of truncated responses
        """
        # preprocess responses: for non-chat models, need to truncate
        dataset['response'] = dataset['response'].apply(self._pre_process)

        # Process labels for the entire dataset at once
        dataset['label'] = self._get_labels(dataset)
        # drop rows where label is None; reset index
        if augmentation_type != 'raw':
            dataset = dataset.dropna(subset=['label'], axis=0).reset_index(drop=True)
        
        # Return early if no augmentation is needed
        if not augmentation_type or augmentation_type == '' or augmentation_type == 'raw':
            dataset['augmented_labels'] = dataset['label'].apply(lambda x: [x] if x and x <= self.MAX_TOKEN_LENGTH else [])
            # dataset['truncated_responses'] = dataset['response'].apply(
            #     lambda x: [] if len(x) > self.MAX_TOKEN_LENGTH else [x]
            # )
            # all items set to None
            dataset['truncated_responses'] = dataset['response'].apply(
                lambda x: []
            )

            return dataset
            
        # # Perform augmentation based on template type
        if self.template_type in ['token_length']:
            return self._augment_token_length(dataset, augmentation_type)
        elif self.template_type in ['cot_reasoning']:
            return self._augment_cot_reasoning(dataset, augmentation_type)
        elif self.template_type in ['multiple_choice']:
            return self._augment_multiple_choice(dataset, augmentation_type)
        elif self.template_type in ['story_continuation']:
            return self._augment_story_continuation(dataset, augmentation_type)
        elif self.template_type in ['multiple_choice_selection']:
            return self._augment_multiple_choice_selection(dataset, augmentation_type)
        elif self.template_type in ['truthfulness']:
            return self._augment_truthfulness(dataset)
        else:
            raise NotImplementedError(f"Augmentation not implemented for {self.template_type}")

    def _pre_process(self, response: str) -> str:
            """
            Pre-process response for non-chat models by truncating at the end sequence.
            For chat models, return the original response.
            
            Args:
                response: Raw response string
                
            Returns:
                Processed response string
            """
            if self.is_chat_model:
                return response
                
            end_sequence = self.template_config['end_sequence']
            if end_sequence in response:
                # Split by end sequence and take the content before the first occurrence
                parts = response.split(end_sequence)
                return parts[0].strip()
            return response
    
    def _get_labels(self, dataset: pd.DataFrame) -> pd.Series:
        """
        Process labels for the entire dataset at once.
        If a response contains multiple strings (as a list), calculates labels for each string
        and returns their average value.
        """
        def process_single_or_list(row):
            # Convert single item to list for unified processing
            response = row['response']
            label = getattr(row, 'label', None)
            
            # pre-process response
            if isinstance(response, list):
                items = response
            else:
                items = [response]
            labels = label if isinstance(label, list) else [label]
            
            if len(items) == 0:
                return 0  # or another default value for empty lists
            
            if self.template_type == 'random_translate':
                key_words = ['Chinese', 'French']
                labels = [self._label_random_translate(x, key_words) for x in items]
            elif self.template_type in ['summarize', 'token_length']:
                labels = [self._label_token_length(x) for x in items]
            elif self.template_type == 'cot_reasoning':
                labels = [self._label_cot_reasoning(x) for x in items]
            elif self.template_type == 'multiple_choice':
                labels = [self._label_multiple_choice(x, l) for x, l in zip(items, labels)]
            elif self.template_type == 'story_continuation':
                labels = [self._label_story_continuation(x) for x in items]
            elif self.template_type == 'multiple_choice_selection':
                labels = [self._label_multiple_choice_selection(x) for x in items]
            elif self.template_type == 'truthfulness':
                labels = [self._label_truthfulness(x, l) for x, l in zip(items, labels)]
                raise NotImplementedError(f"Labeling not implemented for {self.template_type}")
            
            # filter out None values
            labels = [x for x in labels if x is not None]
            return np.mean(labels) if labels else None

        return dataset.apply(process_single_or_list, axis=1)
    
    def _label_random_translate(self, response: str, key_words: List[str]) -> Optional[int]:
        counts = {key: response.count(key) for key in key_words}
        main_key_word = [key for key, count in counts.items() if count >= 1]
        return key_words.index(main_key_word[0]) if len(main_key_word) == 1 else None

    def _label_token_length(self, response: str) -> int:
        tokens = self.tokenizer(response, return_tensors='pt', add_special_tokens=False).input_ids[0]
        return len(tokens)
    
    def _label_cot_reasoning(self, response: str) -> Optional[int]:
        """
        Count the number of reasoning steps in a chain-of-thought response.
        Returns the number of steps if the response follows the step format, None otherwise.
        
        Example input: "Step 1: First analyze... Step 2: Then calculate... Step 3: Finally..."
        Returns: 3
        """
        # Convert to lowercase and handle variations in step formatting
        response = response.lower()
        
        # Look for different possible step formats
        step_patterns = [
            r"step\s*\d+\s*:",  # matches "step 1:", "step1:", "step 1 :"
            # r"\d+\)\s*",        # matches "1)", "1) "
            # r"\d+\.\s*",        # matches "1.", "1. "
        ]
        
        max_steps = 0
        for pattern in step_patterns:
            # Find all matches for current pattern
            steps = re.findall(pattern, response)
            if steps:
                # Extract numbers from the steps
                numbers = [int(re.search(r'\d+', step).group()) for step in steps]
                # Check if steps are sequential and start from 1
                if numbers == list(range(1, len(numbers) + 1)):
                    max_steps = max(max_steps, len(numbers))
                else:
                    # If steps are not sequential, return None
                    return None
        
        return max_steps if max_steps > 0 else None

    @staticmethod
    def _match_choice(text: str, target_choice: Optional[str] = None) -> Union[set[str], list[int]]:
        """
        Find all potential multiple choice answers (A-E) in the given text using regex patterns,
        or find positions of a specific choice if target_choice is provided.
        
        Args:
            text: The text to search for answer patterns
            target_choice: Optional specific choice to find positions for
            
        Returns:
            set[str]: Set of unique matched choices (A-E) if target_choice is None
            list[int]: List of positions where target_choice was found if provided
        """
        patterns = [
            r"ANSWER IS ([A-E])",              # Matches: "ANSWER IS B"
            r"([A-E]) IS THE CORRECT ANSWER",  # Matches: "A IS THE CORRECT ANSWER"
            r"THE ANSWER IS ([A-E])",          # Matches: "THE ANSWER IS A"
            r"\(([A-E])\)",                    # Matches: "(A)"
            r"OPTION ([A-E])",                 # Matches: "OPTION A"
            r"CHOICE\s*([A-E])\b",             # Matches: "CHOICE A" or "CHOICE    B"
            r"OPTION ([A-E])\b(?!XPLANATION)", # Matches: "OPTION A" but not "OPTION EXPLANATION"
            r"^([A-E])[\.\)]\s",              # Matches: "A. " or "A) " at start of line
            r"\s([A-E])\)\s",                 # Matches: " B) " with spaces around
            r"\s\*?\*?([A-E])\)",             # Matches: " A)", " *A)", or " **A)"
            r"\s([A-E])\.?$",                 # Matches: " A" or " A." at end of line
            r"(?:^|\s)([A-E])\)"              # Matches: "A)" or " A)" at start/after space
            r"\*?\*?([A-E])\)"                # Matches: "A)", "*A)", or "**A)"
        ]
        
        if target_choice:
            target_choice = target_choice.upper()
            positions = [
                match.start()
                for pattern in patterns
                for match in re.finditer(pattern, text)
                if match.group(1) == target_choice
            ]
            return sorted(positions)
        
        # return {
        #     match.group(1)
        #     for pattern in patterns
        #     for match in re.finditer(pattern, text)
        # }
        choices = {
            match.group(1)
            for pattern in patterns
            for match in re.finditer(pattern, text)
        }
        # if len(choices) != 1:
        #     breakpoint()
        return choices

    @staticmethod
    def _match_truthfulness(text: str) -> Optional[bool]:
        """
        Find support/non-support indicators in the text.
        
        Args:
            text: The text to search for truthfulness patterns
            
        Returns:
            bool: True if text indicates support, False if non-support, None if unclear
        """
        # Store patterns in lowercase for readability
        support_patterns = [
            # Direct support patterns
            r"(?:this|the) statement is (?:accurate|true|correct)",
            r"(?:this|the) (?:information|claim) is (?:accurate|true|correct)",
            
            # Personal stance patterns
            r"i [completely ](?:support|verify|confirm|agree with) (?:this|the) statement",
            r"i am in favo[u]?r of (?:this|the) statement",
            r"i confirm (?:this|the) (?:accuracy|statement)",
            
            # General truth assertions
            r"(?:this|the) is a true statement",
            r"(?:this|the) information is reliable",
            r"(?:this|the) claim is (?:valid|accurate)",
            r"yes, i agree that",
            r"i completely agree!",
            
            r"i support this statement as accurate",
            r"100% accurate",
            r"100% confidence",
            r"am confidence",
            r"very confident",
            r"confidence level as 1",
            # r"1.",

        ]
        
        non_support_patterns = [
            # Direct negation patterns
            r"(?:this|the) statement is (?:false|incorrect|inaccurate)",
            r"(?:this|the) (?:information|claim) is (?:false|incorrect|inaccurate)",
            
            # Personal stance patterns
            r"i (?:disagree with|reject|cannot verify) (?:this|the) statement",
            r"i am not in favo[u]?r of (?:this|the) statement",
            r"i cannot be confident (?:about|in) (?:this|the) (?:accuracy|statement)",
            
            # Misinformation patterns
            r"(?:this|the) is (?:false|incorrect) (?:information|statement)",
            r"(?:this|the) contains misinformation",
            r"i am in favo[u]?r of correcting.*misinformation",
            
            # Negation with evidence
            r"(?:this|the) claim (?:lacks|has no) evidence",
            r"(?:this|the) statement is not (?:supported|verified|true)",
            
            # Express unsure
            r"i am unsure (?:about|of) (?:this|the) statement",
            r"i am unsure (?:about|of)",
            r"(?:this|the|that) statement is incorrect",
            r"i cannot be confident",

            r"am uncertain",
            r"confidence level as 0",
            # r"0.",
        ]
        
        # Convert input text to lowercase for case-insensitive matching
        text = text.lower()
        
        # Check for support patterns
        support_matches = sum(1 for pattern in support_patterns if re.search(pattern, text))
        
        # Check for non-support patterns
        non_support_matches = sum(1 for pattern in non_support_patterns if re.search(pattern, text))
        
        # Return result based on pattern matches
        if support_matches > 0 and non_support_matches == 0:
            return True
        elif non_support_matches > 0 and support_matches == 0:
            return False
        else:
            # breakpoint()
            return None  # Unclear or conflicting indicators
        
    def _label_multiple_choice(self, response: str, label: int) -> Optional[int]:
        """
        Extract answer choice and verify if it matches the ground truth.
        Returns 1 if match, 0 if mismatch, None if invalid/multiple/no answers.
        """
        text = response.upper()
        
        choices = self._match_choice(text)
        
        # Handle special cases
        if len(choices) == 0:
            return None  # No clear answer found
        elif len(choices) > 1:
            return None  # Multiple different answers
        
        # Convert the single answer to numeric index (0-3)
        answer_idx = ord(choices.pop()) - ord('A')
        
        # Compare with ground truth
        return 1 if answer_idx == label else 0

    def _label_multiple_choice_selection(self, response: str, label: Optional[int] = None) -> Optional[int]:
        """
        Extract which answer choice (A-E) was selected, mapping to indices 0-4.
        Returns the index of the selected choice, or None if invalid/multiple/no answers.
        
        Args:
            response: The model's response text
            label: Optional original label (unused in this function, included for API compatibility)
            
        Returns:
            int: Index of selected choice (0-4 for A-E) or None if invalid
        """
        text = response.upper()
        choices = self._match_choice(text)
        
        # Handle special cases
        if len(choices) != 1:  # No answer or multiple answers
            return None
            
        # Convert the single answer to numeric index (0-4 for A-E)
        answer_idx = ord(choices.pop()) - ord('A')
        return answer_idx if 0 <= answer_idx <= 4 else None

    def _label_story_continuation(self, response: str) -> Optional[int]:
        """
        Label story continuation based on the animal that appears in the response.
        Returns the index of the animal in the common animals list, or None if no common animal is found.
        """
        # Define common animals to look for
        common_animals = [
            'cat', 'dog', 'rabbit', 'bird', 'squirrel', 'mouse', 'rat',
            'hamster', 'fish', 'parrot', 'owl', 'raccoon', 'fox', 'bear',
            'deer', 'horse', 'pig', 'monkey', 'elephant', 'lion', 'tiger',
            'wolf', 'dolphin', 'whale', 'octopus', 'penguin', 'duck',
            'chicken', 'turtle', 'snake', 'butterfly', 'spider', 'bee'
        ]
        
        # Convert response to lowercase for case-insensitive matching
        response = response.lower()
        
        # Find all animal mentions in the response
        found_animals = []
        for i, animal in enumerate(common_animals):
            # Check for both singular and plural forms
            if animal in response or f"{animal}s" in response:
                found_animals.append((i, animal))
        
        if not found_animals:
            return None
            
        # If multiple animals found, use the first one that appears in the text
        if len(found_animals) > 1:
            return None
            # positions = [(animal[0], response.find(animal[1])) for animal in found_animals]
            # # Sort by position in text and take the first one
            # positions.sort(key=lambda x: x[1])
            # return positions[0][0]
            
        # Return index of the single found animal
        return found_animals[0][0]
    
    def _label_truthfulness(self, response: str, ground_truth: str) -> Optional[int]:
        """
        Label truthfulness recognition based on support/non-support and ground truth.
        
        Args:
            response: The model's response text
            ground_truth: Ground truth label ('true' or 'false')
            
        Returns:
            int: Label encoding as follows:
                0: false statement, non-support (correct)
                1: false statement, support (incorrect)
                2: true statement, non-support (incorrect)
                3: true statement, support (correct)
                None: if support/non-support cannot be determined
        """
        # Convert ground truth to boolean
        if isinstance(ground_truth, bool) or isinstance(ground_truth, int):
            is_true = ground_truth
        elif isinstance(ground_truth, float):
            is_true = int(ground_truth)
        else:
            is_true = ground_truth.lower() == 'true'
        
        # Get support/non-support from response
        support_indicator = self._match_truthfulness(response)
        
        if support_indicator is None:
            return None
            
        # Encode the four possible combinations
        if is_true:
            return 3 if support_indicator else 2  # true statement: 3 for support, 2 for non-support
        else:
            return 1 if support_indicator else 0  # false statement: 1 for support, 0 for non-support
        

    def _augment_token_length(self, df: pd.DataFrame, sample_type: str) -> Tuple[pd.DataFrame, List]:
        """Augment token length data with sophisticated sampling strategies."""
        # Filter and prepare data
        df = df[df['label'] > 5].copy()
        df['label'] = df['label'].astype(int)
        
        # Configure augmentation parameters
        num_bins = 30
        base_samples = 3 # 2 #5
        max_samples = 5 # 4 #20
        bins = np.linspace(df['label'].min(), min(df['label'].max(), self.MAX_TOKEN_LENGTH), num_bins)
        
        # Calculate weights and initial bin counts
        df['bin'] = pd.cut(df['label'], bins, labels=False)
        bin_counts = df['bin'].value_counts().sort_index()
        weights = (bin_counts.cumsum() / bin_counts.cumsum().max())
        df['weight'] = df['bin'].map(lambda x: weights.get(x, 1.0))
        
        # Track bin counts during sampling for dynamic distribution balancing
        running_bin_counts = bin_counts.copy()
        
        # Generate augmented labels
        df['augmented_labels'] = df.apply(
            lambda row: self._sample_values(
                row['label'],
                row['weight'],
                bins,
                running_bin_counts,  # Pass running counts to track distribution
                base_samples,
                max_samples,
                sample_type
            ),
            axis=1
        )
        
        # Generate truncated responses
        df['truncated_responses'] = df.apply(
            lambda x: self._truncate_response(x['response'], x['label'], x['augmented_labels']), 
            axis=1
        ).tolist()
        
        return df.drop(columns=['bin', 'weight'])

    def _augment_cot_reasoning(self, df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
            """
            Augment chain-of-thought reasoning data.
            For a response with n steps:
            - Label n starts with empty string (all steps remain)
            - Label n-1 contains through step 1 (n-1 steps remain)
            - Label 1 contains through step n-1 (1 step remains)
            """
            def get_step_ends(response: str) -> List[int]:
                """Find positions after each step's content."""
                steps = list(re.finditer(r"step\s*\d+\s*:", response.lower()))
                if not steps:
                    return []
                    
                positions = []
                for i in range(len(steps)-1):
                    end = steps[i+1].start()
                    positions.append(end)
                    
                return positions
            
            def process_response(response: str, total_steps: int) -> Tuple[List[int], List[str]]:
                """Process single response into augmented labels and truncated responses."""
                step_ends = get_step_ends(response)
                if not step_ends:
                    return [total_steps], [""]
                    
                # Start with empty string for highest label
                truncated = [""]  # All steps remain
                labels = [total_steps]
                
                # Add truncations after each step (except last)
                for i, end in enumerate(step_ends):
                    remaining_steps = total_steps - (i + 1)
                    if remaining_steps >= 1:  # Don't include label 0
                        trunc = response[:end].strip()
                        truncated.append(trunc)
                        labels.append(remaining_steps)
                        
                return labels, truncated
            
            # Process each response
            results = df.apply(
                lambda row: process_response(row['response'], int(row['label'])), 
                axis=1
            )
            df['augmented_labels'] = [r[0] for r in results]
            df['truncated_responses'] = [r[1] for r in results]
            
            return df

    def _augment_multiple_choice(self, df: pd.DataFrame, sample_type: str) -> pd.DataFrame:
        """
        Augment multiple choice data by creating variations:
        1. First at punctuation marks
        2. Then at blank spaces
        3. Finally at random positions if needed
        Aims to generate > 10k samples after augmentation
        """
        # Balance original data first
        # print("Label distribution:", {label: count for label, count in df['label'].value_counts().items()})
        min_label_size = df['label'].value_counts().min()
        balanced_df = pd.concat([
            group.sample(n=min_label_size, replace=True)
            for _, group in df.groupby('label')
        ]).sample(frac=1).reset_index(drop=True)
        
        # Calculate samples needed per row to reach >10k total
        target_size = 10000
        if sample_type == 'balanced':
            sample_num = target_size // len(balanced_df)
        elif sample_type == 'unfolding':
            sample_num = 8
        
        expanded_data = []
        
        def get_truncation_positions(response: str, n_samples: int) -> List[int]:
            if not response.strip():
                return [0] * n_samples
                
            positions = []
            
            # 1. First get punctuation positions
            punct_positions = [m.start() for m in re.finditer(r'[.,!?;:]', response)]
            if punct_positions:
                positions.extend(punct_positions)
                
            # 2. Then get space positions
            space_positions = [i for i, char in enumerate(response) if char == ' ']
            positions.extend(space_positions)
            
            # Remove duplicates and sort
            positions = sorted(set(positions))
            
            # 3. If we still need more positions, add random ones
            if len(positions) < n_samples:
                remaining_samples = n_samples - len(positions)
                text_len = len(response)
                possible_positions = list(set(range(1, text_len)) - set(positions))
                
                if possible_positions:
                    random_positions = np.random.choice(
                        possible_positions,
                        size=remaining_samples,
                        replace=True
                    )
                    positions.extend(random_positions)
                else:
                    # If no other positions available, duplicate existing ones
                    additional_positions = np.random.choice(
                        positions or [0],
                        size=remaining_samples,
                        replace=True
                    )
                    positions.extend(additional_positions)
            
            # Sort all positions
            positions = sorted(set(positions))
            
            # Sample if we have too many positions
            if len(positions) > n_samples:
                # sample n_samples data from the set
                positions = np.random.choice(positions, size=n_samples, replace=False)
            return positions
        
        # Generate variations for each row
        for _, row in balanced_df.iterrows():
            response = row['response']
            label = row['label']
            
            # Get truncation positions
            trunc_positions = get_truncation_positions(response, sample_num)
            
            # sort if unfolding
            if sample_type == 'unfolding':
                trunc_positions = sorted(trunc_positions)
            
            # Create truncated responses starting with empty string
            truncated_responses = [""] + [response[:pos].strip() for pos in trunc_positions]
            augmented_labels = [label] * len(truncated_responses)
            
            # Add row with all its truncations
            expanded_data.append({
                'prompt': row['prompt'] if 'prompt' in row else None,
                'response': response,
                'label': label,
                'augmented_labels': augmented_labels,
                'truncated_responses': truncated_responses
            })
        
        # Create final balanced dataset
        final_df = pd.DataFrame(expanded_data)
        final_df = final_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return final_df

    def _augment_story_continuation(self, df: pd.DataFrame, sample_type) -> pd.DataFrame:
        """
        Process and augment story continuation data:
        1. Label stories based on first animal appearance
        2. Select top 4 most common animals and relabel as 0-3
        3. Create truncated versions before animal appearances
        4. Scale augmentation to reach ~10k balanced samples
        """
        # Common animals for detection
        common_animals = [
            'cat', 'dog', 'rabbit', 'bird', 'squirrel', 'mouse', 'rat',
            'hamster', 'fish', 'parrot', 'owl', 'raccoon', 'fox', 'bear',
            'deer', 'horse', 'pig', 'monkey', 'elephant', 'lion', 'tiger',
            'wolf', 'dolphin', 'whale', 'octopus', 'penguin', 'duck',
            'chicken', 'turtle', 'snake', 'butterfly', 'spider', 'bee'
        ]

        label_mappings = {
            "llama-3-8B-Instruct": {
                "0": "rabbit",
                "1": "squirrel",
                "2": "cat",
                "3": "owl"
            },
            "llama-2-7B-chat": {
                "0": "cat",
                "1": "squirrel",
                "2": "rabbit",
                "3": "owl"
            },
            "Mistral-7B-Instruct": {
                "0": "owl",
                "1": "fox",
                "2": "squirrel",
                "3": "monkey"
            },
            "Qwen2-7B-Instruct": {
                "0": "rabbit",
                "1": "squirrel",
                "2": "fox",
                "3": "raccoon"
            },
            "llama-3-8B": {
                "0": "bird",
                "1": "butterfly",
                "2": "deer",
                "3": "cat"
            },
            "llama-2-7B": {
                "0": "squirrel",
                "1": "fox",
                "2": "bird",
                "3": "mouse"
            },
            "Mistral-7B": {
                "0": "bird",
                "1": "rabbit",
                "2": "fox",
                "3": "squirrel"
            },
            "Qwen2-7B": {
                "0": "deer",
                "1": "duck",
                "2": "bird",
                "3": "raccoon"
            }
        }
        
        if 'ROCStories' in self.dataset_name:
            # based on model name, only keep the four animals in the mapping, and relabel them
            model_name = self.model_name
            if model_name in label_mappings:
                label_mapping = label_mappings[model_name]
                top_4_labels = [int(label) for label in label_mapping.keys()]
                # map back the top_4_labels to the original label
                top_4_labels = [common_animals.index(label_mapping[str(label)]) for label in top_4_labels]
                label_mapping = {label: idx for idx, label in enumerate(top_4_labels)}
                print("Label mapping:", {common_animals[old_label]: new_label
                                        for old_label, new_label in label_mapping.items()})

        else:
            # Get top 4 animal classes
            label_counts = df['label'].astype(int).value_counts()
            top_4_labels = label_counts.nlargest(4).index.tolist()
            # Print mapping for verification
            label_mapping = {label: idx for idx, label in enumerate(top_4_labels)}
            print("Label mapping:", {common_animals[old_label]: new_label 
                                for old_label, new_label in label_mapping.items()})
        
        # Filter for top 4 animals; perform class balance
        balanced_dfs = []
        filtered_df = df[df['label'].isin(top_4_labels)].copy()
        class_counts = filtered_df['label'].value_counts()
        min_class_size = min(class_counts.min(), 5000//4)
        for label in top_4_labels:
            class_data = filtered_df[filtered_df['label'] == label]
            # Randomly sample min_class_size samples from this class
            balanced_class = class_data.sample(n=min_class_size, random_state=self.seed)
            balanced_dfs.append(balanced_class)

        # Combine all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        # if filtered_df length > 5000: truncate to 5001
        # if len(balanced_df) > 5000:
        #     balanced_df = balanced_df.sample(n=5000, random_state=self.seed)

        expanded_data = []
    
        # sample: each data, sample equal times, so that the total num >=10k
        if 'unfolding' in sample_type:
            unfolding_step_num = 8
            for _, row in balanced_df.iterrows():
                # Convert label to integer and get corresponding animal
                original_label = int(row['label'])
                animal = common_animals[original_label]
                
                # Find animal position in response
                response = row['response']
                pos = response.lower().find(animal)
                if pos < 0:
                    pos = response.lower().find(f"{animal}s")
                    
                if pos < 0:
                    continue

                # Find all punctuation and space positions before the animal
                punct_space_positions = [
                    i+1 for i, char in enumerate(response[1:pos]) 
                    if char in ['.', '!', '?', ',', ';', ' ']
                ]
                
                
                # If we have fewer than 10 positions, add random positions
                if len(punct_space_positions) < unfolding_step_num:
                    possible_positions = list(set(range(1, pos)) - set(punct_space_positions))
                    if possible_positions:
                        needed_positions = unfolding_step_num - len(punct_space_positions)
                        random_positions = np.random.choice(
                            possible_positions,
                            size=min(needed_positions, len(possible_positions)),
                            replace=False
                        )
                        punct_space_positions.extend(random_positions)
                
                # Sort positions and uniformly sample 10
                if len(punct_space_positions) > unfolding_step_num:
                    punct_space_positions = np.random.choice(punct_space_positions, size=unfolding_step_num, replace=False)
                trunc_pos = sorted(punct_space_positions)
                
                
                
                # Create truncated responses starting with empty string
                truncated_responses = [""] + [response[:p].strip() for p in trunc_pos]
                augmented_labels = [label_mapping[original_label]] * len(truncated_responses)
                
                # Add row with all its truncations
                expanded_data.append({
                    'prompt': row['prompt'],
                    'info': animal,
                    'response': row['response'],
                    'label': augmented_labels[0],
                    'augmented_labels': augmented_labels,
                    'truncated_responses': truncated_responses
                })
        else:
            sample_num = 9999 // len(balanced_df)
            for _, row in balanced_df.iterrows():
                # Convert label to integer and get corresponding animal
                original_label = int(row['label'])
                animal = common_animals[original_label]
                
                # Find animal position in response
                response = row['response']
                pos = response.lower().find(animal)
                if pos < 0:
                    pos = response.lower().find(f"{animal}s")
                    
                if pos < 0:
                    continue

                # Get truncation points (randomly select before animal position; repeat if sample_num < the num pos allow)
                if pos <= 0:
                    trunc_pos = [0] * sample_num
                else:
                    space_positions = [i for i, char in enumerate(response[:pos]) if char == ' ']
                    if len(space_positions) >= sample_num:
                        # If we have enough spaces, randomly sample from them
                        trunc_pos = np.random.choice(space_positions, size=sample_num, replace=False)
                    else:
                        # If we don't have enough spaces, use all space positions
                        # and fill the rest with random positions
                        remaining_samples = sample_num - len(space_positions)
                        trunc_pos = list(space_positions)
                        
                        # For the remaining samples, get random positions
                        # excluding existing space positions
                        possible_positions = list(set(range(1, pos)) - set(space_positions))
                        if possible_positions:
                            random_positions = np.random.choice(
                                possible_positions, 
                                size=remaining_samples, 
                                replace=True
                            )
                            trunc_pos.extend(random_positions)
                        # still no places: reuse the space_positions
                        else:
                            random_positions = np.random.choice(
                                list(range(0, pos)),
                                size=remaining_samples, 
                                replace=True
                            )
                            trunc_pos.extend(random_positions)
                    

                truncated_responses = [""] + [response[:p].strip() for p in trunc_pos]
                if "ROC_STORIES" in self.dataset_name:
                    augmented_labels = [original_label] * len(truncated_responses)
                else:
                    augmented_labels = [label_mapping[original_label]] * len(truncated_responses)

                    
                # Add row with all its truncations; info: animal chosen
                expanded_data.append({
                    'prompt': row['prompt'],
                    'info': animal,
                    'response': row['response'],
                    'label': augmented_labels[0],
                    'augmented_labels': augmented_labels,
                    'truncated_responses': truncated_responses
                })

        # Create final balanced dataset
        final_df = pd.DataFrame(expanded_data)
        final_df = final_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return final_df

    def _augment_multiple_choice_selection(self, df: pd.DataFrame, sample_type) -> pd.DataFrame:
        """
        Process and augment multiple choice selection data:
        1. Balance classes for options A-E (or A-D if only 4 options present)
        2. Create truncated versions before answer appearances
        3. Scale augmentation to reach balanced samples
        """
        # Get value counts and determine if we have 4 or 5 options
        label_counts = df['label'].value_counts()
        num_options = min(5, len(label_counts))  # Use 4 or 5 based on data
        
        # select top three classes
        top_labels = label_counts.nlargest(num_options).index.tolist()[:3]
        # output raw label distribution
        print("Label distribution:", {label: count for label, count in label_counts.items()})
        
        # filter to only keep the top three classes
        filtered_df = df[df['label'].isin(top_labels)].copy()
        
        # Filter for valid labels and balance classes
        balanced_dfs = []
        # filtered_df = df[filtered_df['label'] < num_options].copy()  # Keep only valid labels (0 to 3/4)
        min_class_size = min(filtered_df['label'].value_counts().min(), 10000) #5000//num_options)
        
        for label in range(num_options):
            class_data = filtered_df[filtered_df['label'] == label]
            if len(class_data) > 0:
                # Randomly sample min_class_size samples from this class
                balanced_class = class_data.sample(n=min_class_size, random_state=self.seed)
                balanced_dfs.append(balanced_class)

        # Combine all balanced classes
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        # balanced_df['augmented_labels'] = balanced_df['label'].apply(lambda x: [x])
        # return balanced_df
        
        # If balanced_df length > 5000: truncate to 5000
        # if len(balanced_df) > 5000:
        #     balanced_df = balanced_df.sample(n=5000, random_state=self.seed)

        expanded_data = []
        
        # Sample: each data point, sample equal times to reach desired total
        sample_num = 9999 // len(balanced_df)# if sample_type == 'dynamic_uniform' else 0
        
        # Options to look for in the text
        options = ['A', 'B', 'C', 'D', 'E'][:num_options]
        
        for _, row in balanced_df.iterrows():
            response = row['response']
            label = int(row['label'])
            option = options[label]
            
            response_upper = response.upper()
            
            positions = self._match_choice(response_upper, target_choice=option)
                
            # Use earliest position where answer appears
            pos = min(positions)
            
            # Get truncation points (first try spaces before answer position)
            if pos <= 0:
                trunc_pos = [0] * sample_num
            else:
                # First find all spaces before the answer
                space_positions = [i for i, char in enumerate(response[:pos]) if char == ' ']
                
                if len(space_positions) >= sample_num:
                    # If we have enough spaces, randomly sample from them
                    trunc_pos = np.random.choice(space_positions, size=sample_num, replace=False)
                else:
                    # If not enough spaces, use all spaces and fill rest with random positions
                    remaining_samples = sample_num - len(space_positions)
                    trunc_pos = list(space_positions)
                    
                    # Get random positions excluding existing space positions
                    possible_positions = list(set(range(1, pos)) - set(space_positions))
                    if possible_positions:
                        random_positions = np.random.choice(
                            possible_positions,
                            size=remaining_samples,
                            replace=True
                        )
                        trunc_pos.extend(random_positions)
                    else:
                        # If still no places, use random positions
                        random_positions = np.random.choice(
                            list(range(0, pos)),
                            size=remaining_samples,
                            replace=True
                        )
                        trunc_pos.extend(random_positions)

            # Create truncated responses starting with empty string
            truncated_responses = [""] + [response[:p].strip() for p in trunc_pos]
            augmented_labels = [label] * len(truncated_responses)
            
            # Add row with all its truncations
            expanded_data.append({
                'prompt': row['prompt'],
                'response': row['response'],
                'label': label,
                'augmented_labels': augmented_labels,
                'truncated_responses': truncated_responses
            })

        # Create final balanced dataset
        final_df = pd.DataFrame(expanded_data)
        # perform re-label: as we only choose 3 top labels from the dataset, we need to re-label them to 0, 1, 2
        final_df['label'] = final_df['label'].map({label: idx for idx, label in enumerate(top_labels)})
        
        
        final_df = final_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        return final_df
    
    def _augment_truthfulness(self, df: pd.DataFrame, sample_type: str=None) -> pd.DataFrame:
        """
        Raw labels:
        int: Label encoding as follows:
                0: false statement, non-support (correct)
                1: false statement, support (incorrect)
                2: true statement, non-support (incorrect)
                3: true statement, support (correct)
                None: if support/non-support cannot be determined
        Process and augment truthfulness recognition data:
        1. Relabel classes into binary truthful/non-truthful
        2. Balance classes between truthful and non-truthful
        3. Create truncated versions at punctuation and spaces
        4. Scale augmentation to reach balanced samples
        """
        # Relabel into binary classes (0: non-truthful, 1: truthful)
        print("Label distribution:", {label: count for label, count in df['label'].value_counts().items()})
        df['label'] = df['label'].map({0.0: 0, 1.0: 1, 2.0: 1, 3.0: 0})
        
        # Print initial distribution after relabeling
        print("Label distribution after relabeling:", {label: count for label, count in df['label'].value_counts().items()})
        
        # Balance classes
        min_label_size = min(df['label'].value_counts().min(), 4000)
        balanced_df = pd.concat([
            group.sample(n=min_label_size, replace=True)
            for _, group in df.groupby('label')
        ]).sample(frac=1).reset_index(drop=True)
        
        # Calculate samples needed per row to reach target size
        target_size = 10000
        sample_num = target_size // len(balanced_df) if len(balanced_df) <= 9000 else 0
        
        expanded_data = []
        
        for _, row in balanced_df.iterrows():
            response = row['response']
            label = int(row['label'])
            
            # Find all punctuation and space positions
            punct_space_positions = [i for i, char in enumerate(response) 
                                if char in ['.', '!', '?', ',', ';', ' ']]
            
            # Always include the full response
            if len(punct_space_positions) >= sample_num:
                # If we have enough positions, randomly sample from them
                trunc_pos = np.random.choice(punct_space_positions, size=sample_num, replace=False)
            else:
                # If not enough positions, use all positions and repeat some randomly
                remaining_samples = sample_num - len(punct_space_positions)
                trunc_pos = punct_space_positions + list(np.random.choice(
                    punct_space_positions,
                    size=remaining_samples,
                    replace=True
                ))
            
            # Create truncated responses (including full response)
            truncated_responses = [''] + [response[:p].strip() for p in trunc_pos]
            augmented_labels = [label] * len(truncated_responses)
            
            # Add row with all its truncations
            expanded_data.append({
                'prompt': row['prompt'] if 'prompt' in row else None,
                'response': response,
                'label': label,
                'augmented_labels': augmented_labels,
                'truncated_responses': truncated_responses
            })
        
        # Create final balanced dataset
        final_df = pd.DataFrame(expanded_data)
        final_df = final_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return final_df


    def _sample_values(self, label: int, weight: float, bins: np.ndarray, 
                    bin_counts: pd.Series, base_samples: int, max_samples: int, 
                    sample_type: str, threshold=1000) -> List[int]:
        """
        Sample values based on specified strategy, considering current distribution.
        Updates bin_counts in place to maintain distribution tracking.
        """
        if sample_type == 'balanced':
            # Calculate number of samples based on weight
            n_samples = int(base_samples + (max_samples - base_samples) * weight**2)
            
            # Find valid bins (bins with upper bound less than label)
            valid_bins = np.where(bins[:-1] < label)[0]
            # make sure do not exceed bin_counts
            valid_bins = valid_bins[:len(bin_counts)]
            if len(valid_bins) > 0:
                # Get counts for valid bins
                valid_bin_counts = bin_counts.iloc[valid_bins]
                # Find bin with minimum count
                min_idx = valid_bin_counts.values.argmin()  # Get position of minimum value
                target_bin = valid_bins[min_idx]  # Map back to original bin index
                
                # Define target range based on bins
                bin_start = int(bins[target_bin])
                bin_end = int(min(bins[target_bin + 1], label))
            else:
                # If no valid bins, use range from 0 to label
                bin_start = 0
                bin_end = int(label)
                target_bin = 0
            
            # Adjust n_samples based on available range
            n_samples = min(n_samples, bin_end - bin_start)
            if n_samples <= 0:
                return [label] if label < self.MAX_TOKEN_LENGTH else []
                
            # Sample values and update bin counts
            samples = np.random.choice(
                range(bin_start, bin_end), 
                n_samples, 
                replace=False
            )
            bin_counts.iloc[target_bin] += n_samples
            
            # Include original label if within MAX_TOKEN_LENGTH
            if label < self.MAX_TOKEN_LENGTH and label not in samples:
                samples = np.append(samples, label)
                
            return list(samples)
        
        elif sample_type == 'linear':
            n_samples = min(label, 10) #base_samples)
            return list(range(label - n_samples + 1, label + 1)) if label <= threshold else []
        
        return [label] if label <= threshold else []
        
    def _truncate_response(self, response: str, label: int, augmented_labels: List[int]) -> List[str]:
        """
        Truncate response tokens according to augmented labels.
        
        Args:
            response: Original response text
            label: Original token length
            augmented_labels: List of target token lengths
        
        Returns:
            List of truncated response texts for each augmented label
        """
        response_tokens = self.tokenizer(response, return_tensors='pt').input_ids[0]
        
        truncated_responses = []
        for aug_label in augmented_labels:
            # Calculate how many tokens to keep (original length minus augmented label)
            truncate_to = int(label) - int(aug_label)
            # Truncate tokens and decode back to text
            truncated_tokens = response_tokens[:truncate_to]
            truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            truncated_responses.append(truncated_text)
            
        return truncated_responses

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='wiki_medical_terms')
    parser.add_argument('--prompt_template_type', type=str, default='token_length')
    parser.add_argument('--model_name', type=str, default='llama-2-7B-chat')
    parser.add_argument('--in_data_prefix', type=str, default='')
    parser.add_argument('--out_data_prefix', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--augmentation_type', type=str, default=None, 
                       choices=["", 'balanced', 'unfolding', 'unfolding2', 'raw'])
    args = parser.parse_args()

    # Load configuration and initialize components
    with open('config.yaml') as f:
        global_config = yaml.load(f, Loader=yaml.FullLoader)
    template_config = global_config['template_types'][args.prompt_template_type]
    args.prompt_template = template_config['template']
    args.model_path = global_config['model_path'][args.model_name]
    
    dataset_manager = DatasetManager(args, "labeling_and_augmentation")
    model_wrapper = ModelWrapper(args.model_path, device='auto', load_model=False, use_sampling=False)
    
    # Process dataset
    processor = DataProcessor(global_config, args.model_name, model_wrapper.tokenizer, args.prompt_template_type, args.seed, args.dataset_name)
    dataset = processor.process_dataset(
        dataset_manager.dataset, 
        args.augmentation_type
    )
    
    dataset_manager.save_results(dataset)

if __name__ == '__main__':
    main()