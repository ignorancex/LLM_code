import os
import gc
import re
import pandas as pd
import torch
import torch.nn as nn
from typing import Literal
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from sentence_transformers import SentenceTransformer

from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel
from loguru import logger

# Set environment variables for better performance
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "10"
torch.set_num_threads(1)

from sentence_transformers.util import (
    batch_to_device,
    truncate_embeddings,
)
from fense.download_utils import RemoteFileMetadata, check_download_resource


class BERTFlatClassifier(nn.Module):
    """
    BERT-based flat classifier for error detection in generated text.

    This classifier is used to detect errors in generated captions and apply
    penalty scores accordingly.

    Args:
        model_type (str): The BERT model type to use as backbone
        num_classes (int): Number of output classes (default: 5)
    """

    def __init__(self, model_type, num_classes=5) -> None:
        super().__init__()
        self.model_type = model_type
        self.num_classes = num_classes

        # Load pre-trained BERT model
        self.encoder = AutoModel.from_pretrained(model_type)
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)

        # Classification head
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        """
        Forward pass of the classifier.

        Args:
            input_ids: Token IDs for input text
            attention_mask: Attention mask for input text
            token_type_ids: Token type IDs for input text

        Returns:
            logits: Classification logits
        """
        # Get BERT outputs
        outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        # Use [CLS] token representation
        x = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(x)

        # Generate classification logits
        logits = self.clf(x)
        return logits


class RefinedErrorChecker(nn.Module):
    """
    Refined error checker for detecting and penalizing errors in generated text.

    This module loads a pre-trained error detection model and applies penalties
    to generated captions that contain errors. It's used to improve the quality
    of evaluation by considering the correctness of generated text.

    Args:
        model_name_or_path (str): Name or path of the pre-trained error checker model
        device (str): Device to run the model on ('cuda' or 'cpu')
        error_threshold (float): Threshold for error detection (default: 0.9)
        penalty (float): Penalty factor for detected errors (default: 0.9)
        use_proxy (bool): Whether to use proxy for downloading models
        proxies (str): Proxy configuration
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cuda", "cpu"] = None,
        error_threshold: float = 0.9,
        penalty: float = 0.9,
        use_proxy: bool = False,
        proxies: str = None,
    ):
        super().__init__()
        # Disable tokenizer parallelism to avoid warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.error_threshold = error_threshold
        self.penalty = penalty

        # Load pre-trained error checker model from FENSE
        self.model = self.load_pretrain_echecker(model_name_or_path, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model.model_type)

    def load_pretrain_echecker(
        self,
        model_name_or_path: str,
        device: Literal["cuda", "cpu"] = None,
        use_proxy: bool = False,
        proxies: str = None,
    ):
        """
        Load pre-trained error checker model from remote repository.

        Args:
            model_name_or_path (str): Model identifier
            device (str): Target device for the model
            use_proxy (bool): Whether to use proxy for downloading
            proxies (str): Proxy configuration

        Returns:
            torch.nn.Module: Loaded error checker model
        """
        # Available pre-trained error checker models
        PRETRAIN_ECHECKERS = {
            "echecker_clotho_audiocaps_base": (
                "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_base.ckpt",
                "1a719f090af70614bbdb9f9437530b7e133c48cfa4a58d964de0d47fc974a2fa",
            ),
            "echecker_clotho_audiocaps_tiny": (
                "https://github.com/blmoistawinde/fense/releases/download/V0.1/echecker_clotho_audiocaps_tiny.ckpt",
                "90ed0ac5033ec497ec66d4f68588053813e085671136dae312097c96c504f673",
            ),
            "none": (None, None),
        }

        # Download model if needed
        url, checksum = PRETRAIN_ECHECKERS[model_name_or_path]
        remote = RemoteFileMetadata(filename=f"{model_name_or_path}.ckpt", url=url, checksum=checksum)
        file_path = check_download_resource(remote, use_proxy, proxies)

        # Load model state and create classifier
        model_states = torch.load(file_path, weights_only=True)
        clf = BERTFlatClassifier(model_type=model_states["model_type"], num_classes=model_states["num_classes"])

        # Load pre-trained weights
        dict_new = clf.state_dict().copy()
        trained_list = [i for i in model_states["state_dict"].keys() if "encoder.embeddings.position_ids" not in i]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = model_states["state_dict"][trained_list[i]]

        clf.load_state_dict(dict_new)
        clf.eval()
        clf.to(device)
        return clf

    def text_preprocess(self, inp):
        """
        Preprocess input text by removing punctuation and converting to lowercase.

        Args:
            inp (str or list): Input text(s) to preprocess

        Returns:
            str or list: Preprocessed text(s)
        """
        if type(inp) == str:
            return re.sub(r"[^\w\s]", "", inp).lower()
        else:
            return [re.sub(r"[^\w\s]", "", x).lower() for x in inp]

    def infer_preprocess(self, tokenizer, texts, max_len):
        """
        Preprocess texts for inference using the tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer
            texts (list): List of texts to tokenize
            max_len (int): Maximum sequence length

        Returns:
            dict: Tokenized batch ready for model input
        """
        texts = self.text_preprocess(texts)
        batch = tokenizer(texts, truncation=True, padding="max_length", max_length=max_len)

        # Convert to tensors
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            batch[k] = torch.LongTensor(batch[k])
        return batch

    def forward(
        self,
        sentences: str | list,
        batch_size: int = 32,
    ):
        """
        Detect errors in input sentences and apply penalties.

        Args:
            sentences (str or list): Input sentence(s) to check for errors
            batch_size (int): Batch size for processing multiple sentences

        Returns:
            torch.Tensor: Penalty scores (1.0 for no error, penalty factor for errors)
        """
        if type(sentences) == str:
            sentences = [sentences]

        if len(sentences) == 1:
            # Process single sentence
            batch = self.infer_preprocess(self.tokenizer, sentences, max_len=64)
            for k, v in batch.items():
                batch[k] = v.to(self.device)

            with torch.no_grad():
                logits = self.model(**batch)
                probs = torch.sigmoid(logits).detach().cpu().numpy()

            # Check if error probability exceeds threshold
            has_error = probs[0][-1] > self.error_threshold
            output = (1 - self.penalty) if has_error else 1
            output = torch.tensor([output])
        else:
            # Process multiple sentences in batches
            probs = []
            for i in trange(0, len(sentences), batch_size):
                batch = self.infer_preprocess(self.tokenizer, sentences[i : i + batch_size], max_len=256)
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                with torch.no_grad():
                    batch_logits = self.model(**batch)
                    batch_probs = torch.sigmoid(batch_logits)[:, -1]
                probs.append(batch_probs)

            # Combine all probabilities and apply penalties
            probs = torch.cat(probs)
            has_error = probs > self.error_threshold
            output = has_error * (1 - self.penalty)
            output[output == 0] = 1

        return output


class RefinedSentenceTransformers(nn.Module):
    """
    Enhanced sentence transformer wrapper for generating various types of embeddings.

    This class provides a unified interface for generating different types of embeddings
    from text input, including word embeddings, token embeddings, and sentence embeddings.
    It serves as the core text encoding component for the DATE evaluation system.

    Args:
        model_name_or_path (str): Name or path of the sentence transformer model
        device (str): Device to run the model on ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: Literal["cuda", "cpu"] = None,
    ):
        super().__init__()
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # CPU priority strategy: keep data on CPU by default, only move to GPU when needed
        self.compute_device = self.device  # Device for actual computation
        self.storage_device = "cpu"  # Device for data storage (always CPU)

        # Load the sentence transformer model
        self.sbert = SentenceTransformer(model_name_or_path, device=self.device)

        # Extract the underlying BERT model for direct access
        for name, module in self.sbert.named_children():
            self.auto_model = module.auto_model
            break

    def encode_features(
        self,
        features: dict,
        output_value: Literal[
            "input_ids", "word_embeddings", "token_embeddings", "sentence_embedding"
        ] = "sentence_embedding",
    ):
        """
        Generate embeddings from input features.

        This method can generate different types of embeddings based on the output_value parameter:
        - input_ids: Return the input token IDs (no embedding computation)
        - word_embeddings: Return word-level embeddings from the embedding layer
        - token_embeddings: Return token-level embeddings from the transformer
        - sentence_embedding: Return sentence-level embeddings (default)

        Args:
            features (dict): Input features containing input_ids, attention_mask, etc.
            output_value (str): Type of embedding to generate

        Returns:
            dict: Dictionary containing the requested embeddings and metadata
        """
        # Move features to the target device
        features = batch_to_device(features, self.device)

        if output_value != "input_ids":
            with torch.no_grad():
                if output_value == "word_embeddings":
                    # Generate word embeddings directly from the embedding layer
                    embeddings = self.auto_model.embeddings.word_embeddings(features["input_ids"])
                else:
                    # Generate embeddings using the sentence transformer
                    out_features = self.sbert.forward(features)

                    # Apply truncation if specified
                    out_features["sentence_embedding"] = truncate_embeddings(
                        out_features["sentence_embedding"], self.sbert.truncate_dim
                    )

                    if output_value == "token_embeddings":
                        # Keep the original shape [n_sents, seq_len, emb_dim] for compatibility
                        # with DATE_detailed.py which expects this format
                        embeddings = out_features[output_value]
                        embeddings = embeddings.to(torch.float16)
                    else:
                        # For sentence embeddings, use the output directly
                        embeddings = out_features[output_value]
        else:
            # Return input IDs without processing
            embeddings = features["input_ids"]

        # Prepare output features dictionary
        output_features = {"embeddings": embeddings, "features": features}
        # Always move output to CPU for storage
        output_features = batch_to_device(output_features, self.storage_device)
        return output_features

    def encode_sentences(
        self,
        sentences: str | list,
        batch_size: int = 32,
        output_value: Literal[
            "input_ids", "word_embeddings", "token_embeddings", "sentence_embedding"
        ] = "sentece_embedding",
    ):
        """
        Encode sentences into embeddings with batched processing.

        This method tokenizes input sentences and generates embeddings using the
        specified embedding type. It supports both single sentences and batches.

        Args:
            sentences (str or list): Input sentence(s) to encode
            batch_size (int): Batch size for processing multiple sentences
            output_value (str): Type of embedding to generate

        Returns:
            dict: Dictionary containing embeddings and tokenization features
        """
        # Ensure sentences is a list
        if isinstance(sentences, str):
            sentences = [sentences]

        # If there are fewer sentences than batch_size, process all at once
        if len(sentences) <= batch_size:
            features = self.sbert.tokenize(sentences)
            return self.encode_features(features, output_value=output_value)
        
        # Process sentences in batches
        all_embeddings = []
        all_features_list = []
        
        # First pass: find maximum sequence length across all batches
        max_length = 0
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i + batch_size]
            batch_features = self.sbert.tokenize(batch_sentences)
            max_length = max(max_length, batch_features["input_ids"].shape[1])
        
        # Second pass: process all batches with consistent padding
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i:i + batch_size]
            batch_features = self.sbert.tokenize(batch_sentences)
            
            # Pad to max_length if needed
            current_length = batch_features["input_ids"].shape[1]
            if current_length < max_length:
                pad_length = max_length - current_length
                
                # Pad input_ids
                batch_features["input_ids"] = torch.cat([
                    batch_features["input_ids"],
                    torch.zeros(batch_features["input_ids"].shape[0], pad_length, dtype=torch.long, device=batch_features["input_ids"].device)
                ], dim=1)
                
                # Pad attention_mask
                batch_features["attention_mask"] = torch.cat([
                    batch_features["attention_mask"],
                    torch.zeros(batch_features["attention_mask"].shape[0], pad_length, dtype=torch.long, device=batch_features["attention_mask"].device)
                ], dim=1)
                
                # Pad token_type_ids if it exists
                if "token_type_ids" in batch_features:
                    batch_features["token_type_ids"] = torch.cat([
                        batch_features["token_type_ids"],
                        torch.zeros(batch_features["token_type_ids"].shape[0], pad_length, dtype=torch.long, device=batch_features["token_type_ids"].device)
                    ], dim=1)
            
            # Process this batch
            batch_output = self.encode_features(batch_features, output_value=output_value)
            
            # Collect results
            all_embeddings.append(batch_output["embeddings"])
            all_features_list.append(batch_output["features"])
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        # Concatenate embeddings
        concatenated_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Concatenate features - ensure all tensors have the same size
        concatenated_features = {}
        if all_features_list:
            for key in all_features_list[0].keys():
                if isinstance(all_features_list[0][key], torch.Tensor):
                    # Check if all tensors have the same size except for the first dimension
                    first_tensor = all_features_list[0][key]
                    tensor_shape = first_tensor.shape[1:]  # All dimensions except batch dimension
                    
                    # Verify all tensors have the same shape
                    for f in all_features_list[1:]:
                        if f[key].shape[1:] != tensor_shape:
                            raise ValueError(f"Tensor size mismatch for key '{key}'. Expected shape {tensor_shape}, got {f[key].shape[1:]}")
                    
                    concatenated_features[key] = torch.cat([f[key] for f in all_features_list], dim=0)
                elif isinstance(all_features_list[0][key], list):
                    concatenated_features[key] = []
                    for f in all_features_list:
                        concatenated_features[key].extend(f[key])
                else:
                    concatenated_features[key] = all_features_list[0][key]
        
        return {
            "embeddings": concatenated_embeddings,
            "features": concatenated_features
        }


class DATE(nn.Module):
    """
    DATE (Discriminability based Audio Task Evaluation) - Main evaluation class.

    This class implements the DATE metric for evaluating audio captioning and QA systems.
    DATE combines similarity and discrimination components using a harmonic mean to provide
    a comprehensive evaluation score. It supports both FENSE and DATE evaluation modes.

    Key Features:
    - Dual evaluation modes: FENSE (similarity only) and DATE (similarity + discrimination)
    - TF-IDF weighted embeddings for improved text representation
    - Error detection and penalty mechanisms
    - Batch processing for efficiency
    - Support for various audio content types

    Args:
        sbert_name_or_path (str): Name or path of the sentence transformer model
        echecker_name_or_path (str): Name or path of the error checker model
        device (str): Device to run models on ('cuda' or 'cpu')
        error_threshold (float): Threshold for error detection (default: 0.9)
        penalty (float): Penalty factor for detected errors (default: 0.9)
        use_proxy (bool): Whether to use proxy for model downloads
        proxies (str): Proxy configuration
        is_clamp_neg_similarity (bool): Whether to clamp negative similarities
        return_type (str): Evaluation mode ('fense' or 'date')
    """

    def __init__(
        self,
        sbert_name_or_path: str = "paraphrase-TinyBERT-L6-v2",
        echecker_name_or_path: str = "echecker_clotho_audiocaps_base",
        device: Literal["cuda", "cpu"] = None,
        error_threshold: float = 0.9,  # parameter of echecker model
        penalty: float = 0.9,  # parameter of echecker model
        use_proxy: bool = False,  # parameter of echecker model
        proxies: str = None,  # parameter of echecker model
        is_clamp_neg_similarity: bool = False,
        return_type: Literal["fense", "date"] = "date",
        cpu_priority: bool = True,  # New parameter for CPU-first strategy
    ):
        super().__init__()
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # CPU priority strategy: keep data on CPU by default, only move to GPU when needed
        self.cpu_priority = cpu_priority
        self.compute_device = self.device  # Device for actual computation
        self.storage_device = "cpu" if cpu_priority else self.device  # Device for data storage
        
        # Log device selection
        if torch.cuda.is_available():
            logger.info(f"CUDA available. Using device: {self.device}")
            if self.device == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                if self.cpu_priority:
                    logger.info("CPU priority mode enabled: data will be stored on CPU and moved to GPU only for computation")
        else:
            logger.info(f"CUDA not available. Using device: {self.device}")
            
        self.return_type = return_type
        self.is_clamp_neg_similarity = is_clamp_neg_similarity

        # Initialize core components
        self.model = RefinedSentenceTransformers(sbert_name_or_path, device)
        self.echecker = RefinedErrorChecker(echecker_name_or_path, device, error_threshold, penalty, use_proxy, proxies)

        # Pre-computed delta values for different field types and data variants
        # These values are optimized for different audio content categories
        self.subset_data_delta = {
            "long": {"all": 0.1703},  # Long-form captions (x)
            "short": {"all": 0.1772},  # Short-form captions (x)
            "speech": {"pure": 0.2078, "mixed": 0.2078},  # Speech content
            "music": {"pure": 0.1566, "mixed": 0.1851},  # Music content
            "sound": {"pure": 0.1988, "mixed": 0.2593},  # General sound effects
            "environment": {"all": 0.2150},  # Environmental sounds (x)
        }

    def fetch_word_embeddings(self, batch_output: dict, batch_size: int = 32):
        """
        Calculate word-level embeddings and similarity matrices for TF-IDF weighting.

        This method processes the batch output to extract word embeddings and compute
        similarity matrices that will be used for TF-IDF weight calculation. It creates
        word-to-word and word-to-document similarity matrices.

        Args:
            batch_output (dict): Output from the sentence transformer containing embeddings
            batch_size (int): Batch size for processing (currently unused)

        Returns:
            tuple: (word_similarity, word2doc_similarity, updated_batch_output)
                - word_similarity: Word-to-word similarity matrix
                - word2doc_similarity: Word-to-document similarity matrix
                - updated_batch_output: Enhanced batch output with similarity data
        """
        # Extract input IDs and embeddings from batch output
        input_ids_batch = batch_output["features"]["input_ids"]  # (n_sents, seq_len)
        embeddings_batch = batch_output["embeddings"].clone()  # (n_sents, seq_len, emb_dim)

        # Ensure embeddings_batch is on CPU for storage, input_ids_batch can stay on its device
        if embeddings_batch.device.type == "cuda":
            embeddings_batch = embeddings_batch.cpu()
        
        # For indexing, we need both tensors on the same device
        # Since embeddings_batch is now on CPU, move input_ids_batch to CPU for indexing
        if input_ids_batch.device.type == "cuda":
            input_ids_batch = input_ids_batch.cpu()

        # Construct the embedding dictionary for unique words
        valid_mask = input_ids_batch != 0  # Create mask for non-padding tokens
        valid_input_ids = input_ids_batch[valid_mask]  # Flatten valid input IDs
        valid_embeddings = embeddings_batch[valid_mask]  # Flatten corresponding embeddings

        # Create dictionary mapping word IDs to their embeddings
        embeddings_dict = {
            int(tmp_id): {"embedding": embedding} for tmp_id, embedding in zip(valid_input_ids, valid_embeddings)
        }

        logger.info("start calculate similarity")

        # Construct embeddings matrix and create mapping
        embeddings = torch.stack([emb_dict["embedding"] for emb_dict in embeddings_dict.values()])
        mapping_wordkey2vecidx = {key: idx for idx, key in enumerate(list(embeddings_dict.keys()))}

        # Calculate word-piece similarity matrix (n_words, n_words)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # In CPU priority mode, keep embeddings on CPU and only move to GPU for computation
        if self.cpu_priority and self.compute_device == "cuda":
            embeddings = embeddings.to(self.storage_device)  # Keep on CPU
        else:
            embeddings = embeddings.to(self.device)

        # calculateht word-piece simialrity (n_words, n_words)
        results = []
        with torch.no_grad():
            for i in tqdm(range(0, embeddings.shape[0], batch_size)):
                batch_block = embeddings[i : i + batch_size]
                # Move batch to compute device only when needed
                if self.cpu_priority and self.compute_device == "cuda":
                    batch_block = batch_block.to(self.compute_device)
                    embeddings_compute = embeddings.to(self.compute_device)
                else:
                    embeddings_compute = embeddings
                
                block_sim = torch.mm(embeddings_compute, batch_block.t())
                results.append(block_sim.cpu())
                
                # Clean up GPU memory immediately
                if self.cpu_priority and self.compute_device == "cuda":
                    del batch_block, embeddings_compute, block_sim
                    torch.cuda.empty_cache()
                else:
                    del batch_block, block_sim
                    
        word_similarity = torch.cat(results, dim=1)
        
        # Clean up GPU memory
        if self.compute_device == "cuda":
            torch.cuda.empty_cache()

        if self.is_clamp_neg_similarity == True:
            rescaled_word_similarity = torch.clamp(word_similarity, min=0)
        else:
            rescaled_word_similarity = word_similarity

        # calculate the word * (n_sents*n_seq) similarity (n_words, n_sents, n_seq)
        n_sents, n_padding_words, n_emb = embeddings_batch.shape
        embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=-1)
        
        # embeddings_batch should already be on CPU from earlier processing
        # Keep it on CPU for storage, only move to GPU for computation when needed
        embeddings_batch = embeddings_batch.reshape(n_sents * n_padding_words, n_emb)

        # calculate the word2doc similarity
        results = []
        with torch.no_grad():
            for i in tqdm(range(0, embeddings_batch.shape[0], batch_size)):
                batch_block = embeddings_batch[i : i + batch_size]
                # Move batch to compute device only when needed
                if self.cpu_priority and self.compute_device == "cuda":
                    batch_block = batch_block.to(self.compute_device)
                    embeddings_compute = embeddings.to(self.compute_device)
                else:
                    embeddings_compute = embeddings
                
                block_sim = torch.mm(embeddings_compute, batch_block.t())
                results.append(block_sim.cpu())
                
                # Clean up GPU memory immediately
                if self.cpu_priority and self.compute_device == "cuda":
                    del batch_block, embeddings_compute, block_sim
                    torch.cuda.empty_cache()
                else:
                    del batch_block, block_sim
                    
        word2doc_similarity = torch.cat(results, dim=1)
        
        # Clean up GPU memory
        if self.compute_device == "cuda":
            torch.cuda.empty_cache()

        if self.is_clamp_neg_similarity == True:
            rescaled_word2doc_similarity = torch.clamp(word2doc_similarity, min=0)
        else:
            rescaled_word2doc_similarity = word2doc_similarity

        # contruct the outputs
        gc.collect()
        batch_output["mapping_wordkey2vecidx"] = mapping_wordkey2vecidx
        return rescaled_word_similarity, rescaled_word2doc_similarity, batch_output

    def calcualte_tf_idf_weight(
        self,
        word_similarity: torch.Tensor,
        word2doc_similarity: torch.Tensor,
        batch_output: dict,
        idf_type: Literal["token", "embedding"] = "embedding",
    ):
        """
        Calculate TF-IDF weights for word embeddings to improve text representation.

        This method computes Term Frequency-Inverse Document Frequency (TF-IDF) weights
        for each word in the corpus. The TF component is based on word similarity within
        sentences, while the IDF component measures how unique a word is across the corpus.

        Args:
            word_similarity (torch.Tensor): Word-to-word similarity matrix
            word2doc_similarity (torch.Tensor): Word-to-document similarity matrix
            batch_output (dict): Batch output containing features and mappings
            idf_type (str): Type of IDF calculation ('token' or 'embedding')

        Returns:
            list: List of TF-IDF weight tensors for each sentence group
        """
        logger.info("calculting the idf of each word")

        # Create mask for non-special tokens (exclude [CLS]=101 and [SEP]=102)
        masked_input_ids = batch_output["features"]["input_ids"].clone()
        masked_input_ids[masked_input_ids == 101] = 0  # Remove [CLS] tokens
        masked_input_ids[masked_input_ids == 102] = 0  # Remove [SEP] tokens
        masked_input_ids[masked_input_ids != 0] = 1  # Convert to binary mask

        n_sents, n_padding_words = masked_input_ids.shape

        # Calculate IDF (Inverse Document Frequency) for each unique word
        idf = torch.zeros(
            [
                word2doc_similarity.shape[0],
            ],
            device=masked_input_ids.device,
        )
        for tmp_sent_id in tqdm(range(n_sents)):
            # Extract similarity scores for current sentence
            tmp_idf = word2doc_similarity[:, tmp_sent_id * n_padding_words : (tmp_sent_id + 1) * n_padding_words].to(
                idf.device
            )

            # Apply masking based on IDF calculation type
            if idf_type == "embedding":
                tmp_idf = tmp_idf * masked_input_ids[tmp_sent_id]
            else:
                tmp_idf = tmp_idf * masked_input_ids[tmp_sent_id]
                tmp_idf = tmp_idf * (tmp_idf == 1)

            # Accumulate document frequency for each word
            tmp_idf = torch.clamp(torch.sum(tmp_idf**2, dim=1), min=0, max=1)
            idf = idf + tmp_idf

        # Calculate final IDF values using log normalization
        idf = torch.log((n_sents + 1) / (idf + 1)) + 1

        logger.info("calculting the tf-idf weight")
        tf_idf_weights = []
        input_ids = batch_output["features"]["input_ids"].detach().cpu().numpy()
        mapping_jsonkey2sentidx = batch_output["mapping_jsonkey2sentidx"]
        mapping_wordkey2vecidx = batch_output["mapping_wordkey2vecidx"]

        # Calculate TF-IDF weights for each sentence group
        for tmp_json_key, (tmp_start, tmp_end) in tqdm(mapping_jsonkey2sentidx.items()):
            batch_tfidf_weights = []
            for input_id in input_ids[tmp_start:tmp_end]:
                # Remove padding tokens
                input_id = input_id[input_id != 0]

                # Get word indices (exclude [CLS] and [SEP] tokens)
                word_indices = [mapping_wordkey2vecidx[tmp_id] for tmp_id in input_id[1:-1]]
                selected = word_similarity[word_indices][:, word_indices].to(self.device)

                # Calculate TF (Term Frequency) based on word similarity
                tf_weights = torch.sum(selected**2, dim=1)

                # Get IDF weights for current words
                idf_weights = idf[word_indices].to(self.device)

                # Combine TF and IDF weights
                tmp_tf_idf = torch.ones([tf_weights.shape[0] + 2], device=self.device)
                tmp_tf_idf[1:-1] = tf_weights * idf_weights

                # Normalize TF-IDF weights
                tmp_tf_idf[1:-1] = tmp_tf_idf[1:-1] / tmp_tf_idf[1:-1].mean()

                # Set special token weights ([CLS] gets max weight, [SEP] gets min weight)
                tmp_tf_idf[0] = torch.max(tmp_tf_idf)
                tmp_tf_idf[-1] = torch.min(tmp_tf_idf)

                batch_tfidf_weights.append(tmp_tf_idf)
            tf_idf_weights.append(batch_tfidf_weights)
        return tf_idf_weights

    def calculate_penalty(self, batch_output: dict, batch_size: int = 32):
        """
        Calculate penalty scores for all sentences using the error checker.

        This method applies the error detection model to identify potential errors
        in generated text and assigns penalty scores accordingly.

        Args:
            batch_output (dict): Batch output containing sentence data
            batch_size (int): Batch size for error checking

        Returns:
            torch.Tensor: Penalty scores for each sentence
        """
        all_sentences = batch_output["all_sentences"]
        mapping_jsonkey2sentidx = batch_output["mapping_jsonkey2sentidx"]
        penalities = self.echecker(all_sentences.copy(), batch_size=batch_size)

        return penalities

    def update_word_embeddings(
        self,
        json_data: dict,
        subtask: str = "long",
        n_sents: int = None,
        tf_idf_weighted: bool = True,
        idf_type: Literal["token", "embedding"] = "embedding",
        batch_size: int = 32,
    ):
        """
        Generate and update word embeddings with optional TF-IDF weighting.

        This method processes input data to generate word-level embeddings and optionally
        applies TF-IDF weighting for improved representation. For FENSE mode, it skips
        the TF-IDF calculation to optimize performance.

        Args:
            json_data (dict): Input data containing text samples
            subtask (str): Field name to extract text from (default: 'long')
            n_sents (int): Number of sentences to use (None for all)
            tf_idf_weighted (bool): Whether to apply TF-IDF weighting
            idf_type (str): Type of IDF calculation ('token' or 'embedding')
            batch_size (int): Batch size for processing

        Returns:
            dict: Enhanced batch output with embeddings and metadata
        """
        all_sentences = []
        mapping_jsonkey2sentidx = {}

        # Extract all sentences from the input data
        start_idx = 0
        sorted_keys = sorted(json_data.keys())
        for tmp_key in sorted_keys:
            tmp_value = json_data[tmp_key]

            # Ensure field value is a list
            if type(tmp_value[subtask]) == str:
                tmp_value[subtask] = [tmp_value[subtask]]

            # Select sentences based on n_sents parameter
            sentences = tmp_value[subtask] if n_sents is None else [tmp_value[subtask][-1]]
            all_sentences.extend(sentences)

            # Track sentence indices for each key
            mapping_jsonkey2sentidx[tmp_key] = (start_idx, start_idx + len(sentences))
            start_idx = start_idx + len(sentences)

        # Generate word embeddings
        logger.info("calculate the word embeddings")
        batch_output = self.model.encode_sentences(all_sentences, batch_size=batch_size, output_value="word_embeddings")
        batch_output["all_sentences"] = all_sentences
        batch_output["mapping_jsonkey2sentidx"] = mapping_jsonkey2sentidx

        # For FENSE mode, skip similarity and TF-IDF calculations for efficiency
        if self.return_type == "fense" or not tf_idf_weighted:
            return batch_output

        # For DATE mode, perform full calculations including TF-IDF weighting
        # Calculate word similarity matrices
        word_sim, word2doc_sim, batch_output = self.fetch_word_embeddings(batch_output)

        # Calculate TF-IDF weights
        tf_idf_weights = self.calcualte_tf_idf_weight(word_sim, word2doc_sim, batch_output)

        # Apply TF-IDF weights to word embeddings
        embeddings_batch = batch_output["embeddings"].clone()

        for (
            tmp_tf_idf,
            tmp_key,
        ) in zip(tf_idf_weights, mapping_jsonkey2sentidx):
            tmp_start, tmp_end = mapping_jsonkey2sentidx[tmp_key]
            for tmp_sent_id in range(len(tmp_tf_idf)):
                # Apply TF-IDF weights element-wise to embeddings
                # Ensure TF-IDF weights are on the same device as embeddings
                tf_idf_weights_device = tmp_tf_idf[tmp_sent_id].to(embeddings_batch.device)
                embeddings_batch[tmp_start + tmp_sent_id][0 : len(tmp_tf_idf[tmp_sent_id]), :] *= tf_idf_weights_device.unsqueeze(1)
        
        # Move embeddings back to storage device (CPU) after processing
        if self.cpu_priority and self.compute_device == "cuda":
            embeddings_batch = embeddings_batch.to(self.storage_device)

        batch_output["features"]["inputs_embeds"] = embeddings_batch
        return batch_output

    def update_sentence_embeddings(
        self,
        batch_output,
        normalize_embeddings: bool = True,
        return_type: Literal["fense", "date"] = "date",
        batch_size: int = 32,
    ):
        """
        Generate sentence-level embeddings from word embeddings with penalty application.

        This method converts word-level embeddings to sentence-level representations
        and applies error penalties. It handles different input formats for FENSE
        and DATE evaluation modes.

        Args:
            batch_output: Batch output containing word embeddings and features
            normalize_embeddings (bool): Whether to normalize sentence embeddings
            return_type (str): Evaluation mode ('fense' or 'date')
            batch_size (int): Batch size for processing

        Returns:
            tuple: (sentence_embeddings, penalties)
                - sentence_embeddings: Normalized sentence-level embeddings
                - penalties: Error penalty scores for each sentence
        """
        # Select appropriate features based on evaluation mode
        if return_type == "fense":
            # For FENSE, exclude TF-IDF weighted embeddings
            kwargs = {k: batch_output["features"][k] for k in batch_output["features"].keys() if k != "inputs_embeds"}
        else:
            # For DATE, exclude original input_ids to use TF-IDF weighted embeddings
            kwargs = {k: batch_output["features"][k] for k in batch_output["features"].keys() if k != "input_ids"}

        # Generate sentence embeddings in batches
        embeddings = []
        with torch.no_grad():
            for i in range(0, kwargs["attention_mask"].shape[0], batch_size):
                batch_block_kwargs = {key: value[i : i + batch_size] for key, value in kwargs.items()}
                batch_output_sents_block = self.model.encode_features(
                    batch_block_kwargs, output_value="sentence_embedding"
                )
                embeddings.append(batch_output_sents_block["embeddings"])
                del batch_output_sents_block
                torch.cuda.empty_cache()
        embeddings = torch.cat(embeddings, dim=0)

        # Calculate error penalties for all sentences
        penalities = self.calculate_penalty(batch_output)

        # Reshape embeddings and penalties based on data structure
        mapping_jsonkey2sentidx = batch_output["mapping_jsonkey2sentidx"]
        for tmp_key, (tmp_start, tmp_end) in mapping_jsonkey2sentidx.items():
            n_choices_in_one_key = tmp_end - tmp_start
            break

        n_sents, n_embs = embeddings.shape
        embeddings = embeddings.reshape(n_sents // n_choices_in_one_key, n_choices_in_one_key, n_embs)
        penalities = penalities.reshape(n_sents // n_choices_in_one_key, n_choices_in_one_key)

        # Normalize embeddings if requested
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        # Move to target device - use CPU priority strategy
        if self.cpu_priority and self.compute_device == "cuda":
            embeddings = embeddings.to(self.storage_device)  # Keep on CPU
            penalities = penalities.to(self.storage_device)  # Keep on CPU
        else:
            embeddings = embeddings.to(self.device)
            penalities = penalities.to(self.device)

        return embeddings, penalities

    def forward(
        self,
        ref_data: dict,
        val_data: dict,
        group_name: str = "content_long",
        delta: int | float = None,
        save_path: str = None,
        dataframe: pd.DataFrame = None,
        field_type: str = None,
        batch_size: int = 32,
    ):
        """
        Main forward pass for DATE evaluation.

        This method performs the complete DATE evaluation process, including:
        1. Data preprocessing and alignment
        2. Embedding generation with optional TF-IDF weighting
        3. Similarity calculation
        4. Discrimination calculation (for DATE mode)
        5. Final score computation

        Args:
            ref_data (dict): Reference data containing ground truth
            val_data (dict): Validation data containing predictions
            group_name (str): Group name from GROUP_CONFIGS (default: 'content_long')
            delta (float): Delta parameter for discrimination calculation
            save_path (str): Path to save embeddings (optional)
            dataframe (pd.DataFrame): DataFrame to store detailed results
            field_type (str): Type of field being evaluated

        Returns:
            tuple: (evaluation_results, updated_dataframe)
                - evaluation_results: Dictionary containing scores
                - updated_dataframe: Enhanced dataframe with results
        """
        # Extract subtask from group_name for delta calculation
        # Map group_name back to subtask for subset_data_delta lookup
        group_to_subtask_mapping = {
            "content_long": "long",
            "content_short": "short",
            "pure_speech": "speech",
            "mixed_speech": "speech",
            "pure_music": "music",
            "mixed_music": "music",
            "pure_sound": "sound",
            "mixed_sound": "sound",
            "environment": "environment"
        }
        subtask_for_delta = group_to_subtask_mapping.get(group_name, "long")
        
        # Set default delta value
        if delta is None:
            # Get delta from subset_data_delta, which is a nested dict
            subtask_delta_dict = self.subset_data_delta.get(subtask_for_delta, {"all": 0.2})
            # Use "all" as default key, or first available key
            if "all" in subtask_delta_dict:
                delta = subtask_delta_dict["all"]
            else:
                delta = list(subtask_delta_dict.values())[0] if subtask_delta_dict else 0.2

        # Remove validation samples that don't have corresponding references
        removed_keys = []
        for tmp_key in val_data:
            if tmp_key not in ref_data:
                removed_keys.append(tmp_key)
        for tmp_key in removed_keys:
            del val_data[tmp_key]

        logger.info(f"Processing {len(ref_data)} reference items and {len(val_data)} validation items")
        logger.info(f"start processing {group_name} (subtask: {subtask_for_delta}): {len(val_data)}")

        # Determine whether to use TF-IDF weighting based on evaluation mode
        tf_idf_weighted = True if self.return_type != "fense" else False

        # Process reference data to generate embeddings
        batch_output = self.update_word_embeddings(
            ref_data, subtask_for_delta, n_sents=None, tf_idf_weighted=tf_idf_weighted, idf_type="embedding", batch_size=batch_size
        )
        embeddings_ref, penalties_ref = self.update_sentence_embeddings(batch_output, return_type=self.return_type, batch_size=batch_size)

        # Process validation data to generate embeddings
        batch_output_val = self.update_word_embeddings(
            val_data, subtask_for_delta, n_sents=1, tf_idf_weighted=tf_idf_weighted, idf_type="embedding", batch_size=batch_size
        )
        embeddings_val, penalties_val = self.update_sentence_embeddings(batch_output_val, return_type=self.return_type, batch_size=batch_size)
        embeddings_val = embeddings_val.squeeze(1)

        # Save embeddings if path is provided
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            save_name = "fense" if self.return_type == "fense" else "date"
            torch.save(embeddings_ref.detach().cpu(), f"{save_path}/embedding-{save_name}-{group_name}-ref.pt")
            torch.save(embeddings_val.detach().cpu(), f"{save_path}/embedding-{save_name}-{group_name}-val.pt")

        # Calculate similarity scores between validation and reference embeddings
        scores = []
        for tmp_idx in tqdm(range(embeddings_ref.shape[1])):
            tmp_scores = embeddings_val @ embeddings_ref[:, tmp_idx, :].T
            # Ensure penalties_val is on the same device as tmp_scores
            penalties_val_device = penalties_val.to(tmp_scores.device)
            scores.append(tmp_scores.squeeze() * penalties_val_device)
        torch.cuda.empty_cache()

        scores = torch.stack(scores)
        if scores.ndim == 3:
            scores = scores.mean(dim=0)

        # Extract diagonal similarity scores (self-similarity)
        similarity = torch.diag(scores).to(self.device)

        # Branch evaluation based on return type
        if self.return_type == "fense":
            # FENSE mode: return only similarity scores
            output = {"field_type": field_type, "similarity": round(similarity.mean().item(), 4)}

            # Store results in dataframe if provided
            if dataframe is not None:
                sorted_keys = sorted(list(ref_data.keys()))
                for tmp_idx, tmp_key in enumerate(sorted_keys):
                    tmp_item = [tmp_key, field_type, 0, similarity[tmp_idx].item(), None, None]
                    dataframe.loc[len(dataframe)] = tmp_item

            return output, dataframe
        else:
            # DATE mode: calculate both similarity and discrimination
            ref_keys = list(ref_data.keys())
            logger.info(f"Scores shape: {scores.shape}")
            delta = delta #* 6  # Scale delta parameter
            rank = torch.zeros(len(ref_data), device=self.device)

            # Calculate discrimination scores for each reference
            for tmp_idx, tmp_ref_key in tqdm(enumerate(ref_keys), desc="calculate the discrimination"):
                # Count samples with scores greater than current + delta/2
                count_ge = torch.sum(scores[tmp_idx] > scores[tmp_idx][tmp_idx] + delta / 2).item()
                # Count samples with scores within delta range
                count_eq = torch.sum(
                    (scores[tmp_idx] <= scores[tmp_idx][tmp_idx] + delta / 2)
                    & (scores[tmp_idx] >= scores[tmp_idx][tmp_idx] - delta / 2)
                ).item()
                # Calculate rank-based discrimination score
                rank[tmp_idx] = 1 - (count_ge + count_eq) / len(ref_data)

            discrimination = rank

            # Calculate the DATE score (harmonic mean of similarity and discrimination)
            DATE_score = 2 * similarity * discrimination / (similarity + discrimination)

            # Convert to numpy for output
            similarity = similarity.detach().cpu().numpy()
            discrimination = discrimination.detach().cpu().numpy()
            DATE_score = DATE_score.detach().cpu().numpy()

            # Store results in dataframe if provided
            if dataframe is not None:
                sorted_keys = sorted(list(ref_data.keys()))
                for tmp_idx, tmp_key in enumerate(sorted_keys):
                    tmp_item = [
                        tmp_key,
                        field_type,
                        delta / 2,
                        similarity[tmp_idx],
                        discrimination[tmp_idx],
                        DATE_score[tmp_idx],
                    ]
                    dataframe.loc[len(dataframe)] = tmp_item

            # Prepare output with all evaluation metrics
            output = {
                "field_type": field_type,
                "delta": delta / 2,
                "similarity": round(similarity.mean(), 4),
                "discrimination": round(discrimination.mean(), 4),
                "date": round(DATE_score.mean(), 4),
            }

            return output, dataframe


class DATEEvaluator:
    """
    DATE (Discriminability based Audio Task Evaluation) evaluator with FENSE-compatible interface.

    This class provides a FENSE-compatible interface for the DATE evaluation metric,
    allowing seamless integration with existing evaluation pipelines. It supports
    both FENSE and DATE evaluation modes through the return_type parameter.

    Key Features:
    - FENSE-compatible API for easy integration
    - Support for both corpus-level and sentence-level evaluation
    - Batch processing for efficiency
    - Configurable evaluation modes (FENSE or DATE)

    Usage:
        evaluator = DATEEvaluator(return_type='date')
        score = evaluator.corpus_score(candidates, references)
    """

    def __init__(
        self,
        batch_size=32,
        device=None,
        sbert_model="paraphrase-TinyBERT-L6-v2",
        echecker_model="echecker_clotho_audiocaps_base",
        error_threshold=0.9,
        penalty=0.9,
        use_proxy=False,
        proxies=None,
        is_clamp_neg_similarity=False,
        return_type="date",
        cpu_priority=True,  # Add CPU priority parameter
    ):
        """
        Initialize DATEEvaluator with FENSE-compatible interface.

        Args:
            batch_size (int): Batch size for processing
            device (str): Device to run models on ('cuda' or 'cpu')
            sbert_model (str): Sentence transformer model name or path
            echecker_model (str): Error checker model name or path
            error_threshold (float): Threshold for error detection
            penalty (float): Penalty factor for detected errors
            use_proxy (bool): Whether to use proxy for model downloads
            proxies (str): Proxy configuration
            is_clamp_neg_similarity (bool): Whether to clamp negative similarities
            return_type (str): Evaluation mode ('fense' or 'date')
            cpu_priority (bool): Whether to prioritize CPU memory usage over GPU speed
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        
        # Initialize DATE model with CPU priority option
        self.date_model = DATE(
            sbert_name_or_path=sbert_model,
            echecker_name_or_path=echecker_model,
            device=self.device,
            error_threshold=error_threshold,
            penalty=penalty,
            use_proxy=use_proxy,
            proxies=proxies,
            is_clamp_neg_similarity=is_clamp_neg_similarity,
            return_type=return_type,
            cpu_priority=cpu_priority,  # Pass CPU priority parameter
        )

    def corpus_score(self, cands, list_refs, agg_score="mean", save_path=None, save_dataframe_path=None, sample_keys=None, group_name="content_long"):
        """
        Calculate corpus-level DATE score with FENSE-compatible interface.

        This method evaluates a corpus of candidate sentences against their corresponding
        reference sentences using the DATE metric. It provides the same interface as
        FENSE's corpus_score method for easy integration.

        Args:
            cands (list): List of candidate sentences to evaluate
            list_refs (list): List of reference sentence lists (each element is a list of references)
            agg_score (str): Aggregation method for scores ('none', 'mean', 'max')
                - 'none': Return individual scores for each candidate
                - 'mean': Return mean score across all candidates
                - 'max': Return maximum score across all candidates
            save_path (str): Path to save embeddings (optional)
            save_dataframe_path (str): Path to save dataframe results (optional)
            sample_keys (list): List of sample keys to preserve in dataframe (optional)
            group_name (str): Group name from GROUP_CONFIGS (default: "content_long")

        Returns:
            float or list: DATE score(s) depending on agg_score parameter
                - If agg_score='mean' or 'max': single float score
                - If agg_score='none': list of scores for each candidate
        """
        assert len(cands) == len(list_refs), "Number of candidates must match number of reference lists"
        assert agg_score in {"none", "mean", "max"}, "agg_score must be 'none', 'mean', or 'max'"

        # Extract subtask from group_name for data processing
        group_to_subtask_mapping = {
            "content_long": "long",
            "content_short": "short",
            "pure_speech": "speech",
            "mixed_speech": "speech",
            "pure_music": "music",
            "mixed_music": "music",
            "pure_sound": "sound",
            "mixed_sound": "sound",
            "environment": "environment"
        }
        subtask_for_data = group_to_subtask_mapping.get(group_name, "long")
        
        # Convert input format to DATE model's expected format
        ref_data = {}
        val_data = {}

        for i, (cand, refs) in enumerate(zip(cands, list_refs)):
            # Use provided sample_keys if available, otherwise use default ref_{i}
            if sample_keys is not None and i < len(sample_keys):
                key = sample_keys[i]
            else:
                key = f"ref_{i}"
            
            ref_data[key] = {subtask_for_data: refs}
            val_data[key] = {subtask_for_data: [cand]}
        
        # Create dataframe for storing detailed results if save_dataframe_path is provided
        dataframe = None
        if save_dataframe_path is not None:
            dataframe = pd.DataFrame(columns=["sample_id", "field_type", "delta", "similarity", "discrimination", "date"])

        # Run DATE evaluation
        output, dataframe = self.date_model(
            ref_data=ref_data,
            val_data=val_data,
            group_name=group_name,
            save_path=save_path,
            dataframe=dataframe,
            field_type=subtask_for_data,
            # delta=0.02,  # Default delta value for discrimination calculation
        )

        # Save dataframe if path is provided
        if save_dataframe_path is not None and dataframe is not None:
            os.makedirs(os.path.dirname(save_dataframe_path), exist_ok=True)
            dataframe.to_csv(save_dataframe_path, index=False)
            logger.info(f"Dataframe results saved to: {save_dataframe_path}")

        # Select appropriate score based on evaluation mode
        if self.date_model.return_type == "fense":
            score_key = "similarity"
        else:
            score_key = "date"

        # Apply aggregation method
        if agg_score == "mean":
            return output[score_key]
        elif agg_score == "max":
            return output[score_key]  # For corpus-level score, max equals the mean score
        else:
            return [output[score_key]] * len(cands)  # Return list of same scores for each candidate

    def sentence_score(self, cand, refs, return_error_prob=False, save_path=None, save_dataframe_path=None, subtask="text"):
        """
        Calculate sentence-level DATE score with FENSE-compatible interface.

        This method evaluates a single candidate sentence against a list of reference
        sentences using the DATE metric. It provides the same interface as FENSE's
        sentence_score method for easy integration.

        Args:
            cand (str): Candidate sentence to evaluate
            refs (list): List of reference sentences
            return_error_prob (bool): Whether to return error probability (legacy parameter)
                Note: DATE doesn't provide error probability, so this returns None
            save_path (str): Path to save embeddings (optional)
            save_dataframe_path (str): Path to save dataframe results (optional)
            subtask (str): Field name for the data (default: "text")

        Returns:
            float or tuple: DATE score, or tuple if return_error_prob=True
                - If return_error_prob=False: single float score
                - If return_error_prob=True: tuple of (score, None, score)
        """
        # Convert input format to DATE model's expected format
        ref_data = {"ref_0": {subtask: refs}}
        val_data = {"ref_0": {subtask: [cand]}}

        # Create dataframe for storing detailed results if save_dataframe_path is provided
        dataframe = None
        if save_dataframe_path is not None:
            dataframe = pd.DataFrame(columns=["sample_id", "field_type", "delta", "similarity", "discrimination", "date"])

        # Run DATE evaluation
        output, dataframe = self.date_model(
            ref_data=ref_data,
            val_data=val_data,
            subtask=subtask,
            save_path=save_path,
            dataframe=dataframe,
            field_type=subtask,
            # delta=0.02,  # Default delta value for discrimination calculation
        )

        # Save dataframe if path is provided
        if save_dataframe_path is not None and dataframe is not None:
            os.makedirs(os.path.dirname(save_dataframe_path), exist_ok=True)
            dataframe.to_csv(save_dataframe_path, index=False)
            logger.info(f"Dataframe results saved to: {save_dataframe_path}")

        # Select appropriate score based on evaluation mode
        if self.date_model.return_type == "fense":
            score_key = "similarity"
        else:
            score_key = "date"

        if return_error_prob:
            # DATE doesn't provide error probability, return score with None for error_prob
            return output[score_key], None, output[score_key]
        else:
            return output[score_key]

    def evaluate(self, predictions, references, subtask="text", save_path=None, save_dataframe_path=None):
        """
        Evaluate predictions against references using the DATE metric.

        This method provides a comprehensive evaluation interface that can handle
        both single references and multiple references per prediction. It returns
        detailed scores including similarity, discrimination, and the final DATE score.

        Args:
            predictions (list): List of prediction strings to evaluate
            references (list): List of reference strings or list of reference string lists
                - If each element is a string: single reference per prediction
                - If each element is a list: multiple references per prediction
            subtask (str): Field name for the data (default: "text")
            save_path (str): Path to save embeddings (optional)
            save_dataframe_path (str): Path to save dataframe results (optional)

        Returns:
            dict: Dictionary containing evaluation scores
                - For FENSE mode: {'fense': score, 'similarity': score}
                - For DATE mode: {'date': score, 'similarity': score, 'discrimination': score}
        """
        # Ensure references is a list of lists for consistent processing
        if isinstance(references[0], str):
            references = [[ref] for ref in references]

        # Convert input format to DATE model's expected format
        ref_data = {}
        val_data = {}

        for i, (pred, refs) in enumerate(zip(predictions, references)):
            ref_data[f"ref_{i}"] = {subtask: refs}
            val_data[f"ref_{i}"] = {subtask: [pred]}

        # Create dataframe for storing detailed results if save_dataframe_path is provided
        dataframe = None
        if save_dataframe_path is not None:
            dataframe = pd.DataFrame(columns=["sample_id", "field_type", "delta", "similarity", "discrimination", "date"])

        # Run DATE evaluation
        output, dataframe = self.date_model(
            ref_data=ref_data,
            val_data=val_data,
            subtask=subtask,
            save_path=save_path,
            dataframe=dataframe,
            field_type=subtask,
            # delta=0.02,  # Default delta value for discrimination calculation
        )

        # Save dataframe if path is provided
        if save_dataframe_path is not None and dataframe is not None:
            os.makedirs(os.path.dirname(save_dataframe_path), exist_ok=True)
            dataframe.to_csv(save_dataframe_path, index=False)
            logger.info(f"Dataframe results saved to: {save_dataframe_path}")

        # Return appropriate scores based on evaluation mode
        if self.date_model.return_type == "fense":
            return {"fense": output["similarity"], "similarity": output["similarity"]}
        else:
            return {
                "date": output["date"],
                "similarity": output["similarity"],
                "discrimination": output["discrimination"],
            }
