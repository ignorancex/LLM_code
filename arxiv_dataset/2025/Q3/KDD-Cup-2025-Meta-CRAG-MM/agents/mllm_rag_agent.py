from typing import Dict, List, Any, Tuple, Optional
import os
import re
import numpy as np
from scipy import stats

import torch
from PIL import Image
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline
from crag_web_result_fetcher import WebSearchResult
import vllm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time

AICROWD_SUBMISSION_BATCH_SIZE = 8
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2
MAX_GENERATION_TOKENS = 75
RECALL_K = 10
RERANK_K = 3
RERANKER_BATCH_SIZE = 16
RERANKER_SCORE_THRESHOLD = 0.1


def initialize_helper_models(router_model_name: str):
    """Initializes and returns all helper models (router, reranker)."""
    print("Initializing helper models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Router LLM (Lightweight)
    print(f"Initializing Router LLM: {router_model_name}")
    router_llm = vllm.LLM(
        router_model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.2,
        max_model_len=2048,
        trust_remote_code=True,
        dtype="bfloat16",
    )
    router_tokenizer = router_llm.get_tokenizer()
    print("Router LLM loaded successfully.")

    # 2. Reranker
    print("Initializing Reranker...")
    reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3").to(device).eval()
    print("Reranker loaded successfully.")

    # verifier_llm = vllm.LLM(
    #     "w11wo/Llama-3.2-11B-Vision-Instruct-bnb-4bit-vqa-verifier",
    #     tensor_parallel_size=1,
    #     gpu_memory_utilization=0.65,
    #     max_model_len=2048,
    #     trust_remote_code=True,
    #     dtype="bfloat16",
    #     enforce_eager=True,
    #     limit_mm_per_prompt={"image": 1},
    # )
    # verifier_tokenizer = verifier_llm.get_tokenizer()

    return {
        "router": (router_llm, router_tokenizer),
        "reranker": (reranker_model, reranker_tokenizer),
        # "verifier": (verifier_llm, verifier_tokenizer),
    }


class MLLMRAGAgent(BaseAgent):
    """
    A RAG agent combining:
    - A dedicated lightweight LLM for intelligent query routing.
    - A robust two-stage retrieval pipeline with dynamic thresholding.
    - Query-aware image captioning.
    - A MLLM verification.
    """

    def __init__(
        self,
        search_pipeline: UnifiedSearchPipeline,
        model_name: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        router_model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
        max_gen_len: int = 64,
    ):
        super().__init__(search_pipeline)

        if search_pipeline is None:
            raise ValueError("Search pipeline is required for RAG agent")

        self.model_name = model_name
        self.router_model_name = router_model_name
        self.max_gen_len = max_gen_len
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.initialize_models()

    def initialize_models(self):
        """Initializes the main vLLM and all helper models."""
        print(f"Initializing main model {self.model_name} with vLLM...")
        main_llm_gpu_mem = VLLM_GPU_MEMORY_UTILIZATION - 0.2
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=main_llm_gpu_mem,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=True,
            limit_mm_per_prompt={"image": 1},
        )
        self.tokenizer = self.llm.get_tokenizer()
        print("Main model loaded successfully.")

        self.helpers = initialize_helper_models(self.router_model_name)

    def get_batch_size(self) -> int:
        return AICROWD_SUBMISSION_BATCH_SIZE

    def batch_route_queries(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Routes a batch of queries using the lightweight router LLM."""
        print("Routing queries with lightweight LLM...")
        router_llm, router_tokenizer = self.helpers["router"]

        router_inputs = []
        for query in queries:
            router_prompt = (
                "Analyze the following user question and classify it. Provide only the classifications.\n\n"
                f'Question: "{query}"\n\n'
                "Classifications:\n"
                "1. Needs External Info: [yes/no] (Does answering require knowledge beyond the image?)\n"
                "2. Is Real-Time: [yes/no] (Does it ask about 'today', 'latest', or current events?)"
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are a query classification expert. Respond concisely with the requested format.",
                },
                {"role": "user", "content": router_prompt},
            ]
            formatted_prompt = router_tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            router_inputs.append(formatted_prompt)

        outputs = router_llm.generate(
            router_inputs, sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=20, skip_special_tokens=True)
        )

        decisions = []
        for output in outputs:
            text = output.outputs[0].text.lower()
            decision = {
                "needs_external_info": "1. yes" in text,
                "is_real_time": "2. yes" in text,
            }
            decisions.append(decision)
        return decisions

    def batch_query_aware_summarize(self, queries: List[str], images: List[Image.Image]) -> List[str]:
        """Generates image summaries that are aware of the user's query."""
        print("Generating query-aware image summaries...")
        inputs = []
        for query, image in zip(queries, images):
            summarize_prompt = (
                f"A user is asking the following question about the image: '{query}'.\n"
                "Please provide a one-sentence summary of the image, focusing on key elements, "
                "objects, and any text that might be relevant to answering the question."
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert at describing images with a focus on details relevant to a specific question.",
                },
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": summarize_prompt}]},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})

        outputs = self.llm.generate(
            inputs, sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=50, skip_special_tokens=True)
        )

        summaries = [output.outputs[0].text.strip() for output in outputs]
        return summaries

    def _extract_common_attributes(self, entity_data):
        def paragraph_chunker(text):
            paragraphs = re.split(r"\n{2,}|==.*==\n", text.strip())
            paragraphs = [paragraph.strip().replace("\n", " ") for paragraph in paragraphs if paragraph.strip()]
            paragraphs = [paragraph for paragraph in paragraphs if not paragraph.endswith("==")]
            return paragraphs

        if not entity_data:
            return ""

        attributes = ["description", "caption", "summary"]
        contexts = []
        for attr in attributes:
            if attr in entity_data:
                info = entity_data[attr]
                if info is None:
                    continue
                contexts += paragraph_chunker(str(info))
        return contexts

    def rerank_and_filter(self, query: str, documents: List[str]) -> Tuple[List[str], float]:
        """Reranks documents and filters them using a dynamic threshold."""
        if not documents:
            return [], 0.0

        reranker_model, reranker_tokenizer = self.helpers["reranker"]
        pairs = [[query, doc] for doc in documents]
        scores = []
        for i in range(0, len(pairs), RERANKER_BATCH_SIZE):
            batch_pairs = pairs[i : i + RERANKER_BATCH_SIZE]
            inputs = reranker_tokenizer(batch_pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            inputs = inputs.to(self.device)

            with torch.no_grad():
                logits = reranker_model(**inputs, return_dict=True).logits.view(-1).float()
                scores += torch.sigmoid(logits).cpu().numpy().tolist()

        scored_docs = sorted(
            [{"doc": doc, "score": score} for doc, score in zip(documents, scores)],
            key=lambda x: x["score"],
            reverse=True,
        )

        # Dynamic thresholding using MAD
        top_scores = [s["score"] for s in scored_docs[:10]]
        if len(top_scores) > 2:
            median_score = np.median(top_scores)
            mad = stats.median_abs_deviation(top_scores)
            dynamic_threshold = max(RERANKER_SCORE_THRESHOLD, median_score - 1.5 * mad)
        else:
            dynamic_threshold = RERANKER_SCORE_THRESHOLD

        final_docs = [item["doc"] for item in scored_docs if item["score"] >= dynamic_threshold][:RERANK_K]
        best_score = scored_docs[0]["score"] if scored_docs else 0.0

        return final_docs, best_score

    def batch_retrieve_and_prepare(
        self,
        queries: List[str],
        images: List[Image.Image],
        image_summaries: List[str],
        message_histories: List[List[Dict[str, Any]]],
        routing_decisions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict], List[float], List[str]]:
        """Performs retrieval and prepares inputs for the main LLM."""
        print("Retrieving and preparing RAG inputs...")
        all_inputs, all_ragged_inputs, all_best_scores, all_rag_contexts = [], [], [], []

        for i, query in enumerate(queries):
            routing = routing_decisions[i]
            rag_context, best_score = "", 0.0

            if routing["needs_external_info"]:
                search_query = f"{queries[i]} {image_summaries[i]}"
                if self.search_pipeline.web_search:  # text-based RAG
                    recalled_docs = self.search_pipeline(search_query, k=RECALL_K)
                else:  # image-based RAG
                    recalled_docs = self.search_pipeline(images[i], k=RECALL_K)
                documents_to_rerank = []
                if recalled_docs:
                    for res in recalled_docs:
                        entities = res.get("entities", [])
                        if entities:
                            contexts = [
                                self._extract_common_attributes(entity["entity_attributes"]) for entity in entities
                            ]
                            contexts = [
                                ctx for context in contexts for ctx in context if ctx
                            ]  # Filter out empty contexts
                            if contexts:
                                documents_to_rerank += contexts

                final_docs, best_score = self.rerank_and_filter(search_query, documents_to_rerank)
                if final_docs:
                    rag_context = "Here is some retrieved information:\n\n" + "\n\n".join(
                        f"[Info {j+1}] {doc}" for j, doc in enumerate(final_docs)
                    )

            all_best_scores.append(best_score)
            all_rag_contexts.append(rag_context)

            SYSTEM_PROMPT = (
                "You are a helpful and truthful assistant. Answer user questions based on the provided image and context. "
                "Keep your response concise. If you are unsure or the information is not available, respond with 'I don't know'."
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": [{"type": "image"}]}]
            if message_histories[i]:
                messages.extend(message_histories[i])

            ragged_messages = messages[:]
            if rag_context:
                ragged_messages.append({"role": "user", "content": rag_context})

            messages.append({"role": "user", "content": query})
            ragged_messages.append({"role": "user", "content": query})

            formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            all_inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": images[i]}})

            formatted_ragged_prompt = self.tokenizer.apply_chat_template(
                ragged_messages, add_generation_prompt=True, tokenize=False
            )
            all_ragged_inputs.append({"prompt": formatted_ragged_prompt, "multi_modal_data": {"image": images[i]}})

        return all_inputs, all_ragged_inputs, all_best_scores, all_rag_contexts

    def parse_verification_response(self, verification_text: str) -> Tuple[float, str]:
        """Parse confidence score and reasoning from verification response."""
        try:
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9]*\.?[0-9]+)', verification_text, re.IGNORECASE)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?:\nCONFIDENCE:|$)', verification_text, re.IGNORECASE | re.DOTALL)
            
            confidence = float(confidence_match.group(1)) if confidence_match else 0.3
            reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
            
            confidence = max(0.0, min(1.0, confidence))
            
            return confidence, reasoning
            
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return 0.3, "Failed to parse verification"


    def batch_verify_with_fusion(
        self, queries: List[str], images: List[Image.Image], answers: List[str], contexts: List[str]
    ) -> List[Tuple[float, str]]:
        """
        Verifies answers using a CoV-style quantitative scoring.
        Returns a list of (confidence_score, reasoning) tuples.
        """
        print(f"Verifying {len(answers)} answers with CoV-style scoring...")

        # 1. Get LLM verification results
        llm_verification_prompts = []
        indices_to_verify = []
        
        for i, answer in enumerate(answers):
            if answer.lower().strip() != "i don't know":
                verifier_prompt = (
                    "You are an expert fact-checker. Carefully verify if the given answer is correct and well-supported by the image and context. and then decompositionally if it passes the first check. Be very concise.\n"

                    "**Crucial Rule:** If the answer requires external knowledge (e.g., names, dates, numbers) that is NOT in the context AND NOT visible in the image, assign a confidence score of 0.0.\n\n"

                    "**Phase 1: Holistic Check.** Quickly assess the answer against these general criteria:\n"
                    "1. Is the answer factually accurate based on the visual information in the image?\n"
                    "2. If context is provided, is the answer consistent with it?\n"
                    "3. Are there any contradictions or uncertain claims?\n"
                    "4. Does the answer directly address the user's question?\n"
                    "5. Is the answer specific enough to be useful?\n\n"

                    "**If the answer fails Phase 1 check, immediately assign a low confidence score (<= 0.5).**\n"

                    "**Phase 2: Decompositional Check (Only if Phase 1 is passed).**\n"
                    "1. **Decompose:** Break the question into core sub-questions.\n"
                    "2. **Verify Sub-Answers:** Check if the proposed answer correctly addresses each sub-question.\n\n"
                    
                    "Rate your confidence on a scale of 0.0 to 1.0, where:\n"
                    "- 1.0 = Completely confident, answer is definitely correct\n"
                    "- 0.9 = Very confident, minor uncertainty\n"
                    "- 0.7 = Moderately confident, some uncertainty\n"
                    "- 0.5 = Low confidence, significant uncertainty\n"
                    "- 0.25 = Very low confidence, answer likely incorrect\n"
                    "- 0.0 = No confidence, answer is definitely wrong\n\n"

                    f"## Context:\n{contexts[i] if contexts[i] else 'No text context provided.'}\n\n"
                    f"## Question:\n{queries[i]}\n\n"
                    f"## Proposed Answer:\n{answers[i]}\n\n"
                    "## Your Task (Follow format strictly):\n"
                    "First, provide a confidence score. Then, on a new line, provide a **brief (1-2 sentences)** step-by-step reasoning for your verification. After that, on a new line, provide brief sub-questions and finding.\n"
                    "Respond in the following format ONLY:\n"
                    "CONFIDENCE: [A single float value between 0.0 and 1.0]\n"
                    "REASONING: [Your brief reasoning here]\n"
                    "SUB-QUESTIONS: Q1: [Sub-question 1], Finding: [e.g., Supported], Q2: [Sub-question 2], Finding: [e.g., Unsupported]..."
                )
                messages = [
                    {"role": "system", "content": "You are a verification expert. Follow the output format strictly."},
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": verifier_prompt}]},
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                llm_verification_prompts.append({"prompt": formatted_prompt, "multi_modal_data": {"image": images[i]}})
                indices_to_verify.append(i)

        final_results = [(1.0, "I don't know") for _ in range(len(answers))]
        if llm_verification_prompts:
            outputs = self.llm.generate(
                llm_verification_prompts,
                sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=100, skip_special_tokens=True),
            )
            for i, output in enumerate(outputs):
                original_idx = indices_to_verify[i]
                response_text = output.outputs[0].text.strip()
                print(f"Verifier output length: {len(self.tokenizer.encode(response_text))}")
                decision = output.outputs[0].text.strip().lower()
                confidence, reasoning = self.parse_verification_response(response_text)
                final_results[original_idx] = (confidence, reasoning)

        return final_results

    def batch_self_consistency_with_fusion(
        self,
        queries: List[str],
        images: List[Image.Image],
        answers: List[str],
        ragged_answers: List[str],
        contexts: List[str],
    ):
        """Implements self-consistency verification."""
        llm_consistency_prompts = []
        for i, (answer, ragged_answer) in enumerate(zip(answers, ragged_answers)):
            consistency_prompt = (
                "You are an impartial judge. You will determine if two answers to a question are consistent with each other and the provided context and image.\n\n"
                f"## Context:\n{contexts[i] if contexts[i] else 'No text context.'}\n\n"
                f"## Question:\n{queries[i]}\n\n"
                f"## Proposed Answer 1:\n{answer}\n\n"
                f"## Proposed Answer 2:\n{ragged_answer}\n\n"
                "## Your Task:\n"
                "Are the answers consistent with each other? Respond with only 'yes' or 'no'."
            )
            messages = [
                {"role": "system", "content": "You are a verification model. Respond with 'yes' or 'no'."},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": consistency_prompt}]},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            llm_consistency_prompts.append({"prompt": formatted_prompt, "multi_modal_data": {"image": images[i]}})

        outputs = self.llm.generate(
            llm_consistency_prompts,
            sampling_params=vllm.SamplingParams(temperature=0.0, max_tokens=5, skip_special_tokens=True),
        )

        results = [output.outputs[0].text.strip() for output in outputs]

        return results

    def batch_generate_response(
        self,
        queries: List[str],
        images: List[Image.Image],
        message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        start_time = time.time()
        print(f"Processing batch of {len(queries)} with Hybrid Advanced RAG pipeline...")

        # Step 1: Route queries using the lightweight LLM
        step1_start = time.time()
        routing_decisions = self.batch_route_queries(queries)
        print(f"  [TIMER] Step 1 (Routing) took: {time.time() - step1_start:.4f}s")

        # Step 2: Generate query-aware image summaries
        step2_start = time.time()
        image_summaries = self.batch_query_aware_summarize(queries, images)
        print(f"  [TIMER] Step 2 (Summarize) took: {time.time() - step2_start:.4f}s")

        # Step 3: Retrieve and prepare RAG inputs for the main LLM
        step3_start = time.time()
        unragged_inputs, rag_inputs, best_retrieval_scores, rag_contexts = self.batch_retrieve_and_prepare(
            queries, images, image_summaries, message_histories, routing_decisions
        )
        print(f"  [TIMER] Step 3 (Retrieve & Prepare) took: {time.time() - step3_start:.4f}s")

        # Step 4: Generate initial responses
        print("Generating initial responses...")
        step4_start = time.time()
        outputs = self.llm.generate(
            rag_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1, top_p=0.9, max_tokens=MAX_GENERATION_TOKENS, skip_special_tokens=True
            ),
        )
        generated_ragged_responses = [output.outputs[0].text.strip() for output in outputs]

        unragged_outputs = self.llm.generate(
            unragged_inputs,
            sampling_params=vllm.SamplingParams(
                temperature=0.1, top_p=0.9, max_tokens=MAX_GENERATION_TOKENS, skip_special_tokens=True
            ),
        )
        generated_unragged_responses = [output.outputs[0].text.strip() for output in unragged_outputs]
        print(f"  [TIMER] Step 4 (Generation) took: {time.time() - step4_start:.4f}s")

        # Step 5: Verify answers with signal fusion
        step5_start = time.time()
        self_consistency_results = self.batch_self_consistency_with_fusion(
            queries, images, generated_unragged_responses, generated_ragged_responses, rag_contexts
        )

        ragged_verification_results = self.batch_verify_with_fusion(
            queries, images, generated_ragged_responses, rag_contexts
        )

        print(f"  [TIMER] Step 5 (Verification) took: {time.time() - step5_start:.4f}s")

        # Step 6: Finalize responses based on verification
        final_responses = []
        HIGH_CONFIDENCE_THRESHOLD = 1
        LOW_CONFIDENCE_THRESHOLD = 0.9
        for i in range(len(queries)):
            ragged_response = generated_ragged_responses[i]
            is_consistent = "yes" in self_consistency_results[i].lower()
            confidence_ragged, _ = ragged_verification_results[i]
            
            has_context = rag_contexts[i] != ""
            retrieval_score = best_retrieval_scores[i]
            routing = routing_decisions[i]

            # Apply a cascade of checks
            if routing["is_real_time"] and retrieval_score < 0.2:
                final_responses.append("I don't know")
                print(f"Query {i}: Real-time, low retrieval score. Final answer: I don't know")
            elif has_context and is_consistent and confidence_ragged >= LOW_CONFIDENCE_THRESHOLD:
                final_responses.append(ragged_response)
                print(f"Query {i}: Consistent with context. Final answer: {ragged_response}")
                # if don't have context, the ragged answer should be equvialent to the unragged answer
            elif not has_context and is_consistent and confidence_ragged >= HIGH_CONFIDENCE_THRESHOLD:
                final_responses.append(ragged_response)
                print(f"Query {i}: No context, valid ragged answer. Final answer: {ragged_response}")
            elif has_context and not is_consistent:
                final_responses.append("I don't know")
                print(f"Query {i}: Inconsistent. Final answer: I don't know")
            else:
                final_responses.append("I don't know")
                print(f"Query {i}: Final answer: I don't know")

        total_time = time.time() - start_time
        print(
            f"Successfully processed batch. Total time: {total_time:.4f}s, Time per query: {total_time / len(queries):.4f}s"
        )
        return final_responses
