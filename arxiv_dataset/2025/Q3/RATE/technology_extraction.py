import os
import re
import yaml
from typing import Dict, List, Optional, Union, Any
import backoff
import logging
import urllib3

import chromadb
from langchain.schema import Document
from openai import OpenAI

import prompts
import config
from doc_processing import sanitize, preprocess_text, load_spacy_model
from data_vector import load_retriever, retrieve_context


# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Disable telemetry
chromadb.config.Settings.telemetry_enabled = False
os.environ["POSTHOG_TELEMETRY_DISABLED"] = "1"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# --- MINIMAL Logging Configuration ---
# Set root logger to only show CRITICAL messages.
# Using a very simple format for any critical errors that might still appear.
logging.basicConfig(level=logging.CRITICAL, format='%(levelname)s:%(name)s: %(message)s')

# Ensure third-party libraries are also very quiet.
# CRITICAL is the highest level. ERROR is below that.
logging.getLogger("httpx").setLevel(logging.WARNING) # Or CRITICAL
logging.getLogger("httpcore").setLevel(logging.WARNING) # Or CRITICAL
logging.getLogger("openai").setLevel(logging.CRITICAL) 
logging.getLogger("chromadb").setLevel(logging.WARNING) # Or CRITICAL
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)
logging.getLogger("backoff").setLevel(logging.ERROR) # To see when backoff gives up
logging.getLogger("urllib3").setLevel(logging.WARNING) # Or CRITICAL


logger = logging.getLogger(__name__)
backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=60, logger=logger)
def generate_technologies(question: str, documents: List[Document], row_index: int, domain_focus: str) -> List[Dict[str, Any]]:
    """
    Generates technologies by querying an LLM with the question and retrieved documents.
    Parses the YAML output from the LLM.
    'row_index' is used for logging. 'domain_focus' tailors LLM instructions.
    """

    formatted_contexts = []
    if not documents:
        logger.warning(f"Row {row_index}: No context documents provided to generate_technologies.")
    
    for doc_idx, doc in enumerate(documents):
        if not isinstance(doc, Document) or not hasattr(doc, 'metadata'):
            logger.warning(f"Row {row_index}, CtxDoc {doc_idx+1}: Invalid/malformed Document object. Skipping.")
            continue

        apps_content = doc.metadata.get('applications')
        related_content = doc.metadata.get('related')
        tags_content = doc.metadata.get('tags')
        
        content_snippet = sanitize(doc.page_content, '')[:200] 

        formatted_context = f"""--- Document {doc_idx+1} ---
Technology Name: {sanitize(doc.metadata.get('name'), 'UnknownName')}
Type: {sanitize(doc.metadata.get('type'), 'UnknownType')}
Domain: {sanitize(doc.metadata.get('domain'), 'UnknownDomain')}
Description: {sanitize(doc.metadata.get('description'), 'No Description')}
Applications: {sanitize(apps_content, 'None')}
Related Technologies: {sanitize(related_content, 'None')}
Tags: {sanitize(tags_content, 'None')}
Content Snippet: {content_snippet}...
"""
        formatted_contexts.append(formatted_context)
    
    context_str = "\n---\n".join(formatted_contexts) if formatted_contexts else "No context documents were available or provided."
    
    final_prompt_str = prompts.TECH_EXTRACTOR_USER_PROMPT.format(question=question, context=context_str)
    log_prompt_snippet = final_prompt_str[:500] + '...' if len(final_prompt_str) > 500 else final_prompt_str
    logger.debug(f"Row {row_index}: LLM API Prompt (domain: {sanitize(domain_focus)}, snippet): {log_prompt_snippet}")


    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    
    response = client.chat.completions.create(
        model="deepseek-chat", 
        messages=[
            {"role": "system", "content": prompts.TECH_EXTRACTOR_SYSTEM_PROMPT.format(domain_focus=sanitize(domain_focus, 'the specified domain'))},
            {"role": "user", "content": final_prompt_str},
        ],
        stream=False,
        temperature=0.0, 
        max_tokens=4096 
    )
    
    answer_content = response.choices[0].message.content
    if not answer_content: 
        logger.error(f"Row {row_index}: LLM API returned an empty response for question: '{sanitize(question[:50], '')}...'.")
        return []

    logger.debug(f"Row {row_index}: LLM API raw response (first 200 chars): '{answer_content[:200]}...'")

    cleaned_content = answer_content.strip()
    if cleaned_content.startswith("```yaml"):
        cleaned_content = cleaned_content[len("```yaml"):].strip()
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[len("```"):].strip()
    if cleaned_content.endswith("```"):
        cleaned_content = cleaned_content[:-len("```")].strip()
    
    if not cleaned_content: 
        logger.warning(f"Row {row_index}: LLM response was empty after cleaning markdown fences.")
        return []

    try:
        parsed_output = yaml.safe_load(cleaned_content)
    except yaml.YAMLError as e:
        logger.error(f"Row {row_index}: YAML parsing failed for LLM response. Error: {e}\nContent Snippet: '{cleaned_content[:500]}'")
        return [] 

    if not isinstance(parsed_output, dict) or "technologies" not in parsed_output:
        if isinstance(parsed_output, list) and all(isinstance(item, dict) and "name" in item for item in parsed_output):
            logger.warning(f"Row {row_index}: Parsed YAML output was a list of technologies, directly using it.")
            parsed_output = {"technologies": parsed_output} 
        elif isinstance(parsed_output, dict) and parsed_output.get("technologies") == []: 
            logger.info(f"Row {row_index}: LLM explicitly outputted an empty 'technologies' list.")
            return []
        else:
            logger.error(f"Row {row_index}: Unexpected YAML structure from LLM. Expected dict with 'technologies' key. "
                                f"Got type: {type(parsed_output)}, Content: '{str(parsed_output)[:200]}'")
            return []

    extracted_technologies_info: List[Dict[str, Any]] = []
    raw_tech_list = parsed_output.get("technologies", []) 

    if not isinstance(raw_tech_list, list): 
        logger.warning(f"Row {row_index}: The 'technologies' field in parsed YAML is not a list. "
                        f"Type: {type(raw_tech_list)}. Content: '{str(raw_tech_list)[:100]}'")
        return []
    
    if not raw_tech_list: 
        logger.info(f"Row {row_index}: LLM returned an empty list under 'technologies' key for question: '{sanitize(question[:50], '')}...'.")
        return []

    for tech_item in raw_tech_list:
        if not isinstance(tech_item, dict): 
            logger.warning(f"Row {row_index}: A technology item in the list is not a dictionary. Skipping: {tech_item}")
            continue
        
        try:
            name = str(tech_item.get("name", "")).strip()
            confidence_val = tech_item.get("confidence") 
            source = str(tech_item.get("source", "")).strip().lower()
            category = str(tech_item.get("category", "N/A")).strip()

            if not name:
                logger.warning(f"Row {row_index}: Technology item has no 'name' or name is empty. Skipping: {tech_item}")
                continue
            if confidence_val is None:
                logger.warning(f"Row {row_index}: Technology item '{name}' is missing 'confidence' field. Skipping: {tech_item}")
                continue
            
            confidence = float(confidence_val) 

            if source != "question": 
                logger.debug(f"Row {row_index}: Technology '{name}' skipped. Source ('{source}') is not 'question'.")
                continue
            
            if confidence >= config.FILTER_CONFIDENCE: 
                extracted_technologies_info.append({
                    "name": name,
                    "confidence": confidence,
                    "category": category
                    })
            else:
                logger.debug(f"Row {row_index}: Technology '{name}' skipped due to low confidence ({confidence:.2f} < {config.FILTER_CONFIDENCE}).")

        except (TypeError, ValueError) as e: 
            logger.error(f"Row {row_index}: Error processing fields of technology item '{tech_item.get('name', 'UnknownName')}'. "
                                f"Error: {e}. Item data: {tech_item}")
            continue
    
    logger.info(f"Row {row_index}: Parsed {len(extracted_technologies_info)} valid technologies from LLM for domain '{sanitize(domain_focus)}'.")
    return extracted_technologies_info



class TechnologyExtractor:
    def __init__(self, technology_filter: Dict[str, List], domain_focus: str = "General Technology"):
        self.retriever: Optional[Any] = None
        self.nlp_model = load_spacy_model() 
        self.TECHNOLOGY_FILTER = technology_filter
        self.domain_focus = domain_focus
        logger.info(f"TechnologyExtractor initialized with domain focus: {self.domain_focus}")

    def initialize(self) -> None:
        """Initializes the RAG retriever. Raises RuntimeError if setup fails."""
        if self.retriever is None:
            logger.info("TechnologyExtractor: Initializing RAG system...")
            try:
                self.retriever = load_retriever()
                if self.retriever is None:
                    raise RuntimeError("RAG system setup failed to return a retriever.")
                logger.info("TechnologyExtractor: RAG system initialized successfully.")
            except Exception as e:
                logger.error(f"TechnologyExtractor: Failed to initialize RAG system: {e}", exc_info=True)
                raise RuntimeError(f"TechnologyExtractor: RAG system initialization failed: {e}") from e
        else:
            logger.info("TechnologyExtractor: RAG system already initialized.")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3, max_time=180, logger=logger)
    def _confirm_and_score_technologies_llm(self,
                                            technology_phrases: List[str],
                                            original_question_text: str, 
                                            row_index: int) -> List[str]:
        """
        Uses an LLM to confirm if each phrase is a technology, considering it within the
        context of the original_question_text, based on provided definitions, and assigns a confidence score.
        Keeps only technologies with a score > 6.
        Returns a list of confirmed technology names (maintaining casing from LLM).
        """
        if not technology_phrases:
            logger.debug(f"Row {row_index}: No technology_phrases provided to _confirm_and_score_technologies_llm. Returning empty list.")
            return []

        confirmed_tech_list: List[str] = []        

        client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

        for tech_phrase in technology_phrases:
            tech_phrase_stripped = tech_phrase.strip()
            if not tech_phrase_stripped:
                continue

            logger.debug(f"Row {row_index}: Sending to LLM for confirmation/scoring: '{tech_phrase_stripped}' (with full text context)")
            
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": prompts.TECH_VALIDATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": prompts.TECH_VALIDATOR_USER_PROMPT.format(question=original_question_text, tech_phrase_stripped=tech_phrase_stripped)},
                    ],
                    stream=False,
                    temperature=0.0,
                    max_tokens=350 
                )
                answer_content = response.choices[0].message.content

                if not answer_content:
                    logger.warning(f"Row {row_index}: LLM returned empty response for confirming '{tech_phrase_stripped}'. Skipping.")
                    continue
                
                logger.debug(f"Row {row_index}: LLM confirmation raw response for '{tech_phrase_stripped}': '{answer_content[:200]}...'")

                cleaned_content = answer_content.strip()
                if cleaned_content.startswith("```yaml"):
                    cleaned_content = cleaned_content[len("```yaml"):].strip()
                elif cleaned_content.startswith("```"):
                    cleaned_content = cleaned_content[len("```"):].strip()
                if cleaned_content.endswith("```"):
                    cleaned_content = cleaned_content[:-len("```")].strip()
                
                if not cleaned_content:
                    logger.warning(f"Row {row_index}: LLM response for '{tech_phrase_stripped}' was empty after cleaning markdown fences. Skipping.")
                    continue

                parsed_yaml = yaml.safe_load(cleaned_content)

                if isinstance(parsed_yaml, dict):
                    is_tech = parsed_yaml.get("is_technology") 
                    confidence = parsed_yaml.get("confidence_score")
                    name = str(parsed_yaml.get("technology_name", tech_phrase_stripped)).strip() 

                    if is_tech is True and isinstance(confidence, (int, float)) and confidence > 6:
                        if name: 
                            confirmed_tech_list.append(name)
                            logger.debug(f"Row {row_index}: Confirmed '{name}' (orig: '{tech_phrase_stripped}') as technology with score {confidence}. Reasoning: {parsed_yaml.get('reasoning', 'N/A')}")
                        else:
                            logger.warning(f"Row {row_index}: LLM confirmed technology for '{tech_phrase_stripped}' but provided an empty name. Skipping.")
                    elif is_tech is True:
                        logger.debug(f"Row {row_index}: Phrase '{name}' (orig: '{tech_phrase_stripped}') deemed technology by LLM but score {confidence} <= 6. Filtered out. Reasoning: {parsed_yaml.get('reasoning', 'N/A')}")
                    else: 
                        logger.debug(f"Row {row_index}: Phrase '{tech_phrase_stripped}' not deemed a technology by LLM. Reasoning: {parsed_yaml.get('reasoning', 'N/A')}")
                else:
                    logger.warning(f"Row {row_index}: Could not parse YAML as dict for confirming '{tech_phrase_stripped}'. Response: '{cleaned_content[:200]}'. Skipping.")

            except yaml.YAMLError as ye:
                logger.error(f"Row {row_index}: YAML parsing error for '{tech_phrase_stripped}': {ye}. Content snippet: '{cleaned_content[:200]}'. Skipping.", exc_info=False)
            except Exception as e: 
                logger.error(f"Row {row_index}: Error during LLM confirmation for '{tech_phrase_stripped}': {e}. Skipping.", exc_info=True)

        logger.info(f"Row {row_index}: LLM confirmed {len(confirmed_tech_list)} technologies with score > 6 from {len(technology_phrases)} candidates.")
        return confirmed_tech_list

    def extract_technologies(self, text: str, row_index: int = 0) -> List[str]:
        """
        Main extraction workflow with revised Step 4 for LLM Confirmation and Scoring.
        Output is a list of unique, sorted, lowercase technology strings.
        """
        self.initialize() 
        if self.retriever is None:
            logger.critical(f"Row {row_index}: Retriever is None in extract_technologies. Cannot proceed.")
            raise RuntimeError("Retriever is None after initialization attempt in extract_technologies.")

        preprocessed_text = preprocess_text(text) 
        if not preprocessed_text.strip():
            logger.warning(f"Row {row_index}: Input text is empty. Skipping extraction.")
            return []

        # Step 1: Retrieve Context
        logger.info(f"Row {row_index}: Retrieving context...")
        documents = retrieve_context(preprocessed_text, self.retriever) 

        # Step 2: Generate Candidate Technologies (LLM Gen 1)
        logger.info(f"Row {row_index}: Generating candidate technologies (LLM Gen 1)...")
        technologies_info_list = generate_technologies(preprocessed_text, documents, row_index, self.domain_focus) 

        # Step 3: Validate LLM Candidates against Text
        validated_tech_names: List[str] = [] 
        if technologies_info_list:
            logger.info(f"Row {row_index}: Validating {len(technologies_info_list)} candidates from LLM Gen 1...")
            validated_tech_names = self._validate_technologies(technologies_info_list, preprocessed_text)
            if not validated_tech_names: 
                logger.info(f"Row {row_index}: No technologies remained after initial validation of LLM Gen 1 candidates.")
        else:
            logger.info(f"Row {row_index}: No candidates generated by LLM Gen 1.")
        
        if not validated_tech_names: 
            logger.info(f"Row {row_index}: No validated technology names to send for LLM Gen 2 confirmation.")
            return []
        logger.info(f"Row {row_index}: {len(validated_tech_names)} technologies after validation (original casing): {validated_tech_names if len(validated_tech_names) < 5 else str(validated_tech_names[:5]) + '...'}")

        # Step 4: LLM-based Confirmation and Scoring (LLM Gen 2) - now with context
        logger.info(f"Row {row_index}: Confirming and scoring {len(validated_tech_names)} validated phrases (LLM Gen 2) with full text context...")
        confirmed_tech_names_from_llm = self._confirm_and_score_technologies_llm(
            validated_tech_names,
            preprocessed_text, # Pass the original preprocessed text as context
            row_index
        )
        
        if not confirmed_tech_names_from_llm:
            logger.info(f"Row {row_index}: No technologies confirmed by LLM Gen 2 with score > 6.")
            return []
        logger.info(f"Row {row_index}: {len(confirmed_tech_names_from_llm)} technologies after LLM Gen 2 confirmation: {confirmed_tech_names_from_llm if len(confirmed_tech_names_from_llm) < 5 else str(confirmed_tech_names_from_llm[:5]) + '...'}")

        # Step 4.5: Prepare for Filtering (Lowercase and Uniquefy for consistent processing)
        lowercase_techs_for_filter_set = set()
        ordered_unique_lowercase_techs = [] 
        for name in confirmed_tech_names_from_llm: 
            name_lower_stripped = name.lower().strip()
            if name_lower_stripped and name_lower_stripped not in lowercase_techs_for_filter_set:
                ordered_unique_lowercase_techs.append(name_lower_stripped)
                lowercase_techs_for_filter_set.add(name_lower_stripped)
        
        if not ordered_unique_lowercase_techs:
            logger.info(f"Row {row_index}: No non-empty unique lowercase technologies after LLM confirmation and processing. Nothing to filter.")
            return []
        logger.info(f"Row {row_index}: Prepared {len(ordered_unique_lowercase_techs)} unique lowercase phrases for filtering.")

        # Step 5: Apply TECHNOLOGY_FILTER (to lowercase terms)
        logger.info(f"Row {row_index}: Applying TECHNOLOGY_FILTER to {len(ordered_unique_lowercase_techs)} LLM-confirmed lowercase components...")
        filtered_techs_lowercase = self.filter_technologies(ordered_unique_lowercase_techs, self.TECHNOLOGY_FILTER) 
        logger.info(f"Row {row_index}: {len(filtered_techs_lowercase)} components after applying TECHNOLOGY_FILTER.")
        if not filtered_techs_lowercase:
            return []

        # Step 6: Acronym Priority Deduplication (operates on lowercase terms here)
        logger.info(f"Row {row_index}: Applying deduplication to {len(filtered_techs_lowercase)} lowercase components...")
        deduplicated_techs_lowercase = self._deduplicate_with_acronym_priority(filtered_techs_lowercase)
        logger.info(f"Row {row_index}: {len(deduplicated_techs_lowercase)} components after deduplication.")
        if not deduplicated_techs_lowercase:
            return []

        # Step 7: Final Sort 
        final_techs_list_lowercase = sorted(deduplicated_techs_lowercase) 

        log_final_techs_snippet = final_techs_list_lowercase if len(final_techs_list_lowercase) < 10 else str(final_techs_list_lowercase[:10]) + '...'
        logger.info(f"Row {row_index}: Final processed technologies ({len(final_techs_list_lowercase)}): {log_final_techs_snippet}")
        return final_techs_list_lowercase

    def _validate_technologies(self, extracted_techs_info: List[Dict[str, Any]], text: str) -> List[str]:
        """
        Validates LLM-extracted technologies against the original text using various heuristics.
        """
        validated_techs_list: List[str] = []
        if not text.strip():
            logger.debug(f"Validation: Input text for row is empty, returning no validated technologies.")
            return validated_techs_list
        
        text_lower = text.lower()
        doc_text_spacy = self.nlp_model(text) if self.nlp_model else None

        for tech_data in extracted_techs_info:
            tech_name_original = str(tech_data.get("name", "")).strip()
            if not tech_name_original:
                logger.warning(f"Validation: Skipping tech data with empty name: {tech_data}")
                continue

            tech_name_lower = tech_name_original.lower()
            try:
                confidence = float(tech_data.get("confidence", 0.0))
            except (ValueError, TypeError):
                confidence = 0.0
                logger.warning(f"Validation: Could not parse confidence for '{tech_name_original}', defaulting to 0.0. Original value: {tech_data.get('confidence')}")
            
            tech_name_escaped_lower = re.escape(tech_name_lower)
            base_term_from_name = re.sub(r'\s*\([^)]*\)', '', tech_name_lower).strip()
            base_term_escaped_lower = re.escape(base_term_from_name) if base_term_from_name else None
            
            acronym_match_obj = re.search(r'\(([A-Z0-9]{2,})\)$', tech_name_original)
            acronym_lower_escaped = re.escape(acronym_match_obj.group(1).lower()) if acronym_match_obj else None

            validated_by_rule = False
            if re.search(rf'\b{tech_name_escaped_lower}\b', text_lower):
                validated_techs_list.append(tech_name_original)
                logger.debug(f"Validation: Validated '{tech_name_original}' by direct match.")
                validated_by_rule = True
            elif base_term_escaped_lower and base_term_from_name != tech_name_lower and \
                 re.search(rf'\b{base_term_escaped_lower}\b', text_lower):
                validated_techs_list.append(tech_name_original)
                logger.debug(f"Validation: Validated '{tech_name_original}' by base term match ('{base_term_from_name}').")
                validated_by_rule = True
            elif acronym_lower_escaped and acronym_match_obj and \
                 re.search(rf'\b{acronym_lower_escaped}\b', text_lower):
                validated_techs_list.append(tech_name_original)
                logger.debug(f"Validation: Validated '{tech_name_original}' by acronym match ('{acronym_match_obj.group(1)}').")
                validated_by_rule = True
            else: 
                tech_significant_words = [w for w in re.split(r'[-\s]', tech_name_lower) if len(w) > 3]
                if len(tech_significant_words) > 1:
                    word_matches_count = sum(1 for word in tech_significant_words if re.search(rf'\b{re.escape(word)}\b', text_lower))
                    if (word_matches_count / len(tech_significant_words)) >= 0.75:
                        validated_techs_list.append(tech_name_original)
                        logger.debug(f"Validation: Validated '{tech_name_original}' by partial compound match ({word_matches_count}/{len(tech_significant_words)} words).")
                        validated_by_rule = True
            
            if not validated_by_rule and confidence >= 0.92: 
                validated_techs_list.append(tech_name_original)
                logger.debug(f"Validation: Validated '{tech_name_original}' due to very high LLM Gen 1 confidence ({confidence:.2f}).")
                validated_by_rule = True
            
            if not validated_by_rule and doc_text_spacy and confidence >= 0.75: 
                try:
                    doc_tech_spacy = self.nlp_model(tech_name_original)
                    if doc_text_spacy.has_vector and doc_tech_spacy.has_vector and \
                       doc_text_spacy.vector_norm and doc_tech_spacy.vector_norm: 
                        similarity_score = doc_text_spacy.similarity(doc_tech_spacy)
                        min_similarity_threshold = 0.70 
                        if similarity_score >= min_similarity_threshold:
                            validated_techs_list.append(tech_name_original)
                            logger.debug(f"Validation: Validated '{tech_name_original}' by semantic similarity ({similarity_score:.2f}).")
                            validated_by_rule = True
                        else:
                             logger.debug(f"Validation: Tech '{tech_name_original}' (LLM Gen 1 conf: {confidence:.2f}) semantic similarity {similarity_score:.2f} too low (threshold {min_similarity_threshold}).")
                    else:
                        logger.warning(f"Validation: Cannot compute semantic similarity for '{tech_name_original}': text or tech phrase lacks vectors or vector norm.")
                except Exception as e: 
                    logger.warning(f"Validation: Error during semantic validation for '{tech_name_original}': {e}", exc_info=False)
            
            if not validated_by_rule:
                 logger.debug(f"Validation: Could not validate '{tech_name_original}' (LLM Gen 1 conf: {confidence:.2f}). Failed all validation rules.")

        return list(dict.fromkeys(validated_techs_list))

    def filter_technologies(self, technologies: List[str], filter_config: Dict[str, List[str]]) -> List[str]:
        """
        Filters technologies based on predefined lists. Input 'technologies' expected to be lowercase.
        """
        if not filter_config or not any(filter_config.values()): 
            logger.debug("Technology filter configuration is empty or all lists are empty; returning original list.")
            return technologies

        filtered_techs_list: List[str] = []
        all_filter_terms_lower_exact: set[str] = set()

        for category, terms_list in filter_config.items():
            if not isinstance(terms_list, list):
                logger.warning(f"Filter category '{category}' does not contain a list of terms. Skipping this category.")
                continue
            for term in terms_list:
                term_lower = str(term).lower().strip()
                if term_lower:
                    all_filter_terms_lower_exact.add(term_lower)
        
        if not all_filter_terms_lower_exact: 
            logger.debug("No valid filter terms found after processing filter configuration. Returning original list.")
            return technologies

        for tech_string_lower in technologies:
            if tech_string_lower in all_filter_terms_lower_exact:
                logger.debug(f"Filtered out by TECHNOLOGY_FILTER: '{tech_string_lower}'.")
            else:
                filtered_techs_list.append(tech_string_lower) 
        
        logger.debug(f"Filtering complete. Input count: {len(technologies)}, Output count: {len(filtered_techs_list)}.")
        return filtered_techs_list

    def _deduplicate_with_acronym_priority(self, technologies: List[str]) -> List[str]:
        """
        Deduplicates a list of technology strings based on their base form.
        If multiple strings reduce to the same base form, it prefers the shorter version.
        Input 'technologies' are expected to be lowercase.
        """
        base_to_preferred_tech_map: Dict[str, str] = {} 
        
        for tech_string in technologies: 
            base_term = re.sub(r'\s*\([^)]*\)', '', tech_string).strip() 
            if not base_term: 
                continue

            if base_term not in base_to_preferred_tech_map:
                base_to_preferred_tech_map[base_term] = tech_string
            else:
                existing_tech_in_map = base_to_preferred_tech_map[base_term]
                if len(tech_string) < len(existing_tech_in_map):
                    base_to_preferred_tech_map[base_term] = tech_string
        
        result: List[str] = []
        processed_base_terms: set[str] = set() 
        for tech_string_in_original_order in technologies: 
            base_term = re.sub(r'\s*\([^)]*\)', '', tech_string_in_original_order).strip()
            if not base_term:
                continue
            if base_term not in processed_base_terms:
                result.append(base_to_preferred_tech_map[base_term]) 
                processed_base_terms.add(base_term)
        
        logger.debug(f"Deduplication (base term, prefer shorter on conflict) resulted in {len(result)} items from {len(technologies)} initial items.")
        return result
