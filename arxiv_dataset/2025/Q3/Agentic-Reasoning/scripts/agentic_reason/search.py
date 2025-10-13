from typing import Optional, List, Dict, Set
import re
from tools.bing_search import (
    bing_web_search,
    extract_relevant_info,
    fetch_page_content,
    extract_snippet_with_context
)

# Special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
BEGIN_URL = "<|begin_url|>"
END_URL = "<|end_url|>"

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    """Extract text between two tags."""
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def process_search_query(
    search_query: str,
    sequence: Dict,
    search_cache: Dict,
    url_cache: Dict,
    max_search_limit: int,
    max_doc_len: int,
    bing_subscription_key: str,
    bing_endpoint: str,
    use_jina: bool = False,
    jina_api_key: str = None,
) -> Optional[List[Dict]]:
    """Process a search query and return relevant information."""
    
    # Check search limits
    if sequence['search_count'] >= max_search_limit:
        return None
        
    if search_query in sequence['executed_search_queries']:
        return None

    # Check if search result is in cache
    cache_key = f"{search_query}_{use_jina}"
    if cache_key in search_cache:
        return search_cache[cache_key]

    # Perform search
    try:
        relevant_info = bing_web_search(
            search_query,
            subscription_key=bing_subscription_key,
            endpoint=bing_endpoint,
            use_jina=use_jina,
            jina_api_key=jina_api_key
        )
        search_cache[cache_key] = relevant_info
    except Exception as e:
        print(f"Search error for query '{search_query}': {e}")
        return None

    return relevant_info

def fetch_and_process_urls(
    relevant_info: List[Dict],
    url_cache: Dict,
    max_doc_len: int,
    use_jina: bool = False,
    jina_api_key: str = None
) -> str:
    """Fetch and process URLs from search results."""
    
    formatted_documents = ""
    for i, doc_info in enumerate(relevant_info):
        url = doc_info['url']
        
        # Get content from cache or fetch
        if url not in url_cache:
            try:
                content = fetch_page_content(
                    url,
                    use_jina=use_jina,
                    jina_api_key=jina_api_key
                )
                url_cache[url] = content
            except Exception as e:
                print(f"Error fetching URL {url}: {e}")
                continue
                
        raw_context = url_cache.get(url, "")
        doc_info['snippet'] = doc_info['snippet'].replace('<b>', '').replace('</b>', '')
        
        success, filtered_context = extract_snippet_with_context(
            raw_context,
            doc_info['snippet'],
            context_chars=max_doc_len
        )
        
        context = filtered_context if success else raw_context[:max_doc_len*2]
        doc_info['context'] = context
        
        formatted_documents += f"**Web Page {i + 1}:**\n"
        formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"

    return formatted_documents

def handle_search_operation(
    sequence: Dict,
    search_query: str,
    search_cache: Dict,
    url_cache: Dict,
    max_search_limit: int,
    max_doc_len: int,
    bing_subscription_key: str,
    bing_endpoint: str,
    use_jina: bool = False,
    jina_api_key: str = None
) -> None:
    """Handle a search operation for a sequence."""
    
    relevant_info = process_search_query(
        search_query,
        sequence,
        search_cache,
        url_cache,
        max_search_limit,
        max_doc_len,
        bing_subscription_key,
        bing_endpoint,
        use_jina,
        jina_api_key
    )
    
    if relevant_info is None:
        limit_message = (
            f"\n{BEGIN_SEARCH_RESULT}\n"
            f"The maximum search limit is exceeded or query already executed.\n"
            f"{END_SEARCH_RESULT}\n"
        )
        sequence['prompt'] += limit_message
        sequence['output'] += limit_message
        sequence['history'].append(limit_message)
        return
        
    formatted_documents = fetch_and_process_urls(
        relevant_info,
        url_cache,
        max_doc_len,
        use_jina,
        jina_api_key
    )
    
    sequence['search_count'] += 1
    sequence['executed_search_queries'].add(search_query)
    
    return formatted_documents