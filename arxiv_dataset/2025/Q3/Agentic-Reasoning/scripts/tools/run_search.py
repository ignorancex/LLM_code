from tools.bing_search import bing_web_search, extract_relevant_info, extract_snippet_with_context, fetch_page_content
from agentic_reason.generation import generate_webpage_to_reasonchain_batch
from agentic_reason.utils import extract_reasoning_context
import json
from agentic_reason.config import BEGIN_SEARCH_QUERY

class search_agent:
    def __init__(self, llm, tokenizer, bing_subscription_key, bing_endpoint, top_k = 10, output_records = [], all_urls_to_fetch = set(), url_snippets = {}, mind_map = None, search_cache = {}, url_cache = {}, use_jina = True, jina_api_key = None, max_doc_len = 1000, max_tokens = 8192, coherent = True, MAX_SEARCH_LIMIT = 10, MAX_TURN = 10):
        """
        Initialize the search agent
        """
        self.name = "search_agent"
        self.bing_subscription_key = bing_subscription_key
        self.bing_endpoint = bing_endpoint
        self.top_k = top_k
        self.mind_map = mind_map
        self.search_cache = search_cache
        self.url_cache = url_cache
        self.use_jina = use_jina
        self.jina_api_key = jina_api_key
        self.max_doc_len = max_doc_len
        self.max_tokens = max_tokens
        self.coherent = coherent
        self.MAX_SEARCH_LIMIT = MAX_SEARCH_LIMIT
        self.MAX_TURN = MAX_TURN
        self.output_records = output_records
        self.llm = llm
        self.tokenizer = tokenizer

        self.formatted_documents = []
        self.relevant_info = []
        self.url_snippets = {}
        self.executed_search_queries = []
        self.history = []
        self.search_count = 0
        self.all_urls_to_fetch =  set()

    def insert_mind_map(self, reason):
        text_to_insert = reason.split(BEGIN_SEARCH_QUERY)[0]
        self.mind_map.insert(text_to_insert)
    
    def check_search_cache(self, search_query):
        # Execute search, use cache if available
        if search_query in self.executed_search_queries:
                limit_message = "You have searched this query. Please refer to previous results."
                self.history.append(limit_message)
                print(f"Repeated search for query: \"{search_query}\"")
                return None
        if self.search_cache:
            if search_query in self.search_cache:
                results = self.search_cache[search_query]
                print(f"Using cached search results for query: \"{search_query}\"")
                return results
            else:
                return None
        else:
            return None
    
    def bing_search(self, bing_subscription_key, bing_endpoint, search_query, top_k = 10):
        """
        Execute Bing search and update caches
        """
        try:
            results = bing_web_search(search_query, bing_subscription_key, bing_endpoint, market='en-US', language='en')
            self.search_cache[search_query] = results
            print(f"Executed and cached search for query: \"{search_query}\"")
        except Exception as e:
            print(f"Error during search query '{search_query}': {e}")
            self.search_cache[search_query] = {}
            results = {}

        # Extract relevant information from Bing search results
        relevant_info = extract_relevant_info(results)[:top_k]
        self.relevant_info.append(relevant_info)

        # Extract URLs and snippets
        urls_to_fetch = [it['url'] for it in relevant_info]
        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

        # Filter URLs that are not cached
        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in self.url_cache]
        cached_urls = [u for u in urls_to_fetch if u in self.url_cache]

        # Store info for all_urls_to_fetch and url_snippets
        for url in urls_to_fetch_filtered:
            self.all_urls_to_fetch.add(url)
            self.url_snippets[url] = snippets.get(url, "")

        self.executed_search_queries.append(search_query)
        self.search_count += 1

        return relevant_info, url, snippets.get(url, "")

    def get_reasoning_context(self, reason):
        """
        Get reasoning context from the reason
        """
        all_reasoning_steps = reason.replace('\n\n', '\n').split("\n")
        truncated_prev_reasoning = extract_reasoning_context(all_reasoning_steps, mind_map=self.mind_map)

        return truncated_prev_reasoning

    # Collect parameters for batch processing
    # batch_relevant_info.append(relevant_info)
    # batch_original_questions.append(seq['item']['Question'])
    # batch_prev_reasonings.append(truncated_prev_reasoning)
    # batch_search_queries.append(search_query)
    def fetch_urls(self, urls = None):
        """
        fetch all URLs to cache
        """
        res = {}
        if not urls:
            urls = self.all_urls_to_fetch

        if urls:
            print(f"Fetching {len(urls)} URLs...")
        try:
            fetched_contents = fetch_page_content(
                list(urls),
                use_jina=self.use_jina,
                jina_api_key=self.jina_api_key,
                # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
            )
            print(f"Fetched {len(fetched_contents)} URLs successfully.")
        except Exception as e:
            print(f"Error during batch URL fetching: {e}")
            fetched_contents = {url: f"Error fetching URL: {e}" for url in urls}
        # Update cache with fetched contents
        for url, content in fetched_contents.items():
            self.url_cache[url] = content
            res[url] = content
        
        return res

    def to_doc(self, relevant_info = None):
        # After fetching, prepare formatted documents for batch processing
        formatted_documents = ""
        if not relevant_info:
            relevant_info = self.relevant_info
        for i, doc_info in enumerate(relevant_info):
            url = doc_info['url']
            raw_context = self.url_cache.get(url, "")
            doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
            success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=self.max_doc_len)
            if success:
                context = filtered_context
            else:
                context = raw_context[:self.max_doc_len*2]

            doc_info['context'] = context
            formatted_documents += f"**Web Page {i + 1}:**\n"
            formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
        self.formatted_documents = formatted_documents
        return formatted_documents
    
    def __call__(self, context, search_query):
        """
        search the query online based on previous reasoning context
        """
        if self.mind_map:
            self.insert_mind_map(context)

        cache_res = self.check_search_cache(search_query)
        if cache_res:
            return cache_res
        
        relevant_info, url, snippets = self.bing_search(self.bing_subscription_key, self.bing_endpoint, search_query, top_k = self.top_k)
        context = self.get_reasoning_context(context)
        urls_content = self.fetch_urls(url) # cache the urls with content
        docs = self.to_doc(relevant_info, self.max_doc_len)

        # After fetching, prepare for batch processing if there are any
        webpage_analyses = generate_webpage_to_reasonchain_batch(
            original_questions= [],
            prev_reasonings=context,
            search_queries=search_query,
            documents=docs,
            batch_output_records=self.output_records,  # Pass the collection list
            llm=self.llm,
            tokenizer=self.tokenizer,
            max_tokens=self.max_tokens,
            coherent=self.coherent,
        )
        print("generation completed, assigning outputs to sequences...")

        return webpage_analyses
    
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
