import logging
import re
from tqdm import tqdm
import sys
sys.path.append("..")
from utils import load_pkl, load_json
from utils import clean_text, extract_domain, load_gzip
from prompts import sys_msg, init_guess_sys_prompt, answer_sys_prompt, requirements, update_description_msg
from google_search import GoogleSearch
from wiki_collection import extract_wiki_info
from article_collection import collect_article_info

logger = logging.getLogger(__name__)

def write_to_file(file_path, data, append=True):
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding='utf-8') as file:
        file.write(data + "\n")


class MediaCheck:
    def __init__(self, 
                 LLM, 
                 sampling_params, 
                 wiki_flag = True, 
                 article_flag = True, 
                 google_flag = True, 
                 credibility_data_file = None, 
                 label_demo_file = None, 
                 description_demo_file = None, 
                 query_file = None, 
                 all_evidence_url_file = None, 
                 all_evidence_text_file = None, 
                 article_info_file = None,
                 wiki_info_file = None):

        logger.info("Initializing MediaCheck class.")

        self.LLM = LLM
        self.sampling_params = sampling_params

        self.credibility_data_file = credibility_data_file
        self.predict_label_demo_file = label_demo_file
        self.description_demo_file = description_demo_file
        self.query_file = query_file
        self.all_evidence_url_file = all_evidence_url_file
        self.all_evidence_text_file = all_evidence_text_file
        self.article_info_file = article_info_file
        self.wiki_info_file = wiki_info_file

        self.wiki_flag = wiki_flag
        self.article_flag = article_flag
        self.google_flag = google_flag

        self.credibility_data = load_pkl(credibility_data_file)

    def get_answers_from_evidence(self, domain, queries, google_evidence, output_file="./media_bg_generated/answers.txt"):

        prompts = []
        no_evidence_query_idx = []
        for k, item in enumerate(queries):
            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
            cur_evid = google_evidence[k]
            if len(cur_evid):
                evid = clean_text(str(cur_evid[0]))[:3000]
                prompts.append(answer_sys_prompt % (cur_query, evid, cur_query))
            else:
                # evid = "No external evidence available."
                no_evidence_query_idx.append(k)

        logger.info("Generating answers from evidence for domain %s.", domain)

        outputs = self.LLM.generate(prompts, self.sampling_params)
        answers = [clean_text(output.outputs[0].text)[:512] for output in outputs]

        # Write the results to the file instead of printing
        results = ""
        idx = 0
        for k, item in enumerate(queries):
            if k not in no_evidence_query_idx:
                cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
                answer = answers[idx]
                idx += 1
                results += f"Question: {cur_query}\n\nPrompt:\n {prompts.pop(0)}\n\n\tAnswer: {answer}\n"
                results += "-------------------------\n"
        write_to_file(output_file, results, append = True)  # Save to file

        return answers, no_evidence_query_idx


    def generate_initial_guess(self, source_names, wiki_collection, articles_collection, num_demo, output_file="./media_bg_generated/initial_guess.txt"):
        prompts = []
        for idx, source_name in enumerate(source_names):
            logger.info("Generating initial guess for %s", source_name)
    

            prompt = "Build a background check for the news source \"" + source_name + "\". Write down everything you know about them, e.g. who funds them, how they make money, if they have any particular bias. Make an ITEMIZED LIST. Be brief, and if you don't know something, just leave it out.\n\n"

            if self.wiki_flag:
                wiki = wiki_collection[idx]
                wiki_summary = wiki.get('summary', "No wikipedia information available.")

                prompt += (
                    "**Wikipedia Information (if any)**:\n"
                    f"{wiki_summary}\n\n"
                )

            if self.article_flag:
                articles = articles_collection[idx]
                if articles:
                    titles = [article['title'] for article in articles[:10]]
                    titles_str = '\n- '.join(titles)
                else:
                    titles_str = "No articles from the website available."
                
                prompt += (
                    f"**Article Titles from the source (if any)**:\n"
                    f"- {titles_str}\n\n"
                )

            if num_demo > 0:
                prompt += "Below are some examples of background checks, you may follow the format."
                prompt += "\n" + self.get_icl_string(num_demo) + "\n\n"
            

            input_prompt = init_guess_sys_prompt + prompt + requirements%(source_name)
            logger.info("Initial prompt generated for initial guess.")
            
            prompts.append(input_prompt)

        outputs = self.LLM.generate(prompts, self.sampling_params)
        extracted_outputs = [clean_text(output.outputs[0].text)[:3000] for output in outputs]

        # Write initial guesses and prompts to the file
        results = ""
        for idx, initial_prompt in enumerate(prompts):
            # guess = extracted_outputs[idx].split("**")[1:]
            # guess = "**" + "**".join(guess)
            # result = f"Prompt: {initial_prompt}\n\nInitial Guess: {guess}"
            results += f"Prompt: {initial_prompt}\n\nInitial Guess: {extracted_outputs[idx]}"
            results += "----------------------------------------------------\n\n"

        write_to_file(output_file, results)  # Save to file

        return extracted_outputs, prompts

    def get_icl_string(self, num_demo):
        all_texts=["The following are examples of background checks:"]

        demos = load_gzip(self.description_demo_file)

        for cat_name in demos:
            demonstrations=demos[cat_name][:num_demo]
            for demo in demonstrations:
                cur_text=[]
                media=re.findall(r'\(.*?\)',demo['media'])
                if len(media):
                    removed=media[0]
                    media=demo['media'].split(removed)[0].strip()
                else:
                    media=demo['media']

                head="## "+media+"\n"
                details=demo['details']
                details="\n".join(details)
                cur_text.append(head)

                if self.wiki_flag:
                    wiki = demo['wiki']
                    wiki_summary = wiki.get('summary', "No wikipedia information available.")

                    cur_text.append(f"**Wikipedia Information (if any)**:\n{wiki_summary}\n")
                    
                if self.article_flag:
                    articles=demo['articles']
                    if articles:
                        titles = [article['title'] for article in articles[:10]]
                        titles_str = '\n- '.join(titles)
                    else:
                        titles_str = "No articles from the website available."
                    
                    cur_text.append(f"**Article Titles from the source (if any)**:\n- {titles_str}\n")

            cur_text.append("**Background Check**:\n"+details)
            cur_text='\n'.join(cur_text)
            all_texts.append(cur_text)
        all_texts='\n\n'.join(all_texts)
        return all_texts

    def generate_description(self, evidence_links, num_demo = 1, output_file="./media_bg_generated/output.txt"):
        domains = []
        wiki_collection = []
        article_collection = []
        external_info_collection = []
        descriptions = {}

        for evidence_link in tqdm(evidence_links, desc = "generating media BG descriptions."):
            domain = extract_domain(evidence_link)

            if domain in self.credibility_data.keys(): 
                description = self.credibility_data[domain]["details"]
                descriptions[domain] = description
                logger.info(f"found existing description for {domain}")
            
            else:
                logger.info(f"generating description for {domain}")
                domains.append(domain)

                if self.google_flag:
                    google_search = GoogleSearch(self.query_file, self.all_evidence_url_file, self.all_evidence_text_file)
                    google_evidence = google_search.google_search(domain)

                    queries = load_json(self.query_file)

                    answers, no_evidence_query_idx = self.get_answers_from_evidence(domain, queries, google_evidence)  # list of answers

                    ext_prompt = "\n\n## Google search has revealed some new information:\n\n %s\n\n"

                    ext_prompt += "Update your background check for \"%s\" using the new information. Do NOT delete any information, but make ADDITIONS where necessary, using the new information. Most likely, you will just need to add an extra item to the itemized list you previously created. Make minimal edits, and only incorporate what is relevant. Begin your response with \"**Background check**\""
                    extra_answers = ""

                    extra_info = []
                    idx = 0
                    for k, item in enumerate(queries):
                        if k not in no_evidence_query_idx:
                            if "Unknown" in answers[idx]:
                                continue
                            cur_query = item['question'].strip().replace("X", "\"" + domain + "\"")
                            extra_answer = cur_query.replace("\"", "") + ": " + answers[idx]
                            extra_info.append(extra_answer)
                            idx+=1
                    if extra_info:
                        extra_answers = '-'+'\n- '.join(extra_info)
                    else:
                        extra_answers = "No information available."
                    external_info = ext_prompt % (extra_answers, domain)
                    external_info_collection.append(external_info)
                
                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain, self.wiki_info_file)
                    wiki_collection.append(wiki_info)

                if self.article_flag:
                    article_info = collect_article_info(domain, self.article_info_file)
                    article_collection.append(article_info)
                
        initial_guesses, init_prompts = self.generate_initial_guess(domains, wiki_collection, article_collection, num_demo)

        if not self.google_flag: #return initial guess
            for idx, domain in domains:
                descriptions[domain] = initial_guesses[idx]

        else: #update bg check with google info
            prompts = []
            for idx, guess in enumerate(initial_guesses):
                prompts.append(init_prompts[idx] + "\n" + guess + external_info_collection[idx] + update_description_msg)             

            outputs = self.LLM.generate(prompts, self.sampling_params)
            
            for idx, output in enumerate(outputs):
                description = clean_text(output.outputs[0].text)[:3000]
                descriptions[domains[idx]] = description
                self.credibility_data[domains[idx]] = {}
                self.credibility_data[domains[idx]]['details'] = description

            # Write final descriptions to the file
            results = ""
            idx = 0
            for idx, output in enumerate(outputs):
                description = clean_text(output.outputs[0].text)[:3000]
                results += f"Domain: {domain}\n\nPrompt:\n{prompts[idx]}\n\nDescription: {description}\n"
                results += "\n-------------------------------------------\n"
                idx += 1

            write_to_file(output_file, results)

        return descriptions

    def create_input_prompt(self, domains_to_label, descriptions, wiki_collection, num_demo):
        logger.info("creating input prompt for media classification")

        input_prompts = []

        for domain in domains_to_label: #loop over to output prompts in a known sequence

            if self.wiki_flag:
                wiki_info = wiki_collection[domain]
                wiki_summary = wiki_info.get('summary', "No wikipedia information available.")
                wiki_summary = wiki_summary
       
            examples=self.get_valid_examples(num_demo) # use mbfc data for few shot examples

            domain_prompt = ""
            idx = 1
            for exp in examples:
                if isinstance(exp["details"], list):
                    details = " ".join(exp["details"])
                domain_prompt += f"## Example {idx}\n"
                idx += 1
                domain_prompt += f"Media Description:\n{details}\n\n"

                if self.wiki_flag:
                    domain_prompt += f"Wikipedia:\n{exp["wiki_summary"]}\n\n"
                domain_prompt += f"Credibility: {exp["credibility"]}\n\n"

            
            domain_prompt += f"Target Media Description:\n{descriptions[domain]}\n\n"

            if self.wiki_flag:
                domain_prompt += f"Target Media Wikipedia:\n{wiki_summary}\n\n"
            domain_prompt += "Target Media Credibility:"
            input_prompts.append(domain_prompt)

        return input_prompts

    def get_valid_examples(self, num_demo):

        predict_label_demo = load_gzip(self.predict_label_demo_file) 
        examples = []
        for cat_name in predict_label_demo:
            demonstrations=predict_label_demo[cat_name][:num_demo]
            for demo in demonstrations:

                article = demo.get("articles", [{}])[0].get("cleaned_text", "")
                wiki_summary = demo.get("wiki", {}).get("summary", "")

                examples.append({
                "media": demo["media"],
                "details": demo["details"],
                "url" : demo["url"],
                "article_info":article,
                "wiki_summary":wiki_summary,
                "credibility": demo["credibility"]
            })

        return examples
    
    def predict_credibility_level(self, evidence_links, num_demo = 1, output_file="./media_bg_generated/labels.txt"):

        results = {}
        evidence_links_to_label = []
        domains_to_label = []
        wiki_collection = {}

        for evidence_link in tqdm(evidence_links, desc = "predicting credibility label"):
            logger.info("Predicting credibility level for evidence link: %s", evidence_link)

            domain = extract_domain(evidence_link)

            if domain in self.credibility_data.keys() and "credibility" in self.credibility_data[domain]: 
                # if yes, use existing credibility label
                logger.info(f"Using existing credibility level for domain {domain}")
                
                credibility_level = self.credibility_data[domain]["credibility"].lower()
                results[domain] = credibility_level

            else:
                # if no, generate label using collected information
                logger.info(f"No existing credibility level for domain {domain}, now generating...")

                evidence_links_to_label.append(evidence_link)
                domains_to_label.append(domain)

                if self.wiki_flag:
                    wiki_info = extract_wiki_info(domain, self.wiki_info_file)
                    wiki_collection[domain] = wiki_info

        descriptions = self.generate_description(evidence_links_to_label)

        msgs = self.create_input_prompt(domains_to_label, descriptions, wiki_collection, num_demo)

        prompts = [sys_msg+msg for msg in msgs]

        outputs = self.LLM.generate(prompts, self.sampling_params)

        valid_preds=['low','medium','high']

        for idx, output in enumerate(outputs):
            output = output.outputs[0].text.lower()
            pattern = re.compile('|'.join(valid_preds))
            match = pattern.search(output)

            if match:
                result = match.group(0)
            else:
                result = 'unknown'

            results[domains_to_label[idx]] = result
            self.credibility_data[domains_to_label[idx]]['credibility'] = result

        # Write prompts and results to the file
        string = ""
        idx = 0
        for idx, output in enumerate(outputs):
            output = output.outputs[0].text.lower()
            string += f"Domain: {domains_to_label[idx]}\n\nPrompt:\n{prompts[idx]}\n\nFinal Label: {results[domains_to_label[idx]]}\n\nOutput: {output}\n\n"
            string += "\n-------------------------------------------\n"
            idx += 1

        write_to_file(output_file, string)  # Save to file
        return results
