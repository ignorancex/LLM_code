import json
import os

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--llm_retriever', type=str, default='openAI')
	args = parser.parse_args()
    	retriever_name = args.llm_retriever

	path = 'unipa-gpt/' # path till main folder
	
	with open(path+'corpora/qa_retrieved_'+retriever_name+'.json', 'r') as f:
    	qa_retrieved = json.load(f)

	template = """Sono Unipa-GPT, chatbot e assistente virtuale dell'Università degli Studi di Palermo che risponde cordialmente e in forma colloquiale.\nAi saluti, rispondi salutando e presentandoti.\nRicordati che il rettore dell'Università è il professore Massimo Midiri.\nSe la domanda riguarda l'università degli studi di Palermo, rispondi in base alle informazioni e riporta i link ad esse associate;\nSe non sai rispondere alla domanda, rispondi dicendo che sei un'intelligenza artificiale che ha ancora molto da imparare e suggerisci di andare su https://www.unipa.it/, non inventare risposte."""
	t2 = """\nDomanda: {question}\nInformazioni: {context}\nRisposta: """
	
	# insert your openai api key
	API_KEY = ""
	client = OpenAI(api_key = API_KEY)
	
	chat_gpt_answers = []
	for i in range(len(qa_retrieved)):
	    context = ''
	    for doc in qa_retrieved[i][1]:
	        context += doc['page_content'] + '\n'
	    query = t2.format(question=qa_retrieved[i][0], context=context)
	    response = client.chat.completions.create(
	        model="gpt-3.5-turbo-0125",
	        messages=[
	        {"role": "system", "content": template},
	        {"role": "user", "content": query}],
	        max_tokens = 256)
	    print(response.choices[0].message.content)
	    chat_gpt_answers.append((template, query, response.choices[0].message.content))

	os.makedirs(path+'results', exists_ok=True)
	with open(path+'results/qa_chatGPT-256.json', 'w') as f:
    	json.dump(chat_gpt_answers, f, indent=4)
