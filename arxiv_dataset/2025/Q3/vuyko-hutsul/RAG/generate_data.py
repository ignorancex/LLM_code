import pandas as pd
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import spacy
import pymorphy2
import difflib

def get_closest_sentences(index, df, query, top_n=3):
    query_embedding = get_embedding(query)
    _, closest_indices = index.search(np.array([query_embedding]), top_n)
    return df.iloc[closest_indices[0]]

api_key_v1=""
client = OpenAI(api_key=api_key_v1)

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   response = client.embeddings.create(input=[text], model=model)
   return response.data[0].embedding

def find_similar_words(word, df, nlp, column='Ukrainian', n=3):
    words = df[column].str.lower().tolist()
    word = word.lower()
    
    if word in words:
        row = df[df[column].str.lower() == word].iloc[0]
        return [{'uk_lemma': row['uk_lemma'], 'Hutsul': row['Hutsul']}]
    
    similar_words = difflib.get_close_matches(word, words, n=n)
    similar_df = df[df[column].str.lower().isin(similar_words)]
    
    scores = []
    final_words = []
    for similar_word in similar_words:
        score = difflib.SequenceMatcher(None, word, similar_word).ratio()
        if score > 0.8:
            doc1 = nlp(word) 
            doc2 = nlp(similar_word) 
            similarity_score = doc1.similarity(doc2)
            if similarity_score > 0.8:
                scores.append((similar_word, score))
                final_words.append(similar_word)
    result = []
    for word in final_words:
        row = df[df[column].str.lower() == word].iloc[0]
        result.append({'uk_lemma': row['uk_lemma'], 'Hutsul': row['Hutsul']})
    return result


df_dict = pd.read_csv('output/hutsul-ukrainian-dictionary_v2.csv')

nlp = spacy.load("uk_core_news_lg")
morph = pymorphy2.MorphAnalyzer(lang='uk')

def get_similar_words(text,df,nlp):
    doc = nlp(text)
    lemmas = []
    for token in doc:
        if token.pos_ in ('NOUN','ADJ','ADP','VERB'):
            lemmas.append(token.lemma_)
    similar_words = []
    for word in lemmas:
        similar_words = similar_words + find_similar_words(word, df, nlp, column='uk_lemma')
    formatted_string = ", ".join([f"{item['uk_lemma']} - {item['Hutsul']}" for item in similar_words])
    return formatted_string

index = faiss.read_index('output/faiss_index_emb_ivanchyk.bin')
df_rag = pd.read_csv('output/embedded_ivanchyk.csv')


import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Example usage
message = req
num_tokens = num_tokens_from_string(message, "cl100k_base")
print(f"Number of tokens: {num_tokens}")


req = '''Here are Grammatical Rules for Converting Ukrainian Text into the Hutsul Dialect
Below is a comprehensive set of grammatical rules to help you convert Ukrainian text into the Hutsul dialect.
They focus on phonetic changes, grammatical transformations, and syntactic patterns characteristic of the Hutsul dialect,
 with a particular emphasis on Word Order.
Note: Apply specific rules such as changing "и" to "і" or softening consonants by adding "ь" only when you are certain
 they are appropriate in the context.
1. Vowel Changes
1.1. "є" instead of "а" or "я" (when stressed)
"а" or "я" become "є" when stressed.
Examples:
"як" → "єк"
"ягода" → "єгода"
"яблуко" → "єблуко"
"яйце" → "єйце"
"ярмарок" → "єрмарок"
"теля" → "телє"
"шапка" → "шєпка"
"капелюх" → "кєпелюх"
"м'ята" → "мнєта"
"колядник" → "колідник"
"є" instead of "и" in some words:
Examples:
"йдеш" → "єдеш"
"йшли" → "єшли"
1.2. "е" or sometimes "і" instead of "и" (when stressed)
"и" becomes "е" or less commonly "і" when stressed.
Examples:
"абись" → "абес"
"абисьти" → "абести"
"жито" → "жето"
"вим'я" → "вім'ї"
1.3. "и" instead of "і"
"і" becomes "и".
Examples:
"Іван" → "Иван"
"іду" → "иду"
"із" → "из"
"зійшла" → "зийшла"
1.4. "и" or "і" instead of "я"
"я" becomes "и" or "і".
Examples:
"найся" → "найси"
"як ся маєте" → "як сі маєте"
"він знаходився" → "він находив сі"
"я забився" → "я забивсі"
1.5. "у" instead of "ю"
"ю" becomes "у".
Examples:
"сюди" → "суда"
"вівцю" → "вівцу"
"кішницю" → "кішницу"
1.6. "ві-" instead of "ви-" in prefixes
The prefix "ви-" becomes "ві-".
Examples:
"витратити" → "вітратити"
"виповісти" → "віповісти"
2. Consonant Changes
2.1. "ґ" instead of "д" (not always, only if confident)
Examples:
"дівка" → "ґівка"
"дірка" → "ґєрка"
"дідо" → "ґедьо"
"дякую" → "ґєкую"
"неділя" → "неґіля"
"понеділок" → "понєґівок"
"сидів" → "сиґів"
"поділа" → "поґіла"
2.2. "ть" instead of "к"
"к" becomes "ть" in certain words.
Example:
"донька" → "доньтя"
2.3. "д" instead of "ґ"
"ґ" becomes "д".
Example:
"леґінь" → "ледінь"
2.4. "к" instead of "т" before "є", "і", sometimes at word endings
"т" becomes "к" before "є", "і", or at the end of words.
Examples:
"християни" → "крескєни"
"тяжко" → "кєшко"
"повитягати" → "повікєгати"
"святі" → "свєкі"
"тіло" → "кіло"
"по світу" → "по свікі"
"смерть" → "смеркь"
2.5. Softened Sibilants (Sh, Ch sounds)
Softening of sibilant consonants by adding a soft sign "ь".
Examples:
"чого" → "чьо"
"нічого" → "нічьо"
"челядь" → "чєлєдь"
"шапка" → "шєпка"
"чари" → "чєри"
2.6. "ц" instead of "ч" and vice versa
"ч" becomes "ц", and "ц" becomes "ч".
Examples:
"чи" → "ци"
"цибух" → "чибук"
2.7. Hard "с", "ц", "з" instead of softened ones
Softened consonants "сь", "зь", "ць" become hard "с", "з", "ц".
Examples:
"щось" → "шос"
"десь" → "дес"
"на вулицю" → "на вулицу"
"крізь" → "кріз"
2.8. "нн", "н" instead of "дн", "тн", "лн"
"дн", "тн", "лн" become "нн" or "н".
Examples:
"передній" → "перенний"
"мельник" → "менник" or "меник"
3. Use of Particles and Conjunctions
3.1. "і" to "тай" or "та й"
The conjunction "і" is replaced with "тай" or "та й".
Example:
"Він і вона" → "Він тай вна"
3.2. "та" becomes "тай"
"та" changes to "тай".
Example:
"Більше та краще" → "Бирше тай красше"
3.3. "й" to "тай"
"й" becomes "тай".
Example:
"Він й вона" → "Він тай вна"
4. Negation
4.1. "не" to "ни"
"не" becomes "ни".
Example:
"Не знаю" → "Ни знаю"
4.2. Double Negation
Double negation is used for emphasis.
Example:
"Ніколи не бачив" → "Николи ни видів"
5. Pronouns and Possessives
5.1. "його" to "йиго"
"його" becomes "йиго".
Example:
"Його книга" → "Йиго книга"
5.2. "мене" to "мині"
"мене" becomes "мині".
Example:
"Він бачив мене" → "Він видів мині"
6. Reflexive Verbs
6.1. "ся" to "си"
"ся" becomes "си".
Example:
"Сміється" → "Смієтси"
6.2. Placement of "си"
"Си" is placed after the verb.
Example:
"Він думається" → "Він думаєтси"
7. Verb Endings and Conjugations
7.1. Third Person Singular Present Tense
Verbs ending in "-є" or "-е" change to "-єт".
Examples:
"Має" → "Маєт"
"Знає" → "Знаєт"
"Ходить" → "Ходит"
7.2. Past Tense Verbs
Past tense verbs often remain the same, but pronunciation may differ.
Example:
"Він сказав" → "Він сказав" (with dialectal pronunciation)
8. Prepositions
8.1. "від" to "вид"
"від" becomes "вид".
Example:
"Від тебе" → "Вид тебе"
8.2. "у" to "в"
"у" becomes "в" before consonants.
Example:
"У хаті" → "В хаті"
8.3. "до" becomes "д’" before vowels
"до" becomes "д'" when followed by a word starting with a vowel.
Example:
"До Андрія" → "Д'Андрія"
9. Word Order and Syntax
Word order in the Hutsul dialect can differ significantly from standard Ukrainian. The dialect often rearranges sentence components to emphasize certain elements, convey nuances, or align with traditional speech patterns.
9.1. Time and Place Expressions at the Beginning
Rule: Time and place expressions are frequently placed at the beginning of the sentence to set the scene or emphasize when or where an action occurs.
Examples:
Ukrainian: "В понеділок я піду до міста."
Hutsul: "В понєґівок я пиду до міста."
Ukrainian: "На засіданні ми обговорили питання."
Hutsul: "На засіданні ми обговорили питанє."
9.2. Emphasis Through Inversion
Rule: The Hutsul dialect uses inversion to emphasize certain parts of a sentence, such as placing the verb before the subject or moving objects to the front.
Examples:
Ukrainian: "Він представив законопроєкт."
Hutsul: "Представів він законопроєкт."
Ukrainian: "Я тебе люблю."
Hutsul: "Люблю я тебе."
9.3. Flexible Placement of Conjunctions
Rule: Conjunctions like "тай" can be used to link clauses or phrases, and their placement can vary to affect the rhythm and flow of the sentence.
Example:
Ukrainian: "Вона співала і танцювала."
Hutsul: "Вна співала, тай танцювала."
9.4. Subject Pronouns May Be Omitted
Rule: Subject pronouns are sometimes omitted if the verb form makes it clear who is performing the action.
Example:
Ukrainian: "Я йду додому."
Hutsul: "Иду додому."
9.5. Use of Reduplication for Emphasis
Rule: Words may be repeated for emphasis or to intensify the meaning.
Example:
Ukrainian: "Швидко-швидко біг."
Hutsul: "Борше-борше біг."
9.6. Placement of Reflexive Particles
Rule: The reflexive particle "си" can influence word order, often following the verb closely.
Example:
Ukrainian: "Він сміється."
Hutsul: "Він смієтси."
9.7. Questions Formed with Intonation and Particles
Rule: Questions may be formed by intonation and the use of particles like "ци," often placed at the beginning.
Example:
Ukrainian: "Ти знаєш?"
Hutsul: "Ци ти знаєш?"
9.8. Negative Constructions and Emphasis
Rule: Negation can be emphasized by placing "ни" before the verb and sometimes repeating it.
Example:
Ukrainian: "Він не прийшов."
Hutsul: "Він ни прийшов."
9.9. Order of Adjectives and Nouns
Rule: Adjectives usually precede nouns, but for emphasis, the order can be inverted.
Example:
Standard Order: "Красна ґівка" (Красна дівка)
Inverted for Emphasis: "Ґівка красна"
9.10. Use of Interjections and Fillers
Rule: Interjections like "ой," "єй," "ну" can be inserted to convey emotion or emphasis, affecting the rhythm and word order.
Example:
"Ой, єк то файно!"
10. Emphasis and Interjections
10.1. Use of "ой", "ай", "єй"
Interjections are used to express emotions and can be placed at various points in the sentence.
Examples:
"Ой, єк добре!"
"Ай, шо то робити?"
"Єй, мині тяжко!"
Expecially in the Hutsul dialect pay attention to the word order.
Also consider additional ukr-hutsul dictionary:
{words_examples}
And additional examples of hutsul:
{rag_examples}
Translate to hutsul, return only translation, each sentence in new line:
{text_to_translate}'''

class ChatGPTConnection:
    def __init__(self, api_key):
        self.api_key = api_key

    def send_message(self, message, model="gpt-4o"):
        response = client.chat.completions.create(model=model,
        messages=[
            {"role": "user", "content": message}
        ])
        return response.choices[0].message.content


with open('output/fiction.tokenized.shuffled_filtered.txt', 'r') as file:
    lines = file.readlines()
print(len(lines))

from tqdm import tqdm


sim_w_full = ''
rag_examples_full = ''
query_full = ''
connection = ChatGPTConnection(api_key=api_key_v1)
counter = 1
r1 = ''
r2 = ''

for query in tqdm(l):
    closest_sentences_df = get_closest_sentences(index, df_rag, query['description'])
    sim_w = get_similar_words(query['description'],df_dict,nlp)
    rag_examples_full = rag_examples_full + '\n '.join(closest_sentences_df['hutsul'].tolist())
    query_full = query_full + query['description'].rstrip()+'. \n'
    sim_w_full = sim_w_full +', '+ sim_w
    req_f = req.format(words_examples=sim_w_full,rag_examples=rag_examples_full,text_to_translate=query_full)
    num_tokens = num_tokens_from_string(req_f, "cl100k_base")
    if num_tokens > 20000:
        print(f"Number of tokens: {num_tokens}")
        response = connection.send_message(req_f)
        r1 = r1 + response+ "\n"
        sim_w_full = ''
        rag_examples_full = ''
        query_full = ''
        counter += 1

for query in tqdm(l):
    closest_sentences_df = get_closest_sentences(index, df_rag, query['title'])
    sim_w = get_similar_words(query['title'],df_dict,nlp)
    rag_examples_full = rag_examples_full + '\n '.join(closest_sentences_df['hutsul'].tolist())
    query_full = query_full + query['title'].rstrip()+'. \n'
    sim_w_full = sim_w_full +', '+ sim_w
    req_f = req.format(words_examples=sim_w_full,rag_examples=rag_examples_full,text_to_translate=query_full)
    num_tokens = num_tokens_from_string(req_f, "cl100k_base")
    if num_tokens > 20000:
        print(f"Number of tokens: {num_tokens}")
        response = connection.send_message(req_f)
        r2 = r2 + response+ "\n"
        sim_w_full = ''
        rag_examples_full = ''
        query_full = ''
        counter += 1

