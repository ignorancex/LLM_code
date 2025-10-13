import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import time
from tqdm import tqdm
import pandas as pd

with open('data1.pkl', 'rb') as content:
    data = pickle.load(content)


lemmatizer = WordNetLemmatizer()

def remove_stop(NP):
    # stop_words = set(stopwords.words("english"))
    stop_words = {'the', 'a', 'an'}
    words = word_tokenize(NP)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_NP = " ".join(filtered_words)

    return filtered_NP

def lemmatize_NP(NP):
    
    return " ".join([lemmatizer.lemmatize(word) for word in NP.split()])


concept_base_url = "http://api.conceptnet.io/c/en/"

def get_concept(word):
    word = word.replace(" ", "_")
    concept_url = concept_base_url + word
    response = requests.get(concept_url)
    if response.status_code == 200:
        return response.json()
    
def isA_relationship(NP):
    ## Remove articles
    NP = remove_stop(NP)

    ## If isA relationship available
    isA = False
    concept = get_concept(NP)
    for edge in concept['edges']:
        if edge["rel"]["label"] == "IsA":
           isA = True
           break

    if isA:
        print("isA available")

    return isA

def related_to_affordances(NP):

    ## Remove articles
    NP = remove_stop(NP)

    ## check relationship with given affordance list
    isRelated = False
    threshold = 0.04 #Experiment with this
    endpoint = "http://api.conceptnet.io/related/c/en/"
    affordance_list = ['grasp',	'lift',	'throw',	'push',	'fix',	'ride',	'play', 'watch',
                       'sit',	'feed',	'row',	'pour',	'look',	'write',	'type']

    for affordance in affordance_list:
        NP = NP.replace(" ", "_")
        response = requests.get(endpoint + NP + "?filter=/c/en/" + affordance)

        if response.status_code == 200:
            data = response.json()
            # print(data)
        else:
            # print("Request failed.")
            break
        # print(affordance, data)
        if len(data['related']) > 0 and data['related'][0]['weight'] > threshold:
            isRelated = True
            break

    return isRelated


def verify_object(NP):

    lemmatized_NP = lemmatize_NP(NP)

    isA = False
    isRelated = True

    if isA_relationship(NP) or isA_relationship(lemmatized_NP):
        isA = True

    if not isA:
        return False
    # if related_to_affordances(NP) or related_to_affordances(lemmatized_NP):
    #     isRelated = True
    
    if isA and isRelated: 
        return True
    return False




language_dict = {'english':'en',
                 'german':'de'}



def createData(data, start_idx=0, end_idx=5000):
        
    # Create empty dataframe to store results
    dataframe = pd.DataFrame()

    # # Load dataset in language
    # dataset = load_dataset("xnli", language_dict[language])

    # # Creating a corpus with 25 exaplmes from each language
    # corpus = [dataset['train'][i]['hypothesis'] for i in range(number_of_examples)]

    # # Find all noun phrases in corpus
    # text_NP = findNP_in_XNLI(number_of_examples, language)

    filtered_objects = []
    for NP_list in tqdm(data['text_NP'][start_idx:end_idx]):
        print(NP_list)
        obj = []
        for NP in NP_list: 
            if isinstance(NP, str) and verify_object(NP):
                obj.append(NP)
            time.sleep(1)
        filtered_objects.append(obj)

    # Storing noun phrases in a dataframe
    dataframe['INPUT:Sentence'] = pd.Series(data['text'][start_idx:end_idx])
    dataframe['INPUT:Object'] = pd.Series(filtered_objects)

    return dataframe


intervals = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]
intervals = [5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

# print(data['text'][500:503])
# print(data['text_NP'][500:503])

new_start = 5000

number_of_examples = 500
for i in intervals:

    try:
        # Create dataframes
        dataframe_en = createData(data, i-new_start, i-new_start+number_of_examples)
        # dataframe_de = createData('german')
        # dataframe = pd.concat([dataframe_en, dataframe_de])


        dataframe = dataframe_en.explode('INPUT:Object')
        dataframe = dataframe.dropna()
        dataframe = dataframe.reset_index(drop=True)

        # Download dataframe as csv
        # dataframe.to_csv('result1-500.csv')
        dataframe.to_csv('data/noun_phrase_extractions/result%s-%s.tsv'%(i+1, number_of_examples+i+1), sep='\t')

        print(dataframe.head())
    except Exception as e:
        print("%s did not work"%i)
        print(e)
        time.sleep(10)