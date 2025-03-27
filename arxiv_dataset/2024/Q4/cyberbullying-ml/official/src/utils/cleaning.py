import pandas as pd
import numpy as np
import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from langdetect import detect, LangDetectException
import contractions

# GLOBALS

lemmatizer = WordNetLemmatizer()

stemmer = PorterStemmer()

STOP_WORDS = stopwords.words('english') + [w.capitalize() for w in stopwords.words('english')]

# FUNCTIONS

def lowercase(text):
  return text.lower()

def remove_non_english(text):
    try:
        lang = detect(text)
    except LangDetectException:
        lang = "unknown"
    return text if lang == "en" else ""

def remove_links(text):
    return re.sub(r'(?:http[s]?:\/\/)\S*|(?:www\.)?(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|tr\.im|is\.gd|cli\.gs|u\.nu|url\.ie|tiny\.cc|alturl\.com|ow\.ly|bit\.do|adoro\.to)\S*|http[^\s]*', r'', text)

def remove_html_entities(text):
    return re.sub(r'&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6});|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-fA-F]{1,6})', r' ', text)

def remove_mentions(text):
    return re.sub(r'@\w+', '', text)

def remove_rt(text):
    return re.sub(r'(?<=\s)rt(?=\s)|\Art(?=\s)|(?<=\s)rt\Z', r' ', text, flags=re.IGNORECASE)

def decontract(text):
    return contractions.fix(text)

def remove_punctuation(text):
    return re.sub(r'[^\w\s]', r'', text)

def remove_digits(text):
    return re.sub(r'\d', r' ', text)

def remove_short_tweets(text):
    return text if len(text.split()) > 2 else ""

def remove_space_at_beginning_or_end(text):
    return text.strip() 

def remove_excess_spaces(text):
    return re.sub(r'\s+', ' ', text)

def lemmatize(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(t) for t in tokens])

def stem(text):
    tokens = word_tokenize(text)    
    return ' '.join([stemmer.stem(t) for t in tokens])

def remove_stop_words(text):
    return ' '.join([w for w in word_tokenize(text) if w not in STOP_WORDS])     

def remove_broken_links(text):
    text = re.sub(r'http[s]?:\s*\/\s*\/t\.co\s*\/\w+', r' ', text)
    text = re.sub(r'http[s]?:\s*\/\s*\/t\.c', r' ', text)
    return text