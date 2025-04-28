import streamlit as st
from transformers import AutoProcessor, ASTModel
from tqdm import tqdm
from bing_image_downloader import downloader
import torch
import sklearn  
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import librosa
import os
import base64
import matplotlib.pyplot as plt
import webbrowser
import requests
from bs4 import BeautifulSoup



processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()



background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://lh3.googleusercontent.com/pw/AP1GczPRGjf9P6yxyTelG-Ku3XLWHg7xeITK2fGZVFAP9zOPChtQuCyZ8jNuLdcjNcHK2gffbYIXtMe9IyMuSJ0PidcLN-A3ou3P1ffvb4aA0LbWIZPuvk6I=w2400");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)




st.markdown("<h2 style='text-align: center; color: white; font-family: courier	;'>Europe Bird Sound Recognition</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: left; color: white;   font-family: courier; '> Upload Audio Here</h1>", unsafe_allow_html=True)

country_dict = {'Belgium': ['Black Woodpecker', 'African Pied Wagtail'], 'Czech Republic': ['African Pied Wagtail', 'European Turtle Dove', 'River Warbler'], 'Denmark': ['Eurasian Oystercatcher', 'African Pied Wagtail'], 'Estonia': ['African Pied Wagtail', 'Wood Sandpiper'], 'France': ['Black-headed Gull', 'European Nightjar', 'European Herring Gull', 'Rook', 'European Honey Buzzard', 'Western Jackdaw', 'Eurasian Bullfinch', 'African Pied Wagtail', 'Black Woodpecker', 'Common Pheasant'], 'Germany': ['Eurasian Wryneck', 'Eurasian Jay', 'Stock Dove', 'Great Spotted Woodpecker', 'Western Jackdaw', 'Eurasian Oystercatcher', 'African Pied Wagtail', 'Grey Partridge', 'Common Swift', 'Carrion Crow', 'Common Moorhen', 'Northern Raven', 'Eurasian Bullfinch'], 'Italy': ['European Turtle Dove', 'European Honey Buzzard', 'Eurasian Coot', 'African Pied Wagtail'], 'Latvia': ['African Pied Wagtail', 'Wood Sandpiper'], 'Netherlands': ['Eurasian Jay', 'Meadow Pipit', 'Stock Dove', 'Eurasian Coot', 'African Pied Wagtail', 'Carrion Crow', 'Common Moorhen', 'Northern Lapwing'], 'Norway': ['European Herring Gull', 'Meadow Pipit', 'Great Spotted Woodpecker', 'Eurasian Magpie', 'Willow Ptarmigan', 'Wood Sandpiper', 'African Pied Wagtail', 'European Golden Plover'], 'Poland': ['Eurasian Wryneck', 'River Warbler', 'Eurasian Magpie', 'Western Jackdaw', 'Eurasian Bullfinch', 'African Pied Wagtail', 'Northern Raven', 'Common Pheasant'], 'Spain': ['African Pied Wagtail', 'Common Swift'], 'Sweden': ['Eurasian Wryneck', 'European Nightjar', 'Tawny Owl', 'Black Woodpecker', 'Northern Lapwing', 'Northern Raven', 'Common Pheasant', 'Eurasian Oystercatcher', 'Red-throated Loon', 'Common Swift', 'Common Moorhen', 'Great Spotted Woodpecker', 'Eurasian Treecreeper', 'Eurasian Magpie', 'African Pied Wagtail', 'Stock Dove', 'Rook', 'River Warbler', 'Grey Partridge', 'European Golden Plover'], 'United Kingdom': ['Tawny Owl', 'European Honey Buzzard', 'European Greenfinch', 'Eurasian Coot', 'Sedge Warbler', 'Willow Warbler', 'Common Chiffchaff', 'Garden Warbler', 'Corn Bunting', 'Black-headed Gull', 'Common Linnet', 'Eurasian Treecreeper', 'Eurasian Blue Tit', 'Eurasian Wren', 'Common Blackbird', 'African Pied Wagtail', 'European Robin', 'Eurasian Reed Warbler', 'Common Wood Pigeon', 'Song Thrush', 'Barn Swallow', 'Eurasian Skylark', 'Willow Ptarmigan', 'Common Redstart', 'Common Whitethroat', 'Common Nightingale']}

#


option = st.selectbox('Select a country', list(country_dict.keys()))

st.markdown("<h5 style='text-align: center; color: white;   font-family: courier; '> Selected country is : "+option+"</h2>", unsafe_allow_html=True)



filename = 'svm_model.sav'

svm_model = pickle.load(open(filename, 'rb'))

uploaded_file = st.file_uploader("Choose a audio file")




frame_len = 22050*2


label_dict = {0: 'African Pied Wagtail', 1: 'Barn Swallow', 2: 'Black Woodpecker', 3: 'Black-headed Gull', 4: 'Carrion Crow', 5: 'Common Blackbird', 6: 'Common Chiffchaff', 7: 'Common Linnet', 8: 'Common Moorhen', 9: 'Common Nightingale', 10: 'Common Pheasant', 11: 'Common Redstart', 12: 'Common Swift', 13: 'Common Whitethroat', 14: 'Common Wood Pigeon', 15: 'Corn Bunting', 16: 'Eurasian Blue Tit', 17: 'Eurasian Bullfinch', 18: 'Eurasian Coot', 19: 'Eurasian Jay', 20: 'Eurasian Magpie', 21: 'Eurasian Oystercatcher', 22: 'Eurasian Reed Warbler', 23: 'Eurasian Skylark', 24: 'Eurasian Treecreeper', 25: 'Eurasian Wren', 26: 'Eurasian Wryneck', 27: 'European Golden Plover', 28: 'European Greenfinch', 29: 'European Herring Gull', 30: 'European Honey Buzzard', 31: 'European Nightjar', 32: 'European Robin', 33: 'European Turtle Dove', 34: 'Garden Warbler', 35: 'Great Spotted Woodpecker', 36: 'Grey Partridge', 37: 'Meadow Pipit', 38: 'Northern Lapwing', 39: 'Northern Raven', 40: 'Red-throated Loon', 41: 'River Warbler', 42: 'Rook', 43: 'Sedge Warbler', 44: 'Song Thrush', 45: 'Stock Dove', 46: 'Tawny Owl', 47: 'Western Jackdaw', 48: 'Willow Ptarmigan', 49: 'Willow Warbler', 50: 'Wood Sandpiper'}

reversed_dict = {value: key for key, value in label_dict.items()}

data = []



if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file , sr=16000)
    y = ((y-np.amin(y))*2)/(np.amax(y) - np.amin(y)) - 1
    org_len = len(y)
    intervals = librosa.effects.split(y, top_db=15, ref=np.max)
    intervals = intervals.tolist()
    y = (y.flatten()).tolist()
    nonsilent_y = []

    for p, q in intervals:
        nonsilent_y = nonsilent_y + y[p:q + 1]

    y = np.array(nonsilent_y)
    final_len = len(y)
    sil = org_len - final_len

    frame_len = 22050

    for j in range(0, len(y), int(frame_len)):

        frame = y[j:j + frame_len]
        if len(frame) < frame_len:
            frame = frame.tolist() + [0] * (frame_len - len(frame))
        frame = np.array(frame)
        ast = processor(frame, sampling_rate=sr, return_tensors="pt")
        ast = ast.to(device)
        with torch.no_grad():
            outputs = model(**ast)
        last_hidden_states = outputs.last_hidden_state.cpu().squeeze().mean(axis=0).numpy()
        data.append(last_hidden_states)
    data = np.array(data)
    y_pred = svm_model.decision_function(data)
    y_pred_country = country_dict[option]
    y_pred_country_label = [reversed_dict[i] for i in y_pred_country]
    y_pred_country_logits = []
    for j in range(len(y_pred)):
        temp = []
        for i in y_pred_country_label:
            temp.append(y_pred[j][i])
        y_pred_country_logits.append(temp)

    y_pred_country_logits = np.array(y_pred_country_logits)
    y_pred_country_logits_argmax = np.argmax(y_pred_country_logits, axis=1)
    max_probabilities = np.max(y_pred_country_logits, axis=1)

    class_probabilities = {}
    for i, prob in enumerate(max_probabilities):
        class_label = y_pred_country_logits_argmax[i]
        class_probabilities[class_label] = prob

    max_probable_class = max(class_probabilities, key=class_probabilities.get)
    prediction = y_pred_country[max_probable_class]

    html_code = """
        <div style='text-align: center; background-color: white; color: black; padding: 10px; border-radius: 20px;'>
            <h5 style='font-family: courier; color: black;'>Most probable bird in the audio is:</h5>
            <h3 style='weight:100 ;font-family: courier; color: black;'>{}</h3>
            <p style='font-family: courier; color: black;'>Probability: {:.2f}</p>
        </div> 
    """.format(prediction, class_probabilities[max_probable_class])

    st.markdown(html_code, unsafe_allow_html=True)
    #display bird image
    img_path = os.path.join(f'data/images/{prediction}/',   'Image_1' + '.jpg')

    st.write("")
    st.write("")
    st.image(img_path,  use_column_width=True)
    
    #Extract bird information from wikipedia
    bird_name = prediction
    bird_name = bird_name.replace(" ", "_")
    url = f"https://en.wikipedia.org/wiki/{bird_name}"
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = ""
    for paragraph in paragraphs:
        if len(paragraph.text.split()) > 10:
            text += paragraph.text
    text = text.split(".")
    
    outText = ""
    for i in range(3):
        outText += text[i] + "."

    st.markdown(f"<h5 style='opacity:.8; background-color: black; padding: 15px; border-radius: 20px; white: black; font-family: courier;'>{outText}</p>", unsafe_allow_html=True)

