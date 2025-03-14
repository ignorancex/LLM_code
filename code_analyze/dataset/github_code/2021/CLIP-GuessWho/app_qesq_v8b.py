# -*- coding: utf-8 -*-

## Used Imports
import os
import io
import zipfile
import random
import torch
import numpy as np
import streamlit as st
import clip
import gc
# import psutil

from io import BytesIO
from PIL import Image
from zipfile import ZipFile 
from streamlit import caching

## --------------- FUNCTIONS ---------------

def Predict_1_vs_0():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,1]>st.session_state['init_data']['image_current_probs'][i,0]:
            st.session_state['init_data']['image_current_predictions'].append(1)
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)
    
    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    del i
    
def Predict_0_vs_1():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,0]>st.session_state['init_data']['image_current_probs'][i,1]:
            st.session_state['init_data']['image_current_predictions'].append(1)
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    del i    
    
def Predict_1_vs_2():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if st.session_state['init_data']['image_current_probs'][i,1]>st.session_state['init_data']['image_current_probs'][i,2]:
            st.session_state['init_data']['image_current_predictions'].append(1)
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    del i
    
def Predict_bald():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
    
        if st.session_state['init_data']['image_current_probs'][i,1]>st.session_state['init_data']['image_current_probs'][i,2]:
            if st.session_state['init_data']['image_current_probs'][i,3]>st.session_state['init_data']['image_current_probs'][i,0]:
                st.session_state['init_data']['image_current_predictions'].append(1)
            else:
                st.session_state['init_data']['image_current_predictions'].append(0)
        else:
            if st.session_state['init_data']['image_current_probs'][i,4]>st.session_state['init_data']['image_current_probs'][i,0]:
                st.session_state['init_data']['image_current_predictions'].append(1)
            else:
                st.session_state['init_data']['image_current_predictions'].append(0)    

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    del i
    
def Predict_hair_color():
    st.session_state['init_data']['image_current_predictions']=[]
    for i in range(len(st.session_state['init_data']['image_current_probs'][:,0])):
        if np.argmax(st.session_state['init_data']['image_current_probs'][i,:])==0:
            st.session_state['init_data']['image_current_predictions'].append(1)        
        else:
            st.session_state['init_data']['image_current_predictions'].append(0)

    st.session_state['init_data']['image_current_predictions']=np.array(st.session_state['init_data']['image_current_predictions'])
    del i
                                    
def Token_img():
    st.session_state['init_data']['image_current_probs']=np.zeros((st.session_state['init_data']['n_images'],st.session_state['init_data']['n_tokens']))
    for i in range(st.session_state['init_data']['n_images']):
        CLIP_get_probs_only(i)
    del i
    
def CLIP_get_probs_only(i):
    img_proeprocessed = st.session_state['init_data']['clip_transform'](Image.fromarray(st.session_state['init_data']['current_image_files'][i])).unsqueeze(0).to(st.session_state['init_data']['clip_device'])
    img_features = st.session_state['init_data']['clip_model'].encode_image(img_proeprocessed)
    txt_features = st.session_state['init_data']['clip_model'].encode_text(st.session_state['init_data']['clip_text'])
    img_logits, img_logits_txt = st.session_state['init_data']['clip_model'](img_proeprocessed, st.session_state['init_data']['clip_text'])
    st.session_state['init_data']['image_current_probs'][i,:]=np.round(img_logits.detach().numpy()[0],2)
    del img_proeprocessed,img_features,txt_features,img_logits,img_logits_txt
   
def Image_discarding():
    for i in range(len(st.session_state['init_data']['current_images_discarted'])):
        if st.session_state['init_data']['current_images_discarted'][i]==0 and st.session_state['init_data']['image_current_predictions'][i]!=st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
            st.session_state['init_data']['current_images_discarted'][i]=1

    previous_names=st.session_state['init_data']['current_image_names']
    st.session_state['init_data']['current_image_names']=[]
    previous_files=st.session_state['init_data']['current_image_files']     
    st.session_state['init_data']['current_image_files']=[] 
    previous_predictions=st.session_state['init_data']['image_current_predictions'] 
    st.session_state['init_data']['image_current_predictions']=[]
    current_index=0
    new_index=0
    for i in range(st.session_state['init_data']['n_images']):
        if st.session_state['init_data']['current_images_discarted'][current_index]==0:
            st.session_state['init_data']['current_image_files'].append(previous_files[current_index])
            st.session_state['init_data']['current_image_names'].append(previous_names[current_index])
            st.session_state['init_data']['image_current_predictions'].append(previous_predictions[current_index])
            if current_index==st.session_state['init_data']['current_winner_index']:
                st.session_state['init_data']['current_winner_index']=new_index
                
            new_index+=1
            
        current_index+=1
            
    st.session_state['init_data']['n_images']=np.sum(st.session_state['init_data']['current_images_discarted']==0)                     
    st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names']) 
    st.session_state['init_data']['current_images_discarted']=np.zeros(st.session_state['init_data']['n_images'])
    del previous_names,previous_files,previous_predictions,current_index,new_index,i
      
def Show_images():
    st.session_state['init_data']['highlighted_images']=[]     
    for current_index in range(st.session_state['init_data']['n_images']):
        if st.session_state['init_data']['show_results']:
            current_line_width=4
            if st.session_state['init_data']['image_current_predictions'][current_index]==st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                current_color=np.array([0,255,0])
            else:
                current_color=np.array([255,0,0]) 
        else:
            current_line_width=2
            current_color=np.zeros(3)  
        image_size=240
        w,h,c = np.shape(st.session_state['init_data']['current_image_files'][current_index])
        images_separation=image_size-w-current_line_width*2
        image_highlighted=np.zeros([h+current_line_width*2,image_size,c])+255
        image_highlighted[current_line_width:w+current_line_width,current_line_width:w+current_line_width,:]=st.session_state['init_data']['current_image_files'][current_index]
        image_highlighted[:current_line_width,:w+2*current_line_width,:]=current_color
        image_highlighted[w+current_line_width:,:w+2*current_line_width,:]=current_color
        image_highlighted[:,w+current_line_width:w+2*current_line_width,:]=current_color
        image_highlighted[:,:current_line_width,:]=current_color
        st.session_state['init_data']['highlighted_images'].append(image_highlighted)
    
    ## result to array      
    st.session_state['init_data']['highlighted_images']=np.array(st.session_state['init_data']['highlighted_images'])/255
    del image_highlighted,current_index,current_line_width,current_color,image_size,w,h,c

def Load_Images_Randomly(n_images):
    st.session_state['init_data']['current_image_files']=[]
    st.session_state['init_data']['current_image_names']=[]
    image_index=[]
        
    archive = zipfile.ZipFile('guess_who_images.zip', 'r')
    listOfFileNames = archive.namelist()        
    image_index_all=list(range(len(listOfFileNames)))
    image_index.append(random.choice(image_index_all))
    image_index_all.remove(image_index[0])
    current_index=1
    while len(image_index)<n_images:
        image_index.append(random.choice(image_index_all))
        image_index_all.remove(image_index[current_index])
        current_index+=1
        
   # Iterate over the file names
    for current_index in image_index:
        image_current_path=listOfFileNames[current_index]
        st.session_state['init_data']['current_image_files'].append(np.array(Image.open(BytesIO(archive.read(image_current_path)))))
        st.session_state['init_data']['current_image_names'].append(image_current_path[-10:-4])
                
    st.session_state['init_data']['current_image_names']=np.array(st.session_state['init_data']['current_image_names'])
    del image_index,archive,listOfFileNames,image_index_all,current_index,image_current_path
    
def Token_process_query():
    ## Tokenization process
    st.session_state['init_data']['n_tokens']=len(st.session_state['init_data']['current_querys'])
    st.session_state['init_data']['clip_device'] = "cuda" if torch.cuda.is_available() else "cpu"
    st.session_state['init_data']['clip_model'], st.session_state['init_data']['clip_transform'] = clip.load("ViT-B/32", device=st.session_state['init_data']['clip_device'], jit=False)
    st.session_state['init_data']['clip_text'] = clip.tokenize(st.session_state['init_data']['current_querys']).to(st.session_state['init_data']['clip_device'])
    
def Show_Info():
    st.sidebar.markdown('#### Questions List:')
    st.sidebar.write(st.session_state['init_data']['feature_questions'])
    # st.sidebar.write(st.session_state['init_data'])

# ---------------   CACHE   ---------------

# @st.cache(allow_output_mutation=True,max_entries=2,ttl=3600) 
def load_data(total_images_number):
    st.session_state['init_data']={
        'images_selected':False,
        'button_question':False,
        'button_query1':False,
        'button_query2':False,
        'button_winner':False,
        'show_results':False,
        'start_game':False,
        'finished_game':False,
        'reload_game':False,
        'award':100,
        'token_type':0,
        'selected_feature':'Ask a Question',
        'questions_index':0,
        'selected_question':'Are you a MAN?',
        'first_question':'Are you a MAN?',
        'user_input':'A picture of a person',
        'user_input_querys1':'A picture of a person',
        'user_input_querys2':'A picture of a person',
        'current_querys':['A picture of a person','A picture of a person'],
        'selected_winner':'Winner not selected',
        'current_winner_index':-1,
        'N_images':total_images_number,
        'n_images':total_images_number,
        'n_tokens':2,
        'current_image_files':[],
        'highlighted_images':[],
        'current_images_discarted':np.zeros((total_images_number)),
        'winner_options':[],
        'current_image_names':[],
        'highlighted_image_names':[],
        'clip_tokens':['A picture of a person','A picture of a person'],
        'clip_device':0,
        'clip_model':0, 
        'clip_transform':0, 
        'clip_text':0,
        'path_info':'D:/Datasets/Celeba/',
        'path_imgs':'D:/Datasets/Celeba/img_celeba/',
        'querys_list':['A picture of a man', 'A picture of a woman', 'A picture of an attractive person', 'A picture of a young person', 
            'A picture of a person with receding hairline', 'A picture of a chubby person ', 'A picture of a person who is smiling', 'A picture of a bald person',
            'A picture of a person with black hair', 'A picture of a person with brown hair', 'A picture of a person with blond hair', 'A picture of a person with red hair', 
            'A picture of a person with gray hair', 'A picture of a person with straight hair', 'A picture of a person with wavy hair', 
            'A picture of a person who does not wear a beard', 'A picture of a person with mustache', 'A picture of a person with sideburns', 
            'A picture of a person with goatee', 'A picture of a person with heavy makeup', 'A picture of a person with eyeglasses ',             
            'A picture of a person with bushy eyebrows', 'A picture of a person with a double chin', 
            'A picture of a person with high cheekbones', 'A picture of a person with slightly open mouth', 
            'A picture of a person with narrow eyes', 'A picture of a person with an oval face', 
            'A picture of a person wiht pale skin', 'A picture of a person with pointy nose', 'A picture of a person with rosy cheeks', 
            "A picture of a person with five o'clock shadow", 'A picture of a person with arched eyebrows', 'A picture of a person with bags under the eyes', 
            'A picture of a person with bangs', 'A picture of a person with big lips', 'A picture of a person with big nose',            
            'A picture of a person with earrings', 'A picture of a person with hat', 
            'A picture of a person with lipstick', 'A picture of a person with necklace', 
            'A picture of a person with necktie', 'A blurry picture of a person'
            ],
        'feature_questions':['Are you a MAN?', 'Are you a WOMAN?', 'Are you an ATTRACTIVE person?', 'Are you YOUNG?',
                    'Are you a person with RECEDING HAIRLINES?', 'Are you a CHUBBY person?', 'Are you SMILING?','Are you BALD?', 
                    'Do you have BLACK HAIR?', 'Do you have BROWN HAIR?', 'Do you have BLOND HAIR?', 'Do you have RED HAIR?',
                    'Do you have GRAY HAIR?', 'Do you have STRAIGHT HAIR?', 'Do you have WAVY HAIR?',
                    'Do you have a BEARD?', 'Do you have a MUSTACHE?', 'Do you have SIDEBURNS?',
                    'Do you have a GOATEE?', 'Do you wear HEAVY MAKEUP?', 'Do you wear EYEGLASSES?',
                    'Do you have BUSHY EYEBROWS?', 'Do you have a DOUBLE CHIN?', 
                    'Do you have a high CHEECKBONES?', 'Do you have SLIGHTLY OPEN MOUTH?', 
                    'Do you have NARROWED EYES?', 'Do you have an OVAL FACE?', 
                    'Do you have PALE SKIN?', 'Do you have a POINTY NOSE?', 'Do you have ROSY CHEEKS?', 
                    "Do you have FIVE O'CLOCK SHADOW?", 'Do you have ARCHED EYEBROWS?', 'Do you have BUGS UNDER your EYES?', 
                    'Do you have BANGS?', 'Do you have a BIG LIPS?', 'Do you have a BIG NOSE?',
                    'Are you wearing EARRINGS?', 'Are you wearing a HAT?', 
                    'Are you wearing LIPSTICK?', 'Are you wearing NECKLACE?', 
                    'Are you wearing NECKTIE?', 'Is your image BLURRY?'],
        'previous_discarding_images_number':0,
        'function_predict':Predict_0_vs_1,
        'image_current_probs':np.zeros((total_images_number,2)),
        'image_current_predictions':np.zeros((total_images_number))+2}
    
    Load_Images_Randomly(total_images_number)
    Token_process_query()
    del total_images_number
    

st.set_page_config(
    layout="wide",
    page_icon='Logo DIMAI.png',
    page_title='QuienEsQuien',
    initial_sidebar_state="collapsed"
)

## --------------- PROGRAMA ---------------

## SIDEBAR
st.sidebar.markdown('# OPTIONS PANEL')

## Reset App APP
Reset_App = st.sidebar.button('RESET GAME', key='Reset_App')

## Images number
st.sidebar.markdown('# Number of images')
Total_Images_Number=st.sidebar.number_input('Select the number of images of the game and press "RESET GAME"', min_value=5, max_value=40, value=20, 
                                                                    step=1, format='%d', key='Total_Images_Number', help=None)

## INITIALIZATIONS
 
Feature_Options=['Ask a Question', 'Create your own query', 'Create your own 2 querys','Select a Winner']

## Load data to play
if 'init_data' not in st.session_state:
    load_data(20)
 
## Title
if st.session_state['init_data']['finished_game']:
    st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align:left; float:left; color:blue; margin:0px;'>Guess Who?</h1><h2 style='text-align:right; float:right; color:gray; margin:0px;'>score: "+ str(st.session_state['init_data']['award'])+"</h2>", unsafe_allow_html=True)

## GAME
if Reset_App:
    load_data(Total_Images_Number)
    Restart_App = st.button('GO TO IMAGES SELECTION TO START A NEW GAME', key='Restart_App')
else:                    
    ## FINISHED GAME BUTTON TO RELOAD GAME
    if st.session_state['init_data']['finished_game']:
        Restart_App = st.button('GO TO IMAGES SELECTION TO START NEW GAME', key='Restart_App')
        if st.session_state['init_data']['award']==1 or st.session_state['init_data']['award']==-1:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINT !!!</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>¡¡¡ FINISHED WITH</h1><h1 style='text-align:left; float:left; color:green; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>"+str(st.session_state['init_data']['award'])+"</h1><h1 style='text-align:left; float:left; color:black; margin:0px;'>POINTS !!!</h1>", unsafe_allow_html=True)
    
    else:
        st.session_state['init_data']['images_selected']=False
        
        ## INITIALIZATION (SELECT FIGURES)
        if not st.session_state['init_data']['start_game']:
            ## Text - select Celeba images
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Choose the images you like.</h2>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align:left; float:left; color:gray; margin:0px;'>Press the button to randomly modify the selected images.</h3>", unsafe_allow_html=True)
            
            ## Button - randomly change Celeba images
            Random_Images = st.button('CHANGE IMAGES', key='Random_Images')
            if Random_Images:
                Load_Images_Randomly(st.session_state['init_data']['N_images'])
                st.session_state['init_data']['winner_options']=st.session_state['init_data']['current_image_names']
                
            ## Button - start game
            st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to start the game.</h2>", unsafe_allow_html=True)
            Use_Images = st.button('START GAME', key='Use_Images')
            
            if Use_Images:
                ## Choose winner and start game
                st.session_state['init_data']['current_winner_index']=random.choice(list(range(0,st.session_state['init_data']['N_images'])))
                st.session_state['init_data']['start_game']=True
                st.session_state['init_data']['images_selected']=True
                
        ## RUN GAME
        if st.session_state['init_data']['start_game']:
        
            ## Text - Select query type (game mode)
            if st.session_state['init_data']['images_selected']:
                st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>3. Select a type of Query to play.</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>1. Select a type of Query to play.</h2>", unsafe_allow_html=True)
            
            ## SelectBox - Select query type (game mode)
            Selected_Feature=st.selectbox('Ask a question from a list, create your query or select a winner:', Feature_Options, 
                                                   index=0, 
                                                   key='selected_feature', help=None)
            st.session_state['init_data']['selected_feature']=Selected_Feature  # Save Info
                
            ## SHOW ELEMENTS - QUESTIONS MODE
            if Selected_Feature=='Ask a Question':
                ## Game mode id
                st.session_state['init_data']['token_type']=0

                ## Text - Questions mode
                st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Question from the list.</h3>", unsafe_allow_html=True)
                
                ## SelectBox - Select question
                Selected_Question=st.selectbox('Suggested questions:', st.session_state['init_data']['feature_questions'], 
                                                   index=0,
                                                   key='Selected_Question', help=None)
                st.session_state['init_data']['selected_question']=Selected_Question  # Save Info
                
                ## Current question index
                if Selected_Question not in st.session_state['init_data']['feature_questions']:
                    Selected_Question=st.session_state['init_data']['feature_questions'][0]
                
                st.session_state['init_data']['questions_index']=st.session_state['init_data']['feature_questions'].index(Selected_Question)
                   
                ## Text - Show current question
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Question: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+Selected_Question+"</h3>", unsafe_allow_html=True)
                
                ## Button - Use current question
                Check_Question = st.button('USE THIS QUESTION', key='Check_Question')
                st.session_state['init_data']['button_question']=Check_Question  # Save Info
                
                ## Check current question
                if st.session_state['init_data']['show_results']:
                    st.session_state['init_data']['show_results']=False
                    
                else:
                    if Check_Question:
                        if Selected_Question=='Are you bald?':
                            st.session_state['init_data']['current_querys']=['A picture of a person','A picture of a man','A picture of a woman',
                                                                        'A picture of a yes bald man','A picture of a bald person']
                            st.session_state['init_data']['function_predict']=Predict_bald
                            
                        elif Selected_Question=='Do you have BLACK HAIR?':
                            st.session_state['init_data']['current_querys']=['A picture of a person who is black-haired',
                                                                        'A picture of a person who is tawny-haired',
                                                                        'A picture of a person who is blond-haired',
                                                                        'A picture of a person who is gray-haired',
                                                                        'A picture of a person who is red-haired',
                                                                        'A picture of a person who is totally bald']
                            st.session_state['init_data']['function_predict']=Predict_hair_color

                        elif Selected_Question=='Do you have BROWN HAIR?':
                            st.session_state['init_data']['current_querys']=['A picture of a person who is tawny-haired',
                                                                        'A picture of a person who is black-haired',
                                                                        'A picture of a person who is blond-haired',
                                                                        'A picture of a person who is gray-haired',
                                                                        'A picture of a person who is red-haired',
                                                                        'A picture of a person who is totally bald']
                            st.session_state['init_data']['function_predict']=Predict_hair_color

                        elif Selected_Question=='Do you have BLOND HAIR?':
                            st.session_state['init_data']['current_querys']=['A picture of a person who is blond-haired',
                                                                        'A picture of a person who is tawny-haired',
                                                                        'A picture of a person who is black-haired',
                                                                        'A picture of a person who is gray-haired',
                                                                        'A picture of a person who is red-haired',
                                                                        'A picture of a person who is totally bald']
                            st.session_state['init_data']['function_predict']=Predict_hair_color
                            
                        elif Selected_Question=='Do you have RED HAIR?':
                            st.session_state['init_data']['current_querys']=['A picture of a person who is red-haired',
                                                                        'A picture of a person who is tawny-haired',
                                                                        'A picture of a person who is blond-haired',
                                                                        'A picture of a person who is gray-haired',
                                                                        'A picture of a person who is black-haired',
                                                                        'A picture of a person who is totally bald']
                            st.session_state['init_data']['function_predict']=Predict_hair_color
                            
                        elif Selected_Question=='Do you have GRAY HAIR?':
                            st.session_state['init_data']['current_querys']=['A picture of a person who is gray-haired',
                                                                        'A picture of a person who is tawny-haired',
                                                                        'A picture of a person who is blond-haired',
                                                                        'A picture of a person who is black-haired',
                                                                        'A picture of a person who is red-haired',
                                                                        'A picture of a person who is totally bald']
                            st.session_state['init_data']['function_predict']=Predict_hair_color
                            
                        elif Selected_Question=='Are you a man?':
                            st.session_state['init_data']['current_querys']=['A picture of a man','A picture of a woman']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1    
                            
                        elif Selected_Question=='Are you a woman?':
                            st.session_state['init_data']['current_querys']=['A picture of a woman','A picture of a man']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1         
                            
                        elif Selected_Question=='Do you have a beard?':
                            st.session_state['init_data']['current_querys']=['A picture of a person with beard','A picture of a person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1
                            
                        elif Selected_Question=='Are you YOUNG?':
                            st.session_state['init_data']['current_querys']=['A picture of a young person','A picture of an aged person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1

                        elif  not st.session_state['init_data']['show_results']:
                            st.session_state['init_data']['current_querys']=[st.session_state['init_data']['querys_list'][st.session_state['init_data']['questions_index']],'A picture of a person']
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1
                    
                        Token_process_query()
                        Token_img()
                        st.session_state['init_data']['function_predict']()
                        st.session_state['init_data']['show_results']=True
                        
            ## SHOW ELEMENTS - 1 QUERY MOD
            if Selected_Feature=='Create your own query':              
                
                ## Game mode id
                st.session_state['init_data']['token_type']=-1

                ## Text - Query mode
                st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own query and press the button.</h3>", unsafe_allow_html=True)
                
                ## TextInput - Select query
                User_Input = st.text_input('It is recommended to use a text like: "A picture of a ... person" or "A picture of a person ..." (CLIP will check -> "Your query"  vs  "A picture of a person" )', 'A picture of a person', key='User_Input', help=None)
                st.session_state['init_data']['user_input']=User_Input  # Save Info

                ## Text - Show current query
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Query: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input+"</h3>", unsafe_allow_html=True)
                
                ## Button - Use current query
                Check_Query = st.button('USE MY OWN QUERY', key='Check_Query')
                st.session_state['init_data']['button_query1']=Check_Query  # Save Info
                
                ## Check current question            
                if st.session_state['init_data']['show_results']:
                    st.session_state['init_data']['show_results']=False
                else:
                    if Check_Query:
                        if User_Input!='A picture of a person':
                            st.session_state['init_data']['current_querys']=['A Picture of a person',User_Input]
                            st.session_state['init_data']['function_predict']=Predict_1_vs_0
                            Token_process_query()
                            Token_img()
                            st.session_state['init_data']['function_predict']()
                            st.session_state['init_data']['show_results']=True
                            
                        else:
                            st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your query must be different of 'A picture of a person'.</h3>", unsafe_allow_html=True)
                    
            ## SHOW ELEMENTS - 2 QUERYS MODE
            if Selected_Feature=='Create your own 2 querys':
                
                ## Game mode id
                st.session_state['init_data']['token_type']=-2

                ## Text - Querys mode
                st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Write your own querys by introducing 2 opposite descriptions.</h3>", unsafe_allow_html=True)
                
                ## SelectBox - Select querys
                User_Input_Querys1 = st.text_input('Write your "True" query:', 'A picture of a person',
                                                            key='User_Input_Querys1', help=None)
                User_Input_Querys2 = st.text_input('Write your "False" query:', 'A picture of a person',
                                                            key='User_Input_Querys2', help=None)
                st.session_state['init_data']['user_input_querys1']=User_Input_Querys1  # Save Info
                st.session_state['init_data']['user_input_querys2']=User_Input_Querys2  # Save Info
                                 
                ## Text - Show current querys
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Querys: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+User_Input_Querys1+' vs '+User_Input_Querys2+"</h3>", unsafe_allow_html=True)
                
                ## Button - Use current querys
                Check_Querys = st.button('USE MY OWN QUERYS', key='Check_Querys')
                st.session_state['init_data']['button_query2']=Check_Querys  # Save Info
                
                ## Check current querys
                if st.session_state['init_data']['show_results']:
                    st.session_state['init_data']['show_results']=False
                else:
                    if Check_Querys:
                        if User_Input_Querys1!=User_Input_Querys2:
                            st.session_state['init_data']['current_querys']=[User_Input_Querys1,User_Input_Querys2]     
                            st.session_state['init_data']['function_predict']=Predict_0_vs_1
                            Token_process_query()
                            Token_img()
                            st.session_state['init_data']['function_predict']()
                            st.session_state['init_data']['show_results']=True
                            
                        else:
                            st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your two own querys must be different.</h3>", unsafe_allow_html=True)

            ## SHOW ELEMENTS - WINNER MODE
            if Selected_Feature=='Select a Winner': 
                
                ## Game mode id
                st.session_state['init_data']['token_type']=-3

                ## Text - Winner mode
                st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Select a Winner picture name.</h3>", unsafe_allow_html=True)
                
                ## SelectBox - Select winner
                # st.session_state['init_data']['winner_options']=['Winner not selected']
                # st.session_state['init_data']['winner_options'].extend(st.session_state['init_data']['current_image_names'])
                
                # if st.session_state['init_data']['selected_winner'] not in st.session_state['init_data']['winner_options']:
                    # st.write(st.session_state['init_data']['selected_winner'])
                    # st.write(st.session_state['init_data']['winner_options'])
                    
                Selected_Winner=st.selectbox('If you are inspired, Select a Winner image directly:', st.session_state['init_data']['winner_options'],
                                                index=0, key='Selected_Winner', help=None)
                st.session_state['init_data']['selected_winner']=Selected_Winner  # Save Info
                
                ## Text - Show current winner
                st.markdown("<h3 style='text-align:center; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>Current Winner: </h3><h3 style='text-align:left; float:center; color:green; margin:0px;'>"+Selected_Winner+"</h3>", unsafe_allow_html=True)
                
                ## Button - Use current winner
                Check_Winner = st.button('CHECK THIS WINNER', key='Check_Winner')
                st.session_state['init_data']['button_winner']=Check_Winner  # Save Info
                                                    
                ## Check current winner
                if st.session_state['init_data']['show_results']:
                    st.session_state['init_data']['show_results']=False
                else:
                    if Check_Winner:
                        if Selected_Winner in st.session_state['init_data']['current_image_names']:
                            st.session_state['init_data']['selected_winner_index']=np.where(Selected_Winner==st.session_state['init_data']['current_image_names'])[0]
                            st.session_state['init_data']['image_current_predictions']=np.zeros(st.session_state['init_data']['n_images'])
                            st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['selected_winner_index']]=1    
                            st.session_state['init_data']['show_results']=True
                            
                            # Delete Winner elements   
                            # del st.session_state['Selected_Winner']                                    
                        else:
                            st.markdown("<h3 style='text-align:left; float:left; color:red; margin-left:0px; margin-right:0px; margin-top:15px; margin-bottom:-10px;'>Your must select a not discarded picture.</h3>", unsafe_allow_html=True)


            ## ACTIONS SHOWING RESULTS
            if st.session_state['init_data']['show_results']:
            
                ## Continue game
                if not np.sum(st.session_state['init_data']['current_images_discarted']==0)==1:
                    if st.session_state['init_data']['images_selected']: 
                        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>4. Press the button to continue.</h2>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h2 style='text-align:left; float:left; color:black; margin:0px;'>2. Press the button to continue.</h2>", unsafe_allow_html=True)
                    
                    ## Button - Next query
                    Next_Query=st.button('NEXT QUERY', key='Next_Query')

                ## Show current results
                if st.session_state['init_data']['token_type']==0:
                    if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>YES</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['selected_question']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>NO</h3>", unsafe_allow_html=True)
                        
                if st.session_state['init_data']['token_type']==-1:
                    if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>TRUE</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>"+st.session_state['init_data']['user_input']+"</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>FALSE</h3>", unsafe_allow_html=True)
                        
                if st.session_state['init_data']['token_type']==-2:
                    if st.session_state['init_data']['image_current_predictions'][st.session_state['init_data']['current_winner_index']]:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys1']+"</h3>", unsafe_allow_html=True)
                    else:
                        st.markdown("<h3 style='text-align:left; float:left; color:blue; margin-left:0px; margin-right:25px; margin-top:0px; margin-bottom:0px;'>The most accurate query is:</h3><h3 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['user_input_querys2']+"</h3>", unsafe_allow_html=True)
                  
                if st.session_state['init_data']['token_type']==-3:
                    if not st.session_state['init_data']['selected_winner']==st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]:
                        st.markdown("<h3 style='text-align:left; float:left; color:gray; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>The winner picture is not:</h3><h3 style='text-align:left; float:center; color:red; margin:0px;'>"+st.session_state['init_data']['selected_winner']+"</h3>", unsafe_allow_html=True)

     
    ## CREATE IMAGES TO SHOW
    Show_images()
    st.session_state['init_data']['highlighted_image_names']=st.session_state['init_data']['current_image_names']


   ## APPLY DISCARDING
    if st.session_state['init_data']['show_results']:        
        st.session_state['init_data']['previous_discarding_images_number']=st.session_state['init_data']['n_images']
        Image_discarding()
                   
        ## penalty - game not finished                                                       
        if st.session_state['init_data']['n_images']>1:
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-st.session_state['init_data']['n_images']
        
        ## penalty - "select winner" option used
        if st.session_state['init_data']['token_type']==-3:   
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-1-(st.session_state['init_data']['N_images']-st.session_state['init_data']['previous_discarding_images_number'])

        ## penalty - no image is discarted
        if st.session_state['init_data']['previous_discarding_images_number']==st.session_state['init_data']['n_images']:   
            st.session_state['init_data']['award']=st.session_state['init_data']['award']-5


    ## SHOW FINAL RESULTS
    if st.session_state['init_data']['finished_game']:
        st.session_state['init_data']['reload_game']=True

    else:
        ## CHECK FINISHED GAME 
        if np.sum(st.session_state['init_data']['current_images_discarted']==0)==1 and not st.session_state['init_data']['finished_game']:
            st.session_state['init_data']['finished_game']=True
            st.markdown("<h1 style='text-align:left; float:left; color:black; margin-left:0px; margin-right:15px; margin-top:0px; margin-bottom:0px;'>You found the Winner picture:</h1><h1 style='text-align:left; float:left; color:green; margin:0px;'>"+st.session_state['init_data']['current_image_names'][st.session_state['init_data']['current_winner_index']]+"</h1>", unsafe_allow_html=True)
            Finsih_Game = st.button('FINISH GAME', key='Finsih_Game')


    ## SHOW CURRENT IMAGES
    st.image(st.session_state['init_data']['highlighted_images'], use_column_width=False, caption=st.session_state['init_data']['highlighted_image_names'])
    

    ## RELOAD GAME
    if st.session_state['init_data']['reload_game']:
        load_data(st.session_state['init_data']['N_images']) 
        
        
## SHOW EXTRA INFO
Show_Info() 
        

## CLEAR RESOURCES
gc.collect()
caching.clear_cache()
torch.cuda.empty_cache()

    
## gives a single float value
# st.sidebar.write(psutil.cpu_percent())

## gives an object with many fields
# st.sidebar.write(psutil.virtual_memory())
