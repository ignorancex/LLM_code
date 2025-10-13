# coding=utf-8
import time
import sys
import os
import glob
import csv
import datetime
import random
from psychopy import prefs
prefs.general['audioLib'] = ['pyo']
from psychopy import visual,event,core,gui
from fractions import Fraction
import pyaudio
import wave
import json

def get_stim_info(file_name):
# read stimulus information stored in same folder as file_name, with a .json extension
# returns a list of values    
    info_file_name =  os.path.splitext(file_name)[0]+'.json'
    #correct the folder name for the metadata
    info_file_name = info_file_name.replace("audio", "bpf")
    with open(info_file_name,'r', newline='', encoding='utf-8') as file:
      info = json.load(file)
    return info

def enblock(x, n_stims):
    # generator to cut a list of stims into blocks of n_stims
    # returns all complete blocks
    # fixme: this returns [] if n_stims > len(x)
    for i in range(len(x)//n_stims):
        start = n_stims*i
        end = n_stims*(i+1)
        yield x[start:end]
    
def generate_trial_files(subject_number,condition, n_blocks=1,n_stims=100, practice=False):
# generates n_block trial files per subject
# each block contains n_stim trials, randomized from folder which name is inferred from subject_number
# returns an array of n_block file names
    seed = time.getTime()
    random.seed(seed)

    # test if subj folder exists
    stim_folder = "sounds/%s"%(condition)

    folders_list = [x[0] for x in os.walk(stim_folder)]
    sound_folders = []
    for folder in folders_list:
        if "audio" in folder:
            sound_folders.append(folder)

    first_word = [os.path.basename(x) for x in glob.glob(sound_folders[0]+"/*.wav")]
    second_word = [os.path.basename(x) for x in glob.glob(sound_folders[1]+"/*.wav")]

    # trials consist of two random files, one from the first half, and one from the second half of the stimulus list
    # write trials by blocks of n_stims
    trial_files = []

    for block in range(n_blocks):
        practice_tag = 'PRACTICE' if practice else ''
        trial_file = 'trials/' + str(subject_number) + '_' + condition + '_' + practice_tag + '_BLOCK_'+ str(block) + '_' + date.strftime('%y%m%d_%H.%M')+'.csv'
        
        trial_files.append(trial_file)
            # create a randomized ordering of the stimuli
        stim_list = []
        for i in range(n_stims):
          # right now it chooses random from the whole list, there can be repeats, it this ok?
          if (block == 0 or block == 1):
            print(f"one : {block}")
            stim_list.append([f'{sound_folders[0]}/{random.choice(first_word)}', f'{sound_folders[0]}/{random.choice(first_word)}'])
          elif (block == 2 or block == 3):
            print(f"two : {block} ")
            stim_list.append([f'{sound_folders[1]}/{random.choice(second_word)}', f'{sound_folders[1]}/{random.choice(second_word)}'])
          else:
            print(f"oops : {block}")

        with open(trial_file, 'w+', newline='', encoding='utf-8') as file :
            # each trial is stored as a row in a csv file, with format: 
            # StimA,MeanA,PA1,PA2,PA3,PA4,PA5,PA6,PA7,StimB,MeanB,PB1,PB2,PB3,PB4,PB5,PB6,PB7
            # where Mean and P1...P7 are CLEESE parameters found in .txt files stored alongside de .wav stims
            # write header
            writer = csv.writer(file)
            for pair in stim_list:
              writer.writerow(pair)

    return trial_files

def read_trials(trial_file): 
# read all trials in a block of trial, stored as a CSV trial file
    with open(trial_file, 'r',newline='', encoding='utf-8') as fid :
        reader = csv.reader(fid)
        trials = list(reader)
    return trials #trim header

def generate_result_file(subject_number, condition):

    result_file = 'results/results_subj'+str(subject_number)+'_'+condition+'_'+date.strftime('%y%m%d_%H.%M')+'.csv'        
    result_headers = ['subj','sex','age', 'english', 'french', 'date',
                      'condition', 'practice','block','trial',
                      'stim', 'stim_num', 'stim_order', 'eq','pitch','stretch','response','rt', 'confidence', 'confidence_rt']
    with open(result_file, 'w+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(result_headers)
    return result_file

def show_text_and_wait(file_name = None, message = None):
    event.clearEvents()
    if message is None:
        #with codecs.open (file_name, "r", "utf-8") as file :
        with open (file_name, "r", newline='', encoding='utf-8') as file :
            message = file.read()
    text_object = visual.TextStim(win, text = message, color = 'black')
    text_object.height = 0.05
    text_object.draw()
    win.flip()
    while True :
        if len(event.getKeys()) > 0: 
            core.wait(0.2)
            break
        event.clearEvents()
        core.wait(0.2)
        text_object.draw()
        win.flip()

def show_text(file_name = None, message = None):
    if message is None:
        #with codecs.open (file_name, "r", "utf-8") as file :
        with open (file_name, "r",newline='', encoding='utf-8') as file :
            message = file.read()
    text_object = visual.TextStim(win, text = message, color = 'black')
    text_object.height = 0.05
    text_object.draw()
    win.flip()

def show_text_with_sounds_(file_name, play_keys, sounds):
    # show a text with keys to trigger sounds
    event.clearEvents()
    with open (file_name, "r", newline='',  encoding='utf-8') as file :
        message = file.read()
    text_object = visual.TextStim(win, height = 0.05, text = message, color = 'black')
    text_object.draw()
    win.flip()
    while True :
        response_key = event.getKeys()
        if response_key==['space']: 
            event.clearEvents()
            break  
        for play_key, sound in zip(play_keys, sounds): 
            if response_key == [play_key]:
                event.clearEvents()
                play_sound(sound)
                break
        else:      
            event.clearEvents()
            core.wait(0.2)
        
        text_object.draw()
        win.flip()



def update_sound_trial_gui(block): 
    play_instruction.draw()
    play_icon.draw()
    if (block == 0  or block == 1):
      response_instruction_1.draw()
    if block == 2:
      response_instruction_2.draw()
    if block == 3:
      response_instruction_3.draw()
    if block == 4:
      response_instruction_4.draw()
    play_icon.draw()
    for response_label in response_labels: response_label.draw()
    for response_checkbox in response_checkboxes: response_checkbox.draw()
    win.flip()

def update_confidence_trial_gui(): 
    confidence_instruction.draw()
    confidence_scale.draw()
    win.flip()

def play_sound(sound):
        #play sound
        audio = pyaudio.PyAudio()
        #sr,wave = wav.read(fileName)
        wf = wave.open(sound)
        def play_audio_callback(in_data, frame_count, time_info,status):
            data = wf.readframes(frame_count)
            return (data, pyaudio.paContinue)
        #define data stream for playing audio and start it
        output_stream = audio.open(format = audio.get_format_from_width(wf.getsampwidth()),
                                    channels = wf.getnchannels(),
                                    rate = wf.getframerate(),
                                    output = True,
                                    stream_callback = play_audio_callback)
        output_stream.start_stream()
        while output_stream.is_active():
            core.wait(0.01)
            continue 


###########################################################################################
# Experiment parameters

FREN_1_PARAMS = { 'n_blocks':4, 'n_stims': 2, 'repeat_for_internal_noise':False,
                   'practice':True, 'n_practice_trials': 3,
                   'isi': .5, 'pause_duration': 5,
                   'questions': { '1' : u'Quelle prononciation sonne le plus comme bruyant ?',
                                 '2' : u'Quelle prononciation sonne le plus comme brillant ?',
                                 '3' : u'Quelle prononciation sonne le plus comme set ?',
                                 '4' : u'Quelle prononciation sonne le plus comme sat ?'},
                   'instruction_texts': ['text/intro_french.txt'],
                   'practice_start_text': 'text/practice_french.txt',
                   'practice_end_text': 'text/end_practice_french.txt',
                   'pause_start_text': 'text/pause1_french.txt',
                   'pause_end_text': 'text/pause0_french.txt',
                   'end_text': 'text/end_french.txt',
                   'give_sound_model' : False}

FREN_2_PARAMS = {'n_blocks':4, 'n_stims': 2, 'repeat_for_internal_noise':False,
                   'practice':True, 'n_practice_trials': 3,
                   'isi': .5, 'pause_duration': 5,
                   'questions': { '1' : u'Quelle prononciation sonne le plus comme fou ?',
                                 '2' : u'Quelle prononciation sonne le plus comme fût ?',
                                 '3' : u'Quelle prononciation sonne le plus comme bit ?',
                                 '4' : u'Quelle prononciation sonne le plus comme beat ?'},
                   'instruction_texts': ['text/intro_french.txt'],
                   'practice_start_text': 'text/practice_french.txt',
                   'practice_end_text': 'text/end_practice_french.txt',
                   'pause_start_text': 'text/pause1_french.txt',
                   'pause_end_text': 'text/pause0_french.txt',
                   'end_text': 'text/end_french.txt',
                   'give_sound_model' : False}



ENG_2_PARAMS = {'n_blocks':4, 'n_stims': 2, 'repeat_for_internal_noise':False,
                   'practice':True, 'n_practice_trials': 3,
                   'isi': .5, 'pause_duration': 5,
                   'questions': { '1' : u'Which pronounciation sounds most like fou ?',
                                 '2' : u'Which pronounciation sounds most like fût ?',
                                 '3' : u'Which pronounciation sounds most like bit ?',
                                 '4' : u'Which pronounciation sounds most like beat ?'},
                   'instruction_texts': ['text/intro_eng.txt'],
                   'practice_start_text': 'text/practice_eng.txt',
                   'practice_end_text': 'text/end_practice_eng.txt',
                   'pause_start_text': 'text/pause1_eng.txt',
                   'pause_end_text': 'text/pause0_eng.txt',
                   'end_text': 'text/end_eng.txt',
                   'give_sound_model' : False}

ENG_1_PARAMS = {'n_blocks':4, 'n_stims': 2, 'repeat_for_internal_noise':False,
                   'practice':True, 'n_practice_trials': 3,
                   'isi': .5, 'pause_duration': 5,
                   'questions': { '1' : u'Which pronounciation sounds most like bruyant ?',
                                 '2' : u'Which pronounciation sounds most like brillant ?',
                                 '3' : u'Which pronounciation sounds most like set ?',
                                 '4' : u'Which pronounciation sounds most like sat ?'},
                   'instruction_texts': ['text/intro_eng.txt'],
                   'practice_start_text': 'text/practice_eng.txt',
                   'practice_end_text': 'text/end_practice_eng.txt',
                   'pause_start_text': 'text/pause1_eng.txt',
                   'pause_end_text': 'text/pause0_eng.txt',
                   'end_text': 'text/end_eng.txt',
                   'give_sound_model' : False}


CONDITION_PARAMS = {'exp_1': [FREN_1_PARAMS, ENG_1_PARAMS], 'exp_2' : [FREN_2_PARAMS, ENG_2_PARAMS]}



###########################################################################################
# Start experiment

# get participant info
subject_info = { u'Number':1, 
                 u'Age':20,
                 u'Sex': u'f/m',
                 u'Language': u'french/english',
                 u'English Proficency (1 (No proficiency) - 5 (Fluent))':2,
                 u'French Proficency (1 (No proficiency) - 5 (Fluent))':2,
                 u'condition': "/".join(list(CONDITION_PARAMS.keys()))}
dlg = gui.DlgFromDict(subject_info, title=u'REVCOR')
if dlg.OK:
    subject_number = subject_info[u'Number']
    subject_age = subject_info[u'Age']
    subject_sex = subject_info[u'Sex']
    subject_language  = subject_info[u'Language']
    subject_en = subject_info[u'English Proficency (1 (No proficiency) - 5 (Fluent))']
    subject_fr = subject_info[u'French Proficency (1 (No proficiency) - 5 (Fluent))']
    condition = subject_info[u'condition']   
else:
    core.quit() #the user hit cancel so exit

# log start time and date
date = datetime.datetime.now()
time = core.Clock()

# define GUI elements
win = visual.Window([1366,768],fullscr=False,color="lightgray", units='norm', screen = 1)
screen_ratio = (float(win.size[1])/float(win.size[0]))
label_size = 0.1

# read condition parameters
if subject_language == 'french':
  params = CONDITION_PARAMS[condition][0]
elif  subject_language == 'english':
  params = CONDITION_PARAMS[condition][1]
else:
  #End of experiment
  show_text_and_wait("text/incorrect_language")
  
  # Close Python
  win.close()
  core.quit()
  sys.exit()

# sound trial
response_options = ['[g] phrase 1','[h] phrase 2']
response_keys = ['g', 'h']
play_instruction = visual.TextStim(win, units='norm', text='[Space] Phrase 1 & 2', color='red', height=label_size, pos=(0,0.5))
response_instruction_1 = visual.TextStim(win, units='norm', text=params['questions']['1'], color='black', height=label_size, pos=(0,0.1), alignHoriz='center')
response_instruction_2 = visual.TextStim(win, units='norm', text=params['questions']['2'], color='black', height=label_size, pos=(0,0.1), alignHoriz='center')
response_instruction_3 = visual.TextStim(win, units='norm', text=params['questions']['3'], color='black', height=label_size, pos=(0,0.1), alignHoriz='center')
response_instruction_4 = visual.TextStim(win, units='norm', text=params['questions']['4'], color='black', height=label_size, pos=(0,0.1), alignHoriz='center')
play_icon = visual.ImageStim(win, image='images/play_off.png', units='norm', size = (0.15*screen_ratio,0.15), pos=(0,0.5+2*label_size))
response_labels = []
response_checkboxes = []
reponse_ypos = -0.2
reponse_xpos = -0.1
label_spacing = abs(-0.8 - reponse_ypos)/(len(response_options)+1)
for index, response_option in enumerate(response_options):
    y = reponse_ypos - label_spacing * index
    response_labels.append(visual.TextStim(win, units = 'norm', text=response_option, alignHoriz='left', height=label_size, color='black', pos=(reponse_xpos,y)))
    response_checkboxes.append(visual.ImageStim(win, image='images/rb_off.png', size=(label_size*screen_ratio,label_size), units='norm', pos=(reponse_xpos-label_size, y-label_size*.05)))
# confidence trial
if subject_language == 'french':
  confidence_instruction = visual.TextStim(win, text="A quel point êtes-vous sûr/certain de votre choix ?", color='black', height=0.08, pos=(0,0.6))
  confidence_scale=visual.RatingScale(win, 
                            pos = (0, -0.1), size = 1.5, stretch = 1, # position
                            low=1, high=4, precision = 1, # response_codes
                            labels=(u"Pas du tout sûr",
                                u"Tout à fait certain"), # bug in API: not possible to give 5 labels
                            lineColor='black', marker='triangle', markerColor='SkyBlue',  markerStart=1, 
                            textColor='black', textSize = 0.5, 
                            scale = None, 
                            leftKeys=['left','g'], rightKeys=['right','h'], respKeys=(['1','2','3','4']), noMouse=True, 
                            acceptKeys=(['return','space','num_enter']), skipKeys=None, 
                            acceptSize = 0.8, acceptPreText='Choisissez', acceptText='Valider',showValue=True)
else:
   confidence_instruction = visual.TextStim(win, text="How sure/certain are you of your choice?", color='black', height=0.08, pos=(0,0.6))
   confidence_scale=visual.RatingScale(win, 
                            pos = (0, -0.1), size = 1.5, stretch = 1, # position
                            low=1, high=4, precision = 1, # response_codes
                            labels=(u"Not at all certain",
                                u"Completely certain"), # bug in API: not possible to give 5 labels
                            lineColor='black', marker='triangle', markerColor='SkyBlue',  markerStart=1, 
                            textColor='black', textSize = 0.5, 
                            scale = None, 
                            leftKeys=['left','g'], rightKeys=['right','h'], respKeys=(['1','2','3','4']), noMouse=True, 
                            acceptKeys=(['return','space','num_enter']), skipKeys=None, 
                            acceptSize = 0.8, acceptPreText='Select', acceptText='Validate',showValue=True)


# generate data files
result_file = generate_result_file(subject_number, condition)
trial_files = generate_trial_files(subject_number, condition, n_blocks=params['n_blocks'],n_stims=params['n_stims'])
# add practice block in first position
if params['practice']: 
    practice_file = generate_trial_files(subject_number, condition, n_blocks=1, n_stims = params['n_practice_trials'], practice=True)[0]
    trial_files.insert(0, practice_file)
## duplicate last block (for internal noise computation)
if params['repeat_for_internal_noise']:
	trial_files.append(trial_files[-1])

# display instructions 
for instruction_text in params['instruction_texts']:
    show_text_and_wait(file_name=instruction_text)

# start trials 
if params['practice']: 
    show_text_and_wait(file_name=params['practice_start_text'])
    practice_trial = True # tag to identify first practice block in response data
else: 
    practice_trial = False

trial_count = 0
n_blocks = len(trial_files)
for block_count, trial_file in enumerate(trial_files):
    block_trials = read_trials(trial_file)

    trial_count_in_block = 0
    for trial in block_trials :

        if params['give_sound_model']:
            if trial_count_in_block % params['model_every_n_trial'] == 0: 
                # give sound model
                show_text_with_sounds_( file_name = params['model_text'],
                                        play_keys = params['model_keys'],
                                        sounds = params['model_sounds'])

        # for every trial, two successive screens: a sound trial and a confidence measure

        # sound trial
        # focus play instruction and reset checkboxes
        play_instruction.setColor('red')
        play_icon.setImage('images/play_on.png')
        for checkbox in response_checkboxes:
            checkbox.setImage('images/rb_off.png')
        sound_1 = trial[0]
        sound_2 = trial[1]
        

        end_trial = False
        while (not end_trial):
            update_sound_trial_gui(block_count)
            # upon play command...
            if event.waitKeys()==['space']: 
                
                # unfocus play instruction
                play_instruction.setColor('black')
                play_icon.setImage('images/play_off.png')
                update_sound_trial_gui(block_count)
                
                # play sounds
                play_sound(sound_1)
                core.wait(params['isi'])
                play_sound(sound_2)
                
                # focus response instruction
                response_start = time.getTime()
                if (block_count == 0 or block_count == 1):
                  response_instruction_1.setColor('red')
                if block_count == 2:
                  response_instruction_2.setColor('blue')
                if block_count == 3:
                  response_instruction_3.setColor('red')
                if block_count == 4:
                  response_instruction_4.setColor('blue')
                update_sound_trial_gui(block_count)
                
                # upon key response...
                response_key = event.waitKeys(keyList=response_keys)
                sound_rt = time.getTime() - response_start
                
                # unfocus response_instruction, select checkbox
                if (block_count == 0  or block_count == 1):
                  response_instruction_1.setColor('black')
                if block_count == 2:
                  response_instruction_2.setColor('black')
                if block_count == 3:
                  response_instruction_3.setColor('black')
                if block_count == 4:
                  response_instruction_4.setColor('black')
                print(f"block count {block_count}")
                response_checkboxes[response_keys.index(response_key[0])].setImage('images/rb_on.png')
                update_sound_trial_gui(block_count)
                
                # blank screen and end_trial
                core.wait(0.3) 
                win.flip()
                end_trial = True
        
        # confidence trial

        while confidence_scale.noResponse:
            update_confidence_trial_gui()

        # get confidence ratings
        confidence_response = confidence_scale.getRating()
        confidence_rt = confidence_scale.getRT()

        # reset scale for next trial
        confidence_scale.reset()
        # confidence_scale.noResponse = True
        # confidence_scale.setMarkerPos(0)
        # attention RAZ response_time

        # blank screen and end trial
        core.wait(0.3) 
        win.flip()
        core.wait(0.2) 
    
        # log response
        if response_key == ['g']:
            response_choice = 0
        elif response_key == ['h']:
            response_choice = 1
        
        with open(result_file, 'a',newline='', encoding='utf-8') as file :
            writer = csv.writer(file,lineterminator='\n')
            # common response data for all trials
            row = [subject_number, subject_sex, subject_age, subject_language, subject_en, subject_fr, date, 
               condition, practice_trial, block_count, trial_count]
            # trial-specific response data
            for stim_order,stim in enumerate(trial): 
              stim_params = get_stim_info(file_name=stim)
              #for param_counter, param_values in enumerate(stim_params):
              result = row + [stim,
                                  stim_order,
                                  stim_params['eq'],
                                  stim_params['pitch'],
                                  stim_params['stretch'],                                  
                                  response_choice==stim_order, # chosen/not
                                  round(sound_rt,3), 
                                  confidence_response,
                                  round(confidence_rt,3)]
              writer.writerow(result)
        trial_count += 1
        trial_count_in_block += 1
 
    # inform end of practice at the end of first block
    if params['practice'] & (block_count == 0):
       show_text_and_wait(file_name=params['practice_end_text']) 
       practice_trial = False
    # pause at the end of subsequent blocks
    # this will give the wrong percentage finished without the practice 
    elif block_count < n_blocks-1:
        if subject_language == 'french':
          show_text_and_wait(message = "Vous avez fait %s de l'expérience.\n .\n\n (Appuyez sur une touche pour continuer)."%(str(Fraction(block_count, n_blocks-1))))
        else:
          show_text_and_wait(message = "You have completed %s of the experiment.\n .\n\n (Press any button to continue)."%(str(Fraction(block_count, n_blocks-1))))

        show_text(params['pause_start_text'])
        core.wait(params['pause_duration'])
        show_text_and_wait(params['pause_end_text'])         
        
        
#End of experiment
show_text_and_wait(params['end_text'])

# Close Python
win.close()
core.quit()
sys.exit()