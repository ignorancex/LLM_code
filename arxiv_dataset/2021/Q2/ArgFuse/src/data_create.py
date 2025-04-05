import pickle as pkl
with open('../resources/snapshot_event_arg_extraction_data.pickle', 'rb') as pkl_in: #replace with event argument annotated data path
    train = pkl.load(pkl_in)
    valid = pkl.load(pkl_in)
    test = pkl.load(pkl_in)

def fill_events(doc_dict):
    new_dict = {}
    
    for k, v in doc_dict.items():
        event = ''
        etrigger = ''
        new_dict[k] = []
        
        for i, tup in enumerate(v):
            
            if tup[0] == 'O' : 
                new_dict[k].append(tup)
                continue
            
            event = tup[0].split('__')[1]
            
            if event :
                new_dict[k].append(tup)
                continue
            
            if not event:
                
                for j in range(i+1, len(v)):
                    
                    if v[j][0] != 'O' and v[j][0].split('__')[1]:
                        event = v[j][0].split('__')[1]
                        etrigger = v[j][0].split('__')[-1]
                        #if k == 523.0: print(event, etrigger)
                        break
            
            tup = ('__'.join([tup[0].split('__')[0], event, etrigger]), tup[1])
            new_dict[k].append(tup)
            
    return new_dict               


def create_structure(doc_dict):

    data = {}

    for k, v in doc_dict.items():
        flag = 0
        data[k] = {}
        
        data[k]['event'] = []
        data[k]['TIME-ARG'] = [[], [], []]
        data[k]['PLACE-ARG'] = [[], [], []]
        data[k]['CASUALTIES-ARG'] = [[],[], []]
        data[k]['AFTER_EFFECTS-ARG'] = [[],[], []]
        data[k]['REASON-ARG'] = [[],[], []]
        data[k]['PARTICIPANT-ARG'] = [[],[], []]
        data[k]['doc'] = ''
        start = 0
        end = 0
        sent = ''
        sents = []
        arg = ''
        arg_trigger = ''
        arguments = []
        spans = []

        for i, tup in enumerate(v):
            sent = sent + ' '+ tup[1]

            if tup[0] != 'O' :

                if arg and arg != tup[0].split('__')[0].split('B_')[1]: #case for consecutive different arg
                    arguments.append((event,arg,arg_trigger))
                    arg = ''
                    arg_trigger = ''
                    spans.append((start, end))
                    start = 0
                    end = 0

                flag = 1
                event = tup[0].split('__')[1]
                arg = tup[0].split('__')[0].split('B_')[1]
                if start == 0 and not arg_trigger:
                    arg_trigger = tup[1]
                    start = i
                    end = i
                else:
                    arg_trigger = arg_trigger + ' ' + tup[1]
                    end = i
                    
            elif tup[0] == 'O' and flag == 1:
                arguments.append((event,arg,arg_trigger))
                arg = ''
                arg_trigger = ''
                spans.append((start, end))
                start = 0
                end = 0
                flag = 0 #denotes a complete span has been recorded if 0 and 1 if it is in process of recording

            if tup[1] not in ['a.m.', 'am.', 'p.m.', 'pm.', 'v.s.', 'vs.'] and (list(tup[1])[-1] == '.' or list(tup[1])[-1] == '?' or list(tup[1])[-1] == '!' ):
                #checking if next char is in upper case or not --> denotes start of a new sentence
                if i < len(v)-1 and not list(v[i+1][1])[0].isupper() and not list(v[i+1][1])[0] == '“': continue
                sents.append(sent)
                sent = ''
                if arg and arg_trigger:
                    arguments.append((event,arg,arg_trigger)) #case for consecutive same arg diff spans
                    spans.append((start, end))
                    start = 0
                    end = 0
                    arg = ''
                    arg_trigger = ''
                    flag = 0
            if i == len(v)-1 and sent not in sents: sents.append(sent) #some sentences do not have punctuation at end

        for j, in_tup in enumerate(arguments): #filling up the data-structure defined in cell 1
            arg = in_tup[1]
            
            if not data[k]['event']: data[k]['event'] = in_tup[0] #event -- assuming first event in thte document is the entire event of the document
            temp = [s for s in sents if in_tup[2] in s] 
            '''if not temp:
                print(k)
                print(spans[j])
                print(in_tup)
                print(sents)''' #print to debug
            
            data[k][arg][0].append(temp[0]) # arg sentence
            data[k][arg][1].append(spans[j]) #span
            data[k][arg][2].append(in_tup[2]) #arg trigger
            data[k]['doc'] = ' '.join(sents)

    return data

import re
def attach_special_tokens(sent, phrase, arg_type):
    sent = re.sub(r'[^\w\s]', '', sent)
    phrase = re.sub(r'[^\w\s]', '', phrase)
    strt = sent.find(phrase)
    mod = sent[:strt]+'['+arg_type+'] '+sent[strt:strt+len(phrase)]+' ['+arg_type+'] '+sent[strt+len(phrase):]
    return mod.strip()

#pair generation
def pair_generate(data):
    arg_list = ['TIME-ARG', 'PLACE-ARG', 'CASUALTIES-ARG', 'AFTER_EFFECTS-ARG', 'REASON-ARG', 'PARTICIPANT-ARG']

    pairs = {}
    pairs_sent = {}
    track = {}

    for k, v in data.items():

        pairs[k] = []
        pairs_sent[k] = []
        track[k] = []

        for arg in arg_list:

            # argument pairs
            doc_args1 = v[arg][2] #args
            doc_args2 = v[arg][0] #sents

            if len(doc_args2) > 1:

                for i in range(len(doc_args2)):
                    for j in range(len(doc_args2)):
                        if i == j: continue
                        if doc_args1[i] == doc_args1[j]: continue
                        #elif str(k)+arg+str(i)+str(j) not in track[k] and str(k)+arg+str(j)+str(i) not in track[k]:
                        
                        s1 = attach_special_tokens(doc_args2[i], doc_args1[i], arg)
                        s2 = attach_special_tokens(doc_args2[j], doc_args1[j], arg)
                            
                        if s1 + s2 not in track[k] and s2 + s1 not in track[k]:
                            pairs[k].append([s1, s2, v['doc'], v['event']])
                            track[k].append(s1 + s2)
                            track[k].append(s2 + s1)
                            #print('pairs ',k, arg, i, j)


    assert len(data) == len(pairs)  
    return pairs

if __name__ == '__main__':
    train_data = create_structure(fill_events(train))
    val_data = create_structure(fill_events(valid))
    test_data = create_structure(fill_events(test))

    with open('../resources/aggr_data.pickle', 'wb') as pkl_out:
         pkl.dump(train_data, pkl_out)
         pkl.dump(val_data, pkl_out)
         pkl.dump(test_data, pkl_out)

    #using predicted data
    with open('../data/en_pred-cas.txt', 'r') as f:
         y = f.readlines()
         lines = []
         tmp = []
         event = []
         eve = ''
         for line in y:
             if '[SPL]' in line or '[SEP]' in line:
                  eve = eve + ' ' + line.split()[0]
                  continue
             elif line == '\n':
                 lines.append(tmp)
                 event.append(eve.split()[5].upper()) #5 because the template has 5 words before the event type
                 eve = ''
                 tmp = []
             else:
                 tmp.append((line.strip('\n')))

    assert len(lines) == len(event)

    #converting pred data to dict structure as annotated data
    pred = {}
    for k, v in test.items():
        pred[k] = v
    count = 0
    for k, v in pred.items():
        tmp = []
        for i, tup in enumerate(v):
            try: #because length of predicted documents is smaller due to 500 characters/words limit in BERT
                if lines[count][i].split()[-1] == 'O':
                    tmp.append(('O', tup[1]))
                else:
                    tmp.append((lines[count][i].split()[-1] + '__' + event[count], tup[1]))
            except:
                break
        count += 1
        pred[k] = tmp
           
    assert len(pred) == len(test)

    pred_data = create_structure(fill_events(pred))
    assert len(pred_data) == len(pred)

    with open('../resources/aggr_data-pred.pickle', 'wb') as pkl_out:
         pkl.dump(pred_data, pkl_out)

    print('Data creation completed')
