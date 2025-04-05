# Zongyu Li 11/10/2020
# Procedural Error Analysis of JIGSAWS dataset 
# this function outputs an array of objects given the grammar graph and the data including the gesture sequences
# use the .npy file as the 'data' input

def seq_checking(G,data):  
  """
    Parameters
    ----------
    G : networkx graph
        the gesture graph
    data : an array of objects
        data[0]-sequence
        data[1]-id
        data[2]-trial number
        data[3]-skill level

    Returns
    -------
    seq_check : an array of objects
        seq_check[0]-error number
        seq_check[1]-id
        seq_check[2]-trial number
        seq_check[3]-skill level
        seq_check[4]-added gesture(not a gesture in the graph)
        seq_check[5]-error gesture(Gpre-Gnow-Gafter)

    """ 
  seq_check=list()
  for idx in range(data.shape[0]):
    add_ges='0'
    start=['start']
    err=0
    prev = list(G.successors(start[0]))
    path = data.iloc[idx,0]
    err_seq=[]
    add_ges=[]
    for num, i in enumerate(path):
      succset=set(prev)
      if not (i in succset):
        err = err+1
        if num-1<0:
          err_seq_entry=list([path[num]])
        elif (num+1)<path.shape[0]:
          err_seq_entry=list([path[num-1],path[num],path[num+1]])
          err_seq_entry="_".join(err_seq_entry)
        else:
          err_seq_entry=list([path[num-1],path[num]])
          err_seq_entry="_".join(err_seq_entry)
        #if num+1<data.shape[0]:
          #err_seq_entry=err_seq_entry.append(path[num+1])
        #print(err_seq_entry)
    #path=Suturing_seq[-5,:]
def seq_checking(G,data):  
  """
    Parameters
    ----------
    G : networkx graph
        the gesture graph
    data : an array of objects
        data[0]-sequence
        data[1]-id
        data[2]-trial number
        data[3]-skill level

    Returns
    -------
    seq_check : an array of objects
        seq_check[0]-error number
        seq_check[1]-id
        seq_check[2]-trial number
        seq_check[3]-skill level
        seq_check[4]-added gesture(not a gesture in the graph)
        seq_check[5]-error gesture(Gpre-Gnow-Gafter)

    """ 
  seq_check=list()
  for idx in range(data.shape[0]):
    add_ges='0'
    start=['start']
    err=0
    prev = list(G.successors(start[0]))
    path = data.iloc[idx,0]
    err_seq=[]
    add_ges=[]
    for num, i in enumerate(path):
      succset=set(prev)
      if not (i in succset):
        err = err+1
        if num-1<0:
          err_seq_entry=list([path[num]])
        elif (num+1)<path.shape[0]:
          err_seq_entry=list([path[num-1],path[num],path[num+1]])
          err_seq_entry="_".join(err_seq_entry)
        else:
          err_seq_entry=list([path[num-1],path[num]])
          err_seq_entry="_".join(err_seq_entry)
        #if num+1<data.shape[0]:
          #err_seq_entry=err_seq_entry.append(path[num+1])
        #print(err_seq_entry)
        
        err_seq.append(err_seq_entry)
        #print(i)
        if G.__contains__(i):
          prev = list(G.successors(i))
        else:
          add_ges_entry=i
          add_ges.append(add_ges_entry)
          if num+1<path.shape[0]:
            prev = [path[num+1]]
          else:
            break
      else:
        prev = list(G.successors(i))
    entry_new = np.reshape(np.array(list([err,data.iloc[idx,1],data.iloc[idx,2],data.iloc[idx,3],add_ges,err_seq]),dtype=object),(1,6))
    if len(seq_check)==0:
      seq_check=entry_new
    else:
      seq_check=np.vstack((seq_check,entry_new))
  return seq_check    
        err_seq.append(err_seq_entry)
        #print(i)
        if G.__contains__(i):
          prev = list(G.successors(i))
        else:
          add_ges_entry=i
          add_ges.append(add_ges_entry)
          if num+1<path.shape[0]:
            prev = [path[num+1]]
          else:
            break
      else:
        prev = list(G.successors(i))
    entry_new = np.reshape(np.array(list([err,data.iloc[idx,1],data.iloc[idx,2],data.iloc[idx,3],add_ges,err_seq]),dtype=object),(1,6))
    if len(seq_check)==0:
      seq_check=entry_new
    else:
      seq_check=np.vstack((seq_check,entry_new))
  return seq_check
