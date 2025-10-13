import pandas as pd
import pickle as pk
import networkx as nx
import numpy as np
import hypergraphx as hgx
from tqdm import tqdm
import time
import datetime
import os


def create_babboons_TN(root):
    data = pd.read_csv(f"{root}/raw/RFID_data.txt", sep='\t')

    data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y %H:%M')


    unique_values = set(data['i'].unique()).union(set(data['j'].unique()))
    mapping_dict = {val: idx for idx, val in enumerate(unique_values)}
    data['i'] = data['i'].map(mapping_dict)
    data['j'] = data['j'].map(mapping_dict)

    # Make hyperedges
    grouped = data.groupby('DateTime').apply(
        lambda x: set(list(x['i']) + list(x['j']))
    ).reset_index(name='hyperedge')

    grouped['numE'] = grouped['hyperedge'].apply(len)
    
    # Step 2: Floor the timestamps to 15-minute intervals
    dt = '15min'
    grouped[f'{dt}_slot'] = grouped['DateTime'].dt.floor(dt)

    # Step 3: Group values into sets
    _grouped = grouped.groupby(f'{dt}_slot')['hyperedge'].apply(list).reset_index()
    _grouped['hyperedge'] = _grouped['hyperedge'].apply(lambda x: set(frozenset(s) for s in x))
    _grouped['numE'] = _grouped['hyperedge'].apply(len)

    times = _grouped['15min_slot'].tolist()
    hyperedges = _grouped['hyperedge'].tolist()

    TN = dict()
    for i, (t, elist) in enumerate(zip(times, hyperedges)):
        h = hgx.Hypergraph(elist)
        TN[10*(i+1)] = h
    
    print(f"Len={len(TN)}")

    pk.dump(TN, open(f"{root}/processed/TS_babboons.pck", 'wb'), protocol=-1)



def create_Utah_TN(root, ds):
        
    assert ds in ['Mid1', 'Elem1'], f"Dataset should be either Mid1 or Elem1, not {ds}"
    def extract_networks(data_dir, dataset, n_minutes=5, original_nets=True):
        """Function that reads the edgelist (t, i, j) and returns
        a network aggregated at n_minutes snapshots as a dictionary of nx.Graph()s,
        having t as a key.
        If original_nets is set to True it also returns the original non-aggregated network."""

        # Reading the data and setting t0
        f = open(data_dir + '/tij_' + dataset + '.txt')
        (t0, i, j) = map(int, str.split(f.readline()))
        # Special temporal scale for these two Datasets
        if dataset not in ['LyonSchool', 'LH10']:
            t0 = t0 * 20
        f.close()

        # Aggregation on scale of x minutes
        delta_t = 20 * 3 * n_minutes
        if original_nets == True:
            originalnetworks = {}
        aggnetworks = {}
        f = open(data_dir + '/tij_' + dataset + '.txt')
        for line in f:
            (t, i, j) = map(int, str.split(line))
            # Special temporal scale for these two Datasets
            if dataset not in ['LyonSchool', 'LH10']:
                t = t * 20
            if original_nets == True:
                if t not in originalnetworks:
                    originalnetworks[t] = nx.Graph()
                originalnetworks[t].add_edge(i, j)
            # this is a trick using the integer division in python
            aggtime = t0 + ((t - t0) // delta_t) * delta_t
            if aggtime not in aggnetworks:
                aggnetworks[aggtime] = nx.Graph()
            aggnetworks[aggtime].add_edge(i, j)
        f.close()
        if original_nets == True:
            return originalnetworks, aggnetworks
        else:
            return aggnetworks

    dataset_dir = f"{root}/raw"

    n_minutes = 15

    TN = dict()
    aggs = extract_networks(dataset_dir, dataset=ds, n_minutes=n_minutes, original_nets=True)[0]

    for t in tqdm(sorted(aggs.keys())):
        g = aggs[t]
        cliques = list(nx.find_cliques(g))
        TN[t] = hgx.Hypergraph(cliques)

    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)


def create_congress_bills_TN(root):
    ds = 'congress-bills'
    # edge-list necessarily must be calculated and hence can be returned immediately as well
    f = open(f'{root}/raw/{ds}-nverts.txt', 'r') #vector with number of hyperedge sizes
    size_hype = f.read().splitlines()
    size_hype = [int(el) for el in size_hype]

    f = open(f'{root}/raw/{ds}-simplices.txt', 'r') #vector with nodes ID
    nodes_hype = f.read().splitlines()
    nodes_hype = [int(el) for el in nodes_hype]

    f = open(f'{root}/raw/{ds}-times.txt', 'r')
    times_hype = f.read().splitlines()
    times_hype = np.array(times_hype).astype(int)


    node_stack = nodes_hype.copy()
    edge_list = []
    # data = []
    for  (size, t) in tqdm(zip(size_hype, times_hype)):
        hypedge = [(node_stack.pop(0)) for _ in range((size))]
        edge_list.append(hypedge)
        # for e in itertools.combinations(hypedge, 2):
            # data.append([t, e[0], e[1]])

    data = pd.DataFrame(data=zip(times_hype, edge_list), columns=['timestamp', 'edge'])
    
    # group timestamps together
    data = data.groupby(f'timestamp')['edge'].apply(list).reset_index()
    # remove all singleton active nodes
    data['edge'] = data['edge'].apply(lambda row: [sublist for sublist in row if len(sublist) > 1])
    data.sort_values('timestamp')
    # remove all empty rows
    data = data[data['edge'].apply(len) > 1]

    data = data.sort_values('timestamp')

    times = data['timestamp'].tolist()
    hyperedges = data['edge'].tolist()

    TN = dict()
    for t, elist in zip(times, hyperedges):
        h = hgx.Hypergraph(elist)
        TN[t] = h

    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)



def create_DAWN_TN(root):
    ds = "DAWN"

    # edge-list necessarily must be calculated and hence can be returned immediately as well
    f = open(f'{root}/raw/{ds}-nverts.txt', 'r') #vector with number of hyperedge sizes
    size_hype = f.read().splitlines()
    size_hype = [int(el) for el in size_hype]

    f = open(f'{root}/raw/{ds}-simplices.txt', 'r') #vector with nodes ID
    nodes_hype = f.read().splitlines()
    nodes_hype = [int(el) for el in nodes_hype]

    f = open(f'{root}/raw/{ds}-times.txt', 'r')
    times_hype = f.read().splitlines()
    times_hype = np.array(times_hype).astype(int)


    node_stack = nodes_hype.copy()
    edge_list = []
    # data = []
    for  i, (size, t) in enumerate(zip(size_hype, times_hype)):
        print(f"{i}/{2272433}", end='\r')
        hypedge = [(node_stack.pop(0)) for _ in range((size))]
        edge_list.append(hypedge)
        # for e in itertools.combinations(hypedge, 2):
            # data.append([t, e[0], e[1]])

    data = {'t':times_hype, 'edges':edge_list}
    data = pd.DataFrame(data)
    data.head()

    grouped = data.groupby('t')['edges'].apply(list).reset_index()
    grouped['numE'] = grouped['edges'].apply(len)
    grouped['maxsize'] = grouped['edges'].apply(max, key=len).apply(len)
    grouped.head()

    times = grouped['t'].tolist()
    hyperedges = grouped['edges'].tolist()

    TN = dict()
    for t, elist in zip(times, hyperedges):
        h = hgx.Hypergraph(elist)
        TN[t] = h

    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)
    pk.dump(edge_list, open(f"{root}/processed/edge_list_{ds}.pck", 'wb'), protocol=-1)


def create_HS1X_TN(ds, root, bin_size=200):

    if '11'  in ds:
        fname = f"{root}/raw/thiers_2011.csv"
    else: 
        fname = f"{root}/raw/thiers_2012.csv"


    data = pd.read_csv(fname, header=None, sep='\t')
    data.columns = ['contact_time','id1','id2','Ci','Cj']

    # make nodes start from 0
    minnodeval = min(min(data['id1']), min(data['id2']))
    data['id1'] = data['id1'] - minnodeval
    data['id2'] = data['id2'] - minnodeval

    # check that correct
    # _allnodes = set(data['id1'].unique()).union(set(data['id2'].unique())) 

    grouped = data.groupby('contact_time').apply(
        lambda x: set(list(x['id1']) + list(x['id2']))
    ).reset_index(name='hyperedge')

    grouped

    grouped['bin'] = (grouped['contact_time']//bin_size) * bin_size
    grouped = grouped.groupby('bin')['hyperedge'].apply(list).reset_index()
    grouped['hyperedge'] = grouped['hyperedge'].apply(lambda x: set(frozenset(s) for s in x))
    # # grouped
    grouped['numE'] = grouped['hyperedge'].apply(len)
    grouped['maxsize'] = grouped['hyperedge'].apply(max, key=len).apply(len)
    grouped

    times = grouped['bin'].tolist()
    hyperedges = grouped['hyperedge'].tolist()

    TN = dict()
    for t, elist in tqdm(zip(times, hyperedges)):
        h = hgx.Hypergraph(elist)
        TN[t] = h


    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)

def create_malawi_TN(root, bin_size=1000):

    fname = f"{root}/raw/tnet_malawi_pilot2.csv"


    data = pd.read_csv(fname, index_col=0)


    grouped = data.groupby('contact_time').apply(
        lambda x: set(list(x['id1']) + list(x['id2']))
    ).reset_index(name='hyperedge')


    grouped['bin'] = (grouped['contact_time']//bin_size) * bin_size
    grouped = grouped.groupby('bin')['hyperedge'].apply(list).reset_index()
    grouped['hyperedge'] = grouped['hyperedge'].apply(lambda x: set(frozenset(s) for s in x))
    # grouped
    grouped['numE'] = grouped['hyperedge'].apply(len)
    grouped['maxsize'] = grouped['hyperedge'].apply(max, key=len).apply(len)


    times = grouped['bin'].tolist()
    hyperedges = grouped['hyperedge'].tolist()

    TN = dict()
    for t, elist in tqdm(zip(times, hyperedges)):
        h = hgx.Hypergraph(elist)
        TN[t] = h


    pk.dump(TN, open(f"{root}/processed/TS_Malawi.pck", 'wb'), protocol=-1)



def create_InVS_TN(ds, root):

    if '13' in ds: 
        fname = f"{root}/raw/tij_pres_InVS13.dat"
    elif '15' in ds:
        fname = f"{root}/raw/tij_pres_InVS15.dat"
    
    data = pd.read_csv(fname, sep=' ', header=None)
    data.columns = ['contact_time','id1','id2']

    # Group column1 and column2 by 'group' and combine values
    grouped = data.groupby('contact_time').apply(
        lambda x: list(zip(x['id1'], x['id2']))
    ).reset_index(name='dy_edges')
    grouped['numDy'] = grouped['dy_edges'].apply(len)



    times_hype = grouped['contact_time']
    dy_edges = grouped['dy_edges'].tolist()
    TN = dict()
    hedges = dict()
    for i, t in tqdm(enumerate(times_hype)):
        elist = dy_edges[i]
        G = nx.Graph(elist)
        H = hgx.Hypergraph(list(nx.find_cliques(G)))
        if '13' in ds:
            TN[t-28820] = H
        else:
            TN[t] = H
        hedges[t] = H.get_edges()


    grouped.columns = ['contact_time', 'hyperedge', 'numE']
    grouped['hyperedge'] = hedges.values()
    grouped['numE'] = grouped['hyperedge'].apply(len)

    grouped['maxsize'] = grouped['hyperedge'].apply(max, key=len).apply(len)


    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)


def create_Kenyan_TN(root, ds):
    # LOAD DATA
    data_within=pd.read_csv(f'{root}/raw/scc2034_kilifi_all_contacts_within_households.csv')
    for i in [1,2]:
        data_within[f'id{i}'] = data_within[f'h{i}']+data_within[f'm{i}'].astype(str)
        data_within = data_within.drop([f'h{i}', f'm{i}'], axis=1)

    data_within.drop(['age1', 'age2', 'sex1', 'sex2'], axis=1, inplace=True)
    data_within['timestamp'] = (data_within['day'] - 1)*24+data_within['hour']
    data_within = data_within[['id1', 'id2', 'timestamp']]  #omit duration because longest is 12 minutes which is less than the hour so just make all same


    data_across=pd.read_csv(f'{root}/raw/scc2034_kilifi_all_contacts_across_households.csv')
    for i in [1,2]:
        data_across[f'id{i}'] = data_across[f'h{i}']+data_across[f'm{i}'].astype(str)
        data_across = data_across.drop([f'h{i}', f'm{i}'], axis=1)

    data_across.drop(['age1', 'age2', 'sex1', 'sex2'], axis=1, inplace=True)
    data_across['timestamp'] = (data_across['day'] - 1)*24+data_across['hour']
    data_across = data_across[['id1', 'id2', 'timestamp']]  #omit duration because longest is 12 minutes which is less than the hour so just make all same

    # DETERMINE UNIQUE NODES
    unique_nodes = [] 
    unique_nodes += (list(pd.unique(data_within['id1'])))
    unique_nodes += (list(pd.unique(data_within['id2'])))
    unique_nodes += (list(pd.unique(data_across['id1'])))
    unique_nodes += (list(pd.unique(data_across['id2'])))

    # RELABEL THE NODES
    relabelling = {el:i for i, el in enumerate(unique_nodes)} 

    data_within['id1'] = data_within['id1'].apply(lambda x: relabelling[x])
    data_within['id2'] = data_within['id2'].apply(lambda x: relabelling[x])
    data_across['id1'] = data_across['id1'].apply(lambda x: relabelling[x])
    data_across['id2'] = data_across['id2'].apply(lambda x: relabelling[x])


    grouped_within = data_within.groupby('timestamp').apply(
            lambda x: list(zip(x['id1'], x['id2']))
        ).reset_index(name='dy_edges')
    grouped_within['dy_edges'] = grouped_within['dy_edges'].apply(lambda x: set(frozenset(s) for s in x))

    grouped_across = data_across.groupby('timestamp').apply(
            lambda x: list(zip(x['id1'], x['id2']))
        ).reset_index(name='dy_edges_2')
    grouped_across['dy_edges_2'] = grouped_across['dy_edges_2'].apply(lambda x: set(frozenset(s) for s in x))

    grouped = grouped_within.join(grouped_across.set_index('timestamp'), on='timestamp')
    grouped = grouped[['timestamp', 'dy_edges', 'dy_edges_2']]

    TN = dict()
    hedges = dict()
    for i, el in enumerate(dict(grouped.T).values()):
        t, elist, e2 = dict(el).values()
        if not e2 is np.nan:
            elist = elist.union(e2)

        G = nx.Graph(elist)
        H = hgx.Hypergraph(list(nx.find_cliques(G)))
        # TN[t] = H
        TN[i] = H
        hedges[t] = H.get_edges()

    grouped['hyperedge'] = hedges.values()
    grouped = grouped[['timestamp', 'hyperedge']]
    grouped['numE'] = grouped['hyperedge'].apply(len)

    grouped['maxsize'] = grouped['hyperedge'].apply(max, key=len).apply(len)



    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)



def create_FnF_TN(root):

    # Reading the raw data into a more usable format
    if not os.path.isfile(f'{root}/fnf_processed.pck'):
        print("Not yet loaded and saved")

        person1_list = []
        person2_list = []
        time_list = []
        data_name = f"{root}/BluetoothProximity.csv"
        file = open(data_name, "r")
        next(file) # skip first line 
        for line in file:
            # print(line)
            row = line.split(',')
            if row[2] and row[0]:
                person1_list.append(row[0])
                person2_list.append(row[2])
                ts = time.mktime(datetime.datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S").timetuple()) # convert to timestamp
                time_list.append(int(np.round(float(ts))))
        file.close()

        unique_persons = list(set(person1_list).union(set(person2_list)))
        unique_pids = {old:new for new, old in enumerate(unique_persons)}

        person1_list_new = [unique_pids[p] for p in person1_list]
        person2_list_new = [unique_pids[p] for p in person2_list]

        f_n_f = np.array([time_list, person1_list_new, person2_list_new])
        pk.dump(f_n_f, open(f'{root}/fnf_processed.pck','wb'))

    else:
        print("Already loaded and saved!")
        time_list, person1_list_new, person2_list_new = pk.load(open(f'{root}/fnf_processed.pck','rb'))

    # Expand data by splitting columns
    data = {'date':time_list, 'ID':list(zip(person1_list_new, person2_list_new))}
    df = pd.DataFrame(data)

    # Convert the date column to datetime
    df['date'] = pd.to_datetime(df['date'], unit='s')
    # Create year-month and year-month-day columns
    df['ym'] = df['date'].dt.to_period('M')  # Year-month
    df['ymd'] = df['date'].dt.to_period('D')  # Year-month-day

    # Determine unique months and days in each; for each month we create a separate dataset
    unique_months = list(df['ym'].unique())
    for month in unique_months:
        # for each day in the month, we group in slots of 8 hours
        unique_days_in_month = df[df['ym']==month]['ymd'].unique()
        print(month, len(unique_days_in_month))

    # SELECT MONTHS FOR WHICH WE HAVE FULL DATA
    unique_months = unique_months[2:12]

    # We break up into 8-hr blocks
    hours = ['00', '08', '16']

    # We create the time series
    TS_by_months = {}
    for month in tqdm(unique_months):
        TS_month = {}

        unique_days_in_month = df[df['ym']==month]['ymd'].unique()
        for day in unique_days_in_month:
            for hour in hours:
                sel_df = df[df['ymd']==day][['date', 'ID']]

                start_point = pd.Timestamp(f'{day} {hour}:00:00')
                time_delta = pd.Timedelta(minutes=min)
                end_point = start_point + time_delta


                # Filter the DataFrame to include only rows within the time range
                delta_t_df = sel_df[(sel_df['date'] >= start_point) & (sel_df['date'] <= end_point)]

                elist = delta_t_df['ID'].tolist()
                G_ = nx.Graph(elist)
                cliques = list(nx.find_cliques(G_))
                H_ = hgx.Hypergraph(cliques)

                TS_month[f"{day}-{hour}h"] = H_
        
        TS_by_months[month] = TS_month
            

    # We save the time series
    for month in tqdm(unique_months):
        ds = f"FnF_{str(month)}"
        TS_month = TS_by_months[month]
        pk.dump(TS_month, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)



def create_Copenhagen_TN(root):
    ds = 'Copenhagen'
    fname = f'{root}/copenhagen_bt_symmetric.csv'

    data = pd.read_csv(fname, sep=',')

    data['rssi'] = -1*data['rssi']
    data = data[data['user_b']!=-1 ]
    data = data[data['user_b']!=-2]
    data = data.drop(['rssi'], axis=1)
    data = np.array(data)

    print("Reading data file by time step")
    graph = {}
    for l in tqdm(data):
        t, a, b = l
        t = int(t)
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a,b))
        else:
            graph[t] = [(a,b)]

    print("Creating TN")
    TN = {}
    for k in tqdm(graph.keys()):
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        H = hgx.Hypergraph(c)
        TN[k] = H

    print("Saving temporal network")
    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)



def create_Sociopatterns_TN(root, ds):
    dataset = f"{root}/tij_pres_{ds}.dat"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    if ds == 'SFHH':
        timechange = 31500
    elif ds == 'LyonSchool':
        timechange = 34240
    elif ds == 'Thiers13':
        timechange = 29960

    print("Loading data")
    graph = {}
    for l in tqdm(lines):
        t, a, b = l.split()
        t = int(t) - timechange
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a,b))
        else:
            graph[t] = [(a,b)]

    fopen.close()


    print("Creating Temporal Network")
    TN = set()
    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        H = hgx.Hypergraph(c)
        TN[k] = H


    print("Saving temporal network")
    pk.dump(TN, open(f"{root}/processed/TS_{ds}.pck", 'wb'), protocol=-1)