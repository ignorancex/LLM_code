import Metrica_IO2 as mio
import Metrica_Viz3 as mviz
import Metrica_Velocities2 as mvel
import Metrica_PitchControl3 as mpc
#import Metrica_EPV as mepv


import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pdb
import warnings
import re
import argparse
import matplotlib.pyplot as plt

warnings.simplefilter('ignore')

#import third_party as thp
import obso_player as obs


# create parser
parser = argparse.ArgumentParser()
parser.add_argument('--id', type=str, default='1_1_00', help='game id')
parser.add_argument('--data_type', type=str, default='metrica', help='dataset')
parser.add_argument('--data', type=str, default='data0', help='data file name')
parser.add_argument('--start_ev', type=int, default=0, help='dataset')
parser.add_argument('--end_ev', type=int, default=0, help='dataset')
# parser.add_argument('--len_event', type=int, default=0, help='dataset')
args = parser.parse_args()

# select game number
game_id = args.id

def reduce_frame(data, type='tracking'):
    if type == 'tracking':
        data_reduced = data[data.index %2 == 0]
        data_reduced.index = data_reduced.index // 2
    elif type == 'events':
        data_reduced = data[data['Start Frame']%2 == 0]
        data_reduced.index = data_reduced.index // 2
        data_reduced['Start Frame'] = data_reduced['Start Frame'] // 2
        data_reduced['End Frame'] = data_reduced['End Frame'].div(2)
    return data_reduced

if args.data_type == 'metrica':
    # set up initial path to data
    DATADIR = f'./assets/{game_id}'

    # read in the event data
    events = mio.read_event_data(DATADIR,game_id)

    # read in tracking data
    tracking_home = mio.tracking_data(DATADIR,game_id,'Home')
    tracking_away = mio.tracking_data(DATADIR,game_id,'Away')

    # Convert positions from metrica units to meters (note change in Metrica's coordinate system since the last lesson)
    tracking_home = mio.to_metric_coordinates(tracking_home)
    tracking_away = mio.to_metric_coordinates(tracking_away)
    events = mio.to_metric_coordinates(events)

    # reverse direction of play in the second half so that home team is always attacking from right->left
    tracking_home,tracking_away,events = mio.to_single_playing_direction(tracking_home,tracking_away,events)

    # Calculate player velocities
    tracking_home = mvel.calc_player_velocities(tracking_home,smoothing=True)
    tracking_away = mvel.calc_player_velocities(tracking_away,smoothing=True)
    Metrica_df = events
    
elif args.data == 'rdd':
    import pdb; pdb.set_trace()

elif args.data == 'statsbomb':
    error('since we cannot compute velocity, we currently cannot compute obso from statsbomb data')

elif args.data == 'jleague':
    # set folder and file name
    Jdatafolder = "../JLeagueData"
    FMfolder = "/Data_2019FM/"
    Jdata_FM = Jdatafolder + FMfolder
    event_data_name = "/play.csv"
    player_data_name = "/player.csv"
    game_date =  os.listdir(path=Jdata_FM)

    # set event data
    sample_game_data = pd.read_csv(Jdata_FM+game_date[game_id]+event_data_name, encoding="shift_jis")
    #sample_spadl = thp.convert_J2spadl(sample_game_data)

    # set tracking data
    tracking_home = pd.read_csv(Jdata_FM+game_date[game_id]+'/home_tracking.csv')
    tracking_away = pd.read_csv(Jdata_FM+game_date[game_id]+'/away_tracking.csv')
    tracking_home = tracking_home.drop(columns='Unnamed: 0')
    tracking_away = tracking_away.drop(columns='Unnamed: 0')

    # preprocessing player position 
    entry_home_df = tracking_home.loc[0].isnull()
    entry_away_df = tracking_away.loc[0].isnull()
    home_column = tracking_home.columns
    away_column = tracking_away.columns
    home_player_num = [s[:-2] for s in home_column if re.match('Home_\d*_x', s)]
    away_player_num = [s[:-2] for s in away_column if re.match('Away_\d*_x', s)]

    # replace nan 
    for player in home_player_num:
        if entry_home_df[player+'_x']:
            tracking_home[player+'_x'] = tracking_home[player+'_x'].fillna(method='ffill')
            tracking_home[player+'_y'] = tracking_home[player+'_y'].fillna(method='ffill')
        else:
            tracking_home[player+'_x'] = tracking_home[player+'_x'].fillna(method='bfill')
            tracking_home[player+'_y'] = tracking_home[player+'_y'].fillna(method='bfill')

    for player in away_player_num:
        if entry_away_df[player+'_x']:
            tracking_away[player+'_x'] = tracking_away[player+'_x'].fillna(method='ffill')
            tracking_away[player+'_y'] = tracking_away[player+'_y'].fillna(method='ffill')
        else:
            tracking_away[player+'_x'] = tracking_away[player+'_x'].fillna(method='bfill')
            tracking_away[player+'_y'] = tracking_away[player+'_y'].fillna(method='bfill')

    # data interpolation in ball position in tracking data
    tracking_home['ball_x'] = tracking_home['ball_x'].interpolate()
    tracking_home['ball_y'] = tracking_home['ball_y'].interpolate()
    tracking_away['ball_x'] = tracking_away['ball_x'].interpolate()
    tracking_away['ball_y'] = tracking_away['ball_y'].interpolate()

    # check nan ball position x and y in tracking data
    tracking_home['ball_x'] = tracking_home['ball_x'].fillna(method='bfill')
    tracking_home['ball_y'] = tracking_home['ball_y'].fillna(method='bfill')
    tracking_away['ball_x'] = tracking_away['ball_x'].fillna(method='bfill')
    tracking_away['ball_y'] = tracking_away['ball_y'].fillna(method='bfill')

    # event data convert spadl to Metrica
    Metrica_df = obs.convert_Metrica_for_event(sample_spadl)
    # check 'Home' team in tracking and event data
    Metrica_df = obs.check_home_away_event(Metrica_df, tracking_home, tracking_away)
    # delete last event because this event is 'time up' event
    Metrica_df = Metrica_df[:-1]
    import pdb; pdb.set_trace()

# reduce the number of frames by half
tracking_home = reduce_frame(tracking_home, type='tracking')
tracking_away =reduce_frame(tracking_away, type='tracking')
Metrica_df = reduce_frame(Metrica_df, type='events')

# data of players to be removed who are near the disc
df = events[['End Frame','To']]
df['End Frame'] = df['End Frame'] - events['Start Frame'][0]
rows, cols = len(df), 1
data = np.full((rows, cols), np.nan)
removed_players = pd.DataFrame(data)
to_indexes = df[df['End Frame'].notna()]
for index, player in zip(to_indexes['End Frame'], to_indexes['To']):
    removed_players.iloc[int(index)] = player
removed_players.fillna(method='ffill', inplace=True)
removed_players.fillna('Player0', inplace=True)
removed_players = reduce_frame(removed_players, type='tracking')

# set parameter
params = mpc.default_model_params()

# load control and transition model
# EPV = mepv.load_EPV_grid('EPV_grid.csv')
# EPV = EPV / np.max(EPV)
# Trans_df = pd.read_csv('Transition_gauss.csv', header=None)
# Trans = np.array((Trans_df))
# Trans = Trans / np.max(Trans)

# set OBSO data
if args.end_ev == 0:
    args.end_ev = len(Metrica_df)
args.len_event = args.end_ev - args.start_ev
obso = np.zeros((args.len_event, 32, 50))
PPCF = np.zeros((args.len_event, 32, 50)) 
Transition = np.zeros((args.len_event, 32, 50)) 
disc_holders_loc = np.zeros((args.len_event, 2))
event_num0 = 0
for event_num, frame in tqdm(enumerate(Metrica_df['Start Frame'][args.start_ev:args.end_ev])):
    event_num += args.start_ev
    if np.isnan(frame):
        obso[event_num0] = np.zeros((32, 50))
        PPCF[event_num0] = np.zeros((32, 50))
        continue
    elif Metrica_df['Team'].loc[event_num]=='Home':
        # check attack direction 1st half or 2nd half
        if Metrica_df.loc[event_num]['Period']==1:
            direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==1], 'Home')
        elif Metrica_df.loc[event_num]['Period']==2:
            direction = mio.find_playing_direction(tracking_home[tracking_home['Period']==2], 'Home')
        #
        PPCF[event_num0], _, _, _, disc_holders_loc[event_num0] = mpc.generate_pitch_control_for_event(event_num, Metrica_df, tracking_home, tracking_away, removed_players, params, offsides=False, remove=True)
        #pass_frame = Metrica_df.loc[event_num]['Start Frame']
        #pass_team = Metrica_df.loc[event_num].Team
        #PPCF[event_num0], _, _, _ = mpc.generate_pitch_control_for_tracking(tracking_home, tracking_away, pass_frame, pass_team, params)

    elif Metrica_df['Team'].loc[event_num]=='Away': 
        # check attack direction 1st half or 2nd half
        if Metrica_df.loc[event_num]['Period']==1:
            direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==1], 'Away')
        elif Metrica_df.loc[event_num]['Period']==2:
            direction = mio.find_playing_direction(tracking_away[tracking_away['Period']==2], 'Away')
        PPCF[event_num0], _, _, _, _ = mpc.generate_pitch_control_for_event(event_num, Metrica_df, tracking_home, tracking_away, removed_players, params, offsides=False, remove=True)
    
    else:
        obso[event_num0] = np.zeros((32, 50))
        PPCF[event_num0] = np.zeros((32, 50))
        continue
    #obso[event_num0], Transition[event_num0] = obs.calc_obso(PPCF[event_num0], Trans, EPV, tracking_home.loc[frame], attack_direction=direction)
    event_num0 += 1 



# home_obso, away_obso = obs.calc_player_evaluate_match(obso, Metrica_df, tracking_home, tracking_away, args)

# # calculate onball obso
# home_onball_obso, away_onball_obso = obs.calc_onball_obso(Metrica_df, tracking_home, tracking_away, home_obso, away_obso, args)

# # save obso in home and away
# resultfolder = "../OBSO-data/"+args.data+'/'

# if args.data == 'metrica':
#     resultfolder += 'game_'+str(game_id) + '_event_'+str(args.start_ev)+"_"+str(args.end_ev)
# elif args.data == 'jleague':
#     resultfolder += game_date[game_id] + '_event_'+str(args.start_ev)+"_"+str(args.end_ev)

# if not os.path.exists(resultfolder):
#     os.makedirs(resultfolder)
#     print(f"Directory {resultfolder} created.")

# home_obso.to_pickle(resultfolder+'/home_obso.pkl')
# away_obso.to_pickle(resultfolder+'/away_obso.pkl')
# home_onball_obso.to_pickle(resultfolder+'/home_onball_obso.pkl')
# away_onball_obso.to_pickle(resultfolder+'/away_onball_obso.pkl')
# print(f"OBSO was saved at {resultfolder}.")



# create figures
# tracking_frame = 1
# attacking_team = 'Home'
# fig,ax = mviz.plot_pitchcontrol_for_tracking( tracking_frame, tracking_home, tracking_away, attacking_team, PPCF[event_num], annotate=True )
fig_dir = "./results"
'''
event_nums = range(args.start_ev,args.end_ev)
event_num0 = 4
fig,ax = mviz.plot_pitchcontrol_for_event(event_nums[event_num0], Metrica_df,  tracking_home, tracking_away, EPV, annotate=True, colorbar=True)
fig.savefig(fig_dir+"/OBSO/EPV_"+str(game_id)+"_"+str(event_nums[event_num0])+".png")
fig,ax = mviz.plot_pitchcontrol_for_event(event_nums[event_num0], Metrica_df,  tracking_home, tracking_away, Transition[event_num0], annotate=True, colorbar=True)
fig.savefig(fig_dir+"/OBSO/Transition_"+str(game_id)+"_"+str(event_nums[event_num0])+".png")
fig,ax = mviz.plot_pitchcontrol_for_event(event_nums[event_num0], Metrica_df,  tracking_home, tracking_away, PPCF[event_num0], annotate=True, colorbar=True)
fig.savefig(fig_dir+"/OBSO/PPCF_"+str(game_id)+"_"+str(event_nums[event_num0])+".png")
fig,ax = mviz.plot_pitchcontrol_for_event(event_nums[event_num0], Metrica_df,  tracking_home, tracking_away, obso[event_num0], annotate=True, vmax=0.2, colorbar=True)
fig.savefig(fig_dir+"/OBSO/OBSO_"+str(game_id)+"_"+str(event_nums[event_num0])+".png") 
print(f"OBSO figures were saved at {fig_dir}/OBSO.")
'''
np.save(f'./assets/{args.id}/PPCF_{args.id}', PPCF)
np.save(f'./assets/{args.id}/discholder_{args.id}', removed_players)
tracking_home.to_csv(f'./assets/{args.id}/tracking_home_{args.id}')
tracking_away.to_csv(f'./assets/{args.id}/tracking_away_{args.id}')
#np.savetxt(f'data/data0/event/_{args.id}', Metrica_df, delimiter=',', fmt='%s')
mviz.save_match_clip_OBSO(tracking_home, tracking_away, PPCF, f"{fig_dir}", f"PPCF_{args.id}", frames_per_second=30, include_player_velocities=True, vmax=1.0, colorbar=True)