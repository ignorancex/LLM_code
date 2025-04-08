import Metrica_Viz3 as mviz


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data0', help='data folder name')
parser.add_argument('--id', type=str, default='', help='data file name')
parser.add_argument('--frame', type=int, default=0, help='frame num')
parser.add_argument('--arg1', type=int, default=0, help='num of plot')


args = parser.parse_args()

def weight_by_distance(X, Y, center_x, center_y):
    distance = np.zeros((Y, X))
    for i in range(Y):
        for j in range(X):
            distance_to_point = np.sqrt(((i*20/32 - center_y)) ** 2 + ((j*54/50 - center_x)) ** 2)
            distance[i, j] = 1 - (distance_to_point / np.sqrt(20**2 + 54**2))
    return distance

def reduce_frame(data, type='tracking'):
    if type == 'tracking':
        data_reduced = data[data.index %2 == 0].copy()
        data_reduced.index = data_reduced.index // 2
    elif type == 'events':
        data_reduced = data[data['Start Frame']%2 == 0].copy()
        data_reduced.index = data_reduced.index // 2
        data_reduced['Start Frame'] = data_reduced['Start Frame'] // 2
        data_reduced['End Frame'] = data_reduced['End Frame'].div(2)
    return data_reduced

def get_discholder_loc():
    return

Y = 32
X = 50
area = np.zeros((Y, X-10))
area_end = np.ones((Y, 10))
end_upper_left = np.array([Y, X-10])
end_lower_left = np.array([-1, X-10])
for i in range(Y):
    for j in range(X-10):
        point = np.array([i, j])
        vec_1 = end_upper_left - point
        vec_2 = end_lower_left - point

        length_vec_a = np.linalg.norm(vec_1)
        length_vec_c = np.linalg.norm(vec_2)
        inner_product = np.inner(vec_1, vec_2)
        cos = inner_product / (length_vec_a * length_vec_c)

        rad = np.arccos(cos)

        area[i, j] = rad
area_min = area.min()
area_max = area.max()
area = (area - area_min) / (area_max - area_min)
area = np.concatenate((area, area_end), axis=1)


if args.id == '':
    fig, axs = plt.subplots(8, 1, figsize=(10, 30), tight_layout=True, sharex=True)
    N_max = 0
    n = 0
    for file_name, ax1 in zip(sorted(os.listdir('./assets/')), axs.flatten()):
        PPCF_path = f'./assets/{file_name}/PPCF_{file_name}.npy'
        discholder_path = f'./assets/{file_name}/discholder_{file_name}.npy'
        event_path = f'./assets/{file_name}/3on3_{file_name}_events.csv'
        tracking_path = f'./assets/{file_name}/3on3_{file_name}.csv'
        tracking_home_path = f'./assets/{file_name}/3on3_{file_name}_Home.csv'
        tracking_away_path = f'./assets/{file_name}/3on3_{file_name}_Away.csv'

        PPCF = np.load(PPCF_path)
        disc_holder = np.load(discholder_path, allow_pickle=True)
        events = pd.read_csv(event_path)
        tracking = pd.read_csv(tracking_path, header=[0,1,2], index_col=0)
        tracking_home = pd.read_csv(tracking_home_path)
        tracking_away = pd.read_csv(tracking_away_path)

        events = reduce_frame(events, type='events')
        tracking = reduce_frame(tracking, type='tracking')
        passes = events[events.Type == 'PASS']
        pass_frames = (passes['Start Frame'] - events['Start Frame'].iloc[0] + 1)
        recieve_frames = (passes['End Frame'] - events['Start Frame'].iloc[0] + 1)
        disc_X = (1 - events['Start X']) * 54
        disc_x = disc_X * 50 / 54
        disc_y = (1 - events['Start Y']) * 32
        N = PPCF.shape[0]
        N_max = max(N_max, N)
        frame = np.arange(N)
        score = np.zeros(N)
        area_disc = np.zeros(N)
        USO = np.zeros((N, 32, 50))
        for i in range(N):
            disc_holders_loc = tracking.iloc[i].loc[('0', disc_holder[i, 0][-1])]
            distance = weight_by_distance(X, Y, disc_holders_loc[0], disc_holders_loc[1])
            weight = area * distance
            for j in range(len(pass_frames)):
                if pass_frames.iloc[j] <= i and i < recieve_frames.iloc[j]:
                    ax1.axvline(i, 0, 1, linestyle='-', color='gray', alpha=0.2)
                    break
            USO[i] = PPCF[i] * weight
            score[i] = USO[i].max()
            area_disc[i] = area[int(disc_y[i]), int(disc_x[i])]

        np.save(f'data/data0/USO/{file_name[:-4]}', USO)

    #     ax1.plot(frame, score, label='Max of USO', color='red')
    #     ax1.plot(frame, area_disc, label=r'$w_{area}$', color='blue')
    #     ax1.set_xlabel('Frames')
    #     ax1.set_ylabel('Score')
    #     ax1.set_xticks(np.arange(0, N_max, 100))
    #     ax1.set_xticks(np.arange(50, N_max, 100), minor=True)
    #     ax1.set_yticks(np.arange(0, 1.01, 0.1))
    #     ax1.set_xlim(0, N_max)
    #     ax1.set_ylim(0, 1)
    #     ax1.grid(which='major', axis='both', linestyle='--')
    #     ax1.grid(which='minor', axis='x', linestyle='--')
    #     #ax1.set_aspect(250)
    #     ax1.set_title(f'{file_name[:-4]}')
    #     for i in range(len(pass_frames)):
    #         pass_frame = pass_frames.iloc[i]
    #         recieve_frame = recieve_frames.iloc[i]
    #         ax1.axvline(pass_frame, 0, 1, linestyle='-', color='gray')
    #         ax1.axvline(recieve_frame, 0, 1, linestyle='-', color='gray')
    #     ax2 = ax1.twinx()
    #     ax2.plot(frame, disc_X, label='disc_x', color='green')
    #     ax2.set_ylabel('disc X')
    #     ax2.set_yticks(np.arange(0, 54, 10))
    #     ax2.set_ylim(0, 54)
    #     lines_1, labels_1 = ax1.get_legend_handles_labels()
    #     lines_2, labels_2 = ax2.get_legend_handles_labels()

    #     lines = lines_1 + lines_2
    #     labels = labels_1 + labels_2
    #     ax1.legend(lines, labels)

    #     if n == args.arg1:
    #         if n == 3:
    #             fig2, axs2 = plt.subplots(1, 1, figsize=(4*N/500+2, 4), tight_layout=True)
    #         elif n == 7:
    #             fig2, axs2 = plt.subplots(1, 1, figsize=(4*N/500+1.55, 4), tight_layout=True)
    #         else:
    #             fig2, axs2 = plt.subplots(1, 1, figsize=(10, 4), tight_layout=True)
            
    #         axs2.plot(frame, score, label='USO Score', color='red')
    #         #axs2.plot(frame, area_disc, label=r'$w_{area}$ of disc position', color='blue')
    #         axs2.set_xlabel('Frames')
    #         axs2.set_ylabel('Score')
    #         axs2.set_xticks(np.arange(0, N_max, 100))
    #         axs2.set_xticks(np.arange(50, N_max, 100), minor=True)
    #         axs2.set_yticks(np.arange(0, 1.01, 0.1))
    #         axs2.set_xlim(0, len(score))
    #         axs2.set_ylim(0, 1)
    #         #axs2.grid(which='major', axis='both', linestyle='--')
    #         #axs2.grid(which='minor', axis='x', linestyle='--')
    #         axs2.set_aspect('auto', adjustable='box')
    #         for i in range(len(pass_frames)):
    #             pass_frame = pass_frames.iloc[i]
    #             print(pass_frame)
    #             recieve_frame = recieve_frames.iloc[i]
    #             axs2.axvline(pass_frame, 0, 1, linestyle='-', color='gray')
    #             axs2.axvline(recieve_frame, 0, 1, linestyle='-', color='gray')

    #         for i in range(N):
    #             for j in range(len(pass_frames)):
    #                 if pass_frames.iloc[j] <= i and i < recieve_frames.iloc[j]:
    #                     for k in range(4):
    #                         axs2.axvline(i+k*0.25, 0, 1, linestyle='-', color='gray', alpha=0.2)
    #                     if pass_frames.iloc[j] == i: #and j == len(pass_frames)-2:
    #                         axs2.axvline(pass_frames.iloc[j]-10, 0, 1, linestyle='--', color='blue', alpha=1)
    #                         axs2.axvline(pass_frames.iloc[j]-20, 0, 1, linestyle='--', color='blue', alpha=1)
    #                         axs2.axvline(pass_frames.iloc[j]-30, 0, 1, linestyle='--', color='blue', alpha=1)
    #                     break
    #         #ax3 = axs2.twinx()
    #         #ax3.plot(frame, disc_X, label='disc X', color='green')
    #         #ax3.set_ylabel('disc X')
    #         #ax3.set_yticks(np.arange(0, 54, 10))
    #         #ax3.set_ylim(0, 54)
    #         #lines_1, labels_1 = axs2.get_legend_handles_labels()
    #         #lines_2, labels_2 = ax3.get_legend_handles_labels()
    #         #lines = lines_1 + lines_2
    #         #labels = labels_1 + labels_2
    #         #axs2.legend(lines, labels)
    #         axs2.legend()
    #         if n < 4:
    #             axs2.set_title('USO')# (result:score)')
    #             fig2.savefig(f'figure/USO/{args.data}/usoscore_score_{args.arg1}.png')
    #         else:
    #             axs2.set_title('USO (result:turnover)')
    #             fig2.savefig(f'figure/USO/{args.data}/usoscore_turnover_{args.arg1}.png')

    #     n += 1

    # fig.savefig(f'figure/USO/{args.data}/_score.png')
    

else:
    file_name = args.id
    PPCF_path = f'./assets/{file_name}/PPCF_{file_name}.npy'
    discholder_path = f'./assets/{file_name}/discholder_{file_name}.npy'
    event_path = f'./assets/{file_name}/3on3_{file_name}_events.csv'
    tracking_path = f'./assets/{file_name}/3on3_{file_name}.csv'
    tracking_home_path = f'./assets/{file_name}/tracking_home_{file_name}'
    tracking_away_path = f'./assets/{file_name}/tracking_away_{file_name}'
    PPCF = np.load(PPCF_path)
    disc_holder = np.load(discholder_path, allow_pickle=True)
    events = pd.read_csv(event_path)
    tracking = pd.read_csv(tracking_path, header=[0,1,2], index_col=0)
    tracking_home = pd.read_csv(tracking_home_path)
    tracking_away = pd.read_csv(tracking_away_path)
    events = reduce_frame(events, type='events')
    N = PPCF.shape[0]
    area_disc = np.zeros(N)
    USO = np.zeros((N, 32, 50))
    for i in range(N):
        disc_holders_loc = tracking.iloc[i].loc[('0', disc_holder[i, 0][-1])]
        distance = weight_by_distance(X, Y, disc_holders_loc[0], disc_holders_loc[1])
        weight = area * distance
        USO[i] = PPCF[i] * weight

    USO /= USO.max()

    print(f'{file_name}')
    fig_dir = "./results"
    #mviz.save_match_clip_OBSO(tracking_home, tracking_away, PPCF, f"{fig_dir}/{args.data}", f"_PPCF_{file_name}", frames_per_second=30, include_player_velocities=True, vmax=1.0, colorbar=True)
    mviz.save_match_clip_OBSO(tracking_home, tracking_away, USO, f"{fig_dir}", f"USO_{file_name}", frames_per_second=30, include_player_velocities=True, vmax=1.0, colorbar=True, cm='Blues')
    #fig,ax = mviz.plot_pitchcontrol_for_event(args.frame, events,  tracking_home, tracking_away, USO[args.frame], annotate=True, colorbar=True)
    #fig.savefig(f'{fig_dir}/{args.data}/uso_{args.id}_{args.frame}.png')

    #fig,ax = mviz.plot_pitchcontrol_for_event(args.frame, events,  tracking_home, tracking_away, PPCF[args.frame], annotate=True, colorbar=True)
    #fig.savefig(f'{fig_dir}/{args.data}/ppcf_{args.id}_{args.frame}.png')

