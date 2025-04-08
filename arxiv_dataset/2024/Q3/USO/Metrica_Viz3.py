#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 09:10:58 2020

Module for visualising Metrica tracking and event data

Data can be found at: https://github.com/metrica-sports/sample-data

UPDATE for tutorial 4: plot_pitchcontrol_for_event no longer requires 'xgrid' and 'ygrid' as inputs. 

@author: Laurie Shaw (@EightyFivePoint)
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import pickle
import pandas as pd
import matplotlib.animation as animation
import Metrica_IO as mio
import obso_player as obs
import mpl_toolkits.axes_grid1
from tqdm import tqdm

def plot_pitch( field_dimen = (54.0,20.0), field_color ='green', linewidth=2, markersize=20):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1]
    # plot half way line # center circle
    ax.plot([-17,-17],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.plot([17,17],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax

def plot_frame( hometeam, awayteam, figax=None, team_colors=('b','r'), field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False ):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Metrica tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    
    Parameters
    -----------
        hometeam: row (i.e. instant) of the home team tracking data frame
        awayteam: row of the away team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    for team,color in zip( [hometeam,awayteam], team_colors) :
        x_columns = [c for c,v in zip(team.keys(),team.values) if c[-2:].lower()=='_x' and c!='ball_x' and ~np.isnan(v)] # column header for player x positions
        y_columns = [c for c,v in zip(team.keys(),team.values) if c[-2:].lower()=='_y' and c!='ball_y' and ~np.isnan(v)] # column header for player y positions
        ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
            ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if annotate:
            # [ ax.text( team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) if not ( np.isnan(team[x]) or np.isnan(team[y]) ) ] 
            [ ax.text( team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color  ) for x,y in zip(x_columns,y_columns) ] 

    # plot ball
    ax.plot( hometeam['ball_x'], hometeam['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
    return fig,ax
    
def save_match_clip(hometeam,awayteam, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False):
    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( hometeam.index==awayteam.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = hometeam.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen, field_color='white')
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [hometeam.loc[i],awayteam.loc[i]], team_colors) :
                x_columns = [c for c,v in zip(team.keys(),team.values) if c[-2:].lower()=='_x' and c!='ball_x' and ~np.isnan(v)] # column header for player x positions
                y_columns = [c for c,v in zip(team.keys(),team.values) if c[-2:].lower()=='_y' and c!='ball_y' and ~np.isnan(v)] # column header for player y positions

                objs, = ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
                # annotate player number
                if annotate:
                    for x,y in zip(x_columns,y_columns):
                        if np.isnan(team[x]) or np.isnan(team[y]):  
                            continue 
                        else:
                            objs = ax.text(team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color) 
                            figobjs.append(objs)
            # plot ball
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
            figobjs.append(objs)
            # include match time at the top
            frame_minute =  int( team['Time [s]']/60. )
            frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)    


def plot_events( events, figax=None, field_dimen = (54.0,20.0), indicators = ['Marker','Arrow'], color='r', marker_style = 'o', alpha = 0.5, annotate=False):
    """ plot_events( events )
    
    Plots Metrica event positions on a football pitch. event data can be a single or several rows of a data frame. All distances should be in meters.
    
    Parameters
    -----------
        events: row (i.e. instant) of the home team tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        indicators: List containing choices on how to plot the event. 'Marker' places a marker at the 'Start X/Y' location of the event; 'Arrow' draws an arrow from the start to end locations. Can choose one or both.
        color: color of indicator. Default is 'r' (red)
        marker_style: Marker type used to indicate the event position. Default is 'o' (filled ircle).
        alpha: alpha of event marker. Default is 0.5    
        annotate: Boolean determining whether text annotation from event data 'Type' and 'From' fields is shown on plot. Default is False.
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """

    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax 
    for i,row in events.iterrows():
        if 'Marker' in indicators:
            ax.plot(  row['Start X'], row['Start Y'], color+marker_style, alpha=alpha )
        if 'Arrow' in indicators:
            ax.annotate("", xy=row[['End X','End Y']], xytext=row[['Start X','Start Y']], alpha=alpha, arrowprops=dict(alpha=alpha,width=0.5,headlength=4.0,headwidth=4.0,color=color),annotation_clip=False)
        if annotate:
            textstring = row['Type'] + ': ' + row['From']
            ax.text( row['Start X'], row['Start Y'], textstring, fontsize=10, color=color)
    return fig,ax

def plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (54.0,20.0), vmin=0.0, vmax=1.0, colorbar=False):
    """ plot_pitchcontrol_for_event( event_id, events,  tracking_home, tracking_away, PPCF )
    
    Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    NB: this function no longer requires xgrid and ygrid as an input
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """    

    # pick a pass at which to generate the pitch control surface
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    # plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
    # plot pitch control surface
    if pass_team=='Home':
        cmap = cm.bwr_r
        #cmap = 'Blues'
    else:
        cmap = 'Blues'
    im = ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=vmin,vmax=vmax,cmap=cmap,alpha=0.5)
    
    if colorbar:
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        try: fig.colorbar(im, cax=cax)
        except: import pdb; pdb.set_trace()
    return fig,ax


def plot_pitchcontrol_for_tracking( tracking_frame, tracking_home, tracking_away, attacking_team, PPCF, alpha = 0.7, include_player_velocities=True, annotate=False, field_dimen = (54.0,20.0), vmin=0.0, vmax=1.0, colorbar=False):
    """ plot_pitchcontrol_for_event( tracking_frame, tracking_home, tracking_away, attacking_team, PPCF )
    
    Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        tracking_frame: number of frame
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        attacking_team: Home or Away
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    NB: this function no longer requires xgrid and ygrid as an input
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """    

    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( tracking_home.loc[tracking_frame], tracking_away.loc[tracking_frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    # plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
    
    # plot pitch control surface
    if attacking_team=='Home':
        cmap = 'Reds'
    else:
        cmap = 'Blues'
    im = ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=vmin,vmax=vmax,cmap=cmap,alpha=0.5)

    if colorbar:
        fig.colorbar(im)
    return fig,ax


def plot_true_pre(attack_true, attack_pre, defense_true, defense_pre, obso_true, obso_pre, path='./', fname='plot.png', annotate=False):
    '''
    # Args
    attack_true: true data in attack team pd.Series
    attack_pre: predict data in attack team pd.Series
    defense_true: true data in defense team pd.Series
    defense_pre: predict data in defense team pd.Series
    obso_true: obso map in true data, shape = (32, 50)
    obso_pre : obso map in predict data, shape=(32, 50)
    path: path name to save figure file
    fname: file name, format png
    # Returns
    None
    '''
    # set some parameter
    field_dimen = (54.0,20.0)
    alpha = 0.7
    # plot frame
    fig, ax = plot_pitch(field_color='white',field_dimen=field_dimen)
    plot_frame(attack_true, defense_true, figax=(fig,ax), PlayerAlpha=alpha, annotate=annotate)
    # plot predict players
    ax.plot(attack_pre['Home_1_x'], attack_pre['Home_1_y'], color='coral',marker='o', markersize=10,alpha=alpha)
    ax.plot(defense_pre['Away_1_x'], defense_pre['Away_1_y'], color='skyblue', marker='o',markersize=10,alpha=alpha)
    ax.plot(defense_pre['Away_2_x'], defense_pre['Away_2_y'], color='skyblue', marker='o', markersize=10,alpha=alpha)
    # predict annotate
    if annotate:
        ax.text(attack_pre['Home_1_x']+0.5, attack_pre['Home_1_y']+0.5, '1', fontsize=10, color='coral')
        ax.text(defense_pre['Away_1_x']+0.5, defense_pre['Away_1_y']+0.5, '1', fontsize=10, color='skyblue')
        ax.text(defense_pre['Away_2_x']+0.5, defense_pre['Away_2_y']+0.5, '2', fontsize=10, color='skyblue')
    # write obso of A1 and A2
    # a1_true_pos = [attack_true['Home_1_x'], attack_true['Home_1_y']]
    # a1_pre_pos = [attack_pre['Home_1_x'], attack_pre['Home_1_y']]
    a2_true_pos = [attack_true['Home_2_x'], attack_true['Home_2_y']]
    a2_pre_pos = [attack_pre['Home_2_x'], attack_pre['Home_2_y']]
    # a1_true_ev = obs.calc_player_evaluate(a1_true_pos, obso_true)
    # a1_pre_ev = obs.calc_player_evaluate(a1_pre_pos, obso_pre)
    a2_true_ev = obs.calc_player_evaluate(a2_true_pos, obso_true)
    a2_pre_ev = obs.calc_player_evaluate(a2_pre_pos, obso_pre)
    # ax.text(0, field_dimen[1]/2.+1, 'A1 true obso:{:.4f}'.format(a1_true_ev), fontsize=18)
    # ax.text(0, field_dimen[1]/2.-2, 'A1 predict obso:{:.4f}'.format(a1_pre_ev), fontsize=18)
    ax.text(1, field_dimen[1]/2.-4, 'A2 true obso:{:.4f}'.format(a2_true_ev), fontsize=18)
    ax.text(1, field_dimen[1]/2.-8, 'A2 predict obso:{:.4f}'.format(a2_pre_ev), fontsize=18)
    ax.text(1, field_dimen[1]/2.-12, 'A1 evaluation:{:.4f}'.format(a2_true_ev-a2_pre_ev), fontsize=18)
    
    # save figure
    fig.savefig(path+fname)


def plot_EPV_for_event( event_id, events, tracking_home, tracking_away, PPCF, EPV, alpha = 0.7, include_player_velocities=True, annotate=False, autoscale=0.1, contours=False, field_dimen = (54.0,20.0), vmin=0.0, vmax=0.6):
    """ plot_EPV_for_event( event_id, events,  tracking_home, tracking_away, PPCF, EPV, alpha, include_player_velocities, annotate, autoscale, contours, field_dimen)
    
    Plots the EPVxPitchControl surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: (entire) tracking DataFrame for the Home team
        tracking_away: (entire) tracking DataFrame for the Away team
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        EPV: Expected Possession Value surface. EPV is the probability that a possession will end with a goal given the current location of the ball. 
             The EPV surface is saved in the FoT github repo and can be loaded using Metrica_EPV.load_EPV_grid()
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        autoscale: If True, use the max of surface to define the colorscale of the image. If set to a value [0-1], uses this as the maximum of the color scale.
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """    

    # pick a pass at which to generate the pitch control surface
    pass_frame = events.loc[event_id]['Start Frame']
    pass_team = events.loc[event_id].Team
    
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( tracking_home.loc[pass_frame], tracking_away.loc[pass_frame], figax=(fig,ax), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    plot_events( events.loc[event_id:event_id], figax = (fig,ax), indicators = ['Marker','Arrow'], annotate=False, color= 'k', alpha=1 )
       
    # plot pitch control surface
    if pass_team=='Home':
        cmap = 'Reds'
        lcolor = 'r'
        EPV = np.fliplr(EPV) if mio.find_playing_direction(tracking_home,'Home') == -1 else EPV
    else:
        cmap = 'Blues'
        lcolor = 'b'
        EPV = np.fliplr(EPV) if mio.find_playing_direction(tracking_away,'Away') == -1 else EPV
    
    EPVxPPCF = PPCF*EPV
    
    if autoscale is True:
        vmax = np.max(EPVxPPCF)*2.
    elif autoscale>=0 and autoscale<=1:
        vmax = autoscale
    else:
        assert False, "'autoscale' must be either {True or between 0 and 1}"
        
    ax.imshow(np.flipud(EPVxPPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=vmin,vmax=vmax,cmap=cmap,alpha=0.7)
    
    if contours:
        ax.contour( EPVxPPCF,extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),levels=np.array([0.75])*np.max(EPVxPPCF),colors=lcolor,alpha=1.0)
    
    return fig,ax

def plot_EPV(EPV,field_dimen=(54.0,20.0),attack_direction=1, vmin=0, vmax=0.6):
    """ plot_EPV( EPV,  field_dimen, attack_direction)
    
    Plots the pre-generated Expected Possession Value surface 
    
    Parameters
    -----------
        EPV: The 32x50 grid containing the EPV surface. EPV is the probability that a possession will end with a goal given the current location of the ball. 
             The EPV surface is saved in the FoT github repo and can be loaded using Metrica_EPV.load_EPV_grid()
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        attack_direction: Sets the attack direction (1: left->right, -1: right->left)
            
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """    
    if attack_direction==-1:
        # flip direction of grid if team is attacking right->left
        EPV = np.fliplr(EPV)
    ny,nx = EPV.shape
    # plot a pitch
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    # overlap the EPV surface
    ax.imshow(EPV, extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),vmin=vmin,vmax=vmax,cmap='Blues',alpha=0.6)

    return fig, ax
    

def save_match_clip_OBSO(hometeam, awayteam, OBSO, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('b','r'), field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False, vmin=0,vmax=0.3, colorbar=False, cm=cm.bwr_r):
    assert np.all(hometeam.index == awayteam.index), "Home and away team Dataframe indices must be the same"
    index = hometeam.index
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' + fname + '.mp4'
    
    if figax is None:
        fig, ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    else:
        fig, ax = figax
    fig.set_tight_layout(True)
    
    # 初期化
    player_objs = []
    ball_objs = []
    text_objs = []
    quiver_objs = []
    for team, color in zip([hometeam.loc[index[0]], awayteam.loc[index[0]]], team_colors):
        x_columns = [c for c in team.keys() if c[-2:].lower() == '_x' and c != 'ball_x']
        y_columns = [c for c in team.keys() if c[-2:].lower() == '_y' and c != 'ball_y']
        player_objs.append(ax.plot(team[x_columns], team[y_columns], color + 'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)[0])
        ball_objs.append(ax.plot(team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)[0])
        if include_player_velocities:
            vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]
            vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]
            quiver_objs.append(ax.quiver(team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10., width=0.0015, headlength=5, headwidth=3, alpha=PlayerAlpha))
        if annotate:
            for x, y in zip(x_columns, y_columns):
                if not np.isnan(team[x]) and not np.isnan(team[y]):
                    text_objs.append(ax.text(team[x] + 0.5, team[y] + 0.5, x.split('_')[1], fontsize=10, color=color))
    
    cmap = cm
    obs_map = ax.imshow(np.flipud(OBSO[0]), extent=(-field_dimen[0] / 2., field_dimen[0] / 2., -field_dimen[1] / 2., field_dimen[1] / 2.), interpolation='spline36', vmin=vmin, vmax=vmax, cmap=cmap, alpha=0.7)
    if colorbar:
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
        cax = divider.append_axes('right', '5%', pad='3%')
        fig.colorbar(obs_map, cax=cax)
    
    with writer.saving(fig, fname, 100):
        for frame, i in tqdm(enumerate(index), total=len(index), desc='Generating movie'):
            for team, color, player_obj, ball_obj, quiver_obj in zip([hometeam.loc[i], awayteam.loc[i]], team_colors, player_objs, ball_objs, quiver_objs):
                x_columns = [c for c in team.keys() if c[-2:].lower() == '_x' and c != 'ball_x']
                y_columns = [c for c in team.keys() if c[-2:].lower() == '_y' and c != 'ball_y']
                player_obj.set_data(team[x_columns], team[y_columns])
                ball_obj.set_data(team['ball_x'], team['ball_y'])
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]
                    quiver_obj.set_offsets(np.c_[team[x_columns], team[y_columns]])
                    quiver_obj.set_UVC(team[vx_columns], team[vy_columns])
                if annotate:
                    for text_obj, x, y in zip(text_objs, x_columns, y_columns):
                        if not np.isnan(team[x]) and not np.isnan(team[y]):
                            text_obj.set_position((team[x] + 0.5, team[y] + 0.5))
                            text_obj.set_text(x.split('_')[1])
            obs_map.set_data(np.flipud(OBSO[frame]))
            frame_minute = int(team['Time [s]'] / 60.)
            frame_second = (team['Time [s]'] / 60. - frame_minute) * 60.
            timestring = "%1.2f, %d" % (frame_second, frame)
            time_text = ax.text(-2.5, field_dimen[1] / 2. + 1., timestring, fontsize=14)
            writer.grab_frame()
            time_text.remove()
    
    plt.clf()
    plt.close(fig)


def save_match_clip_custom(attack_true, attack_pre, defense_true, defense_pre, OBSO_true, OBSO_pre,fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), 
                        field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False,annotate_pre=False,pkl_save=False,pkl_path='./'):

    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( attack_true.index==defense_true.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = attack_true.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    obso_frame = 0
    a2_pre_evs = []
    a2_true_evs = []
    a1_pre_evs = []
    a1_true_evs = []
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [attack_true.loc[i],defense_true.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
                objs, = ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
                # annotate player number
                if annotate:
                    for x,y in zip(x_columns,y_columns):
                        if np.isnan(team[x]) or np.isnan(team[y]):  
                            continue 
                        else:
                            objs = ax.text(team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color) 
                            figobjs.append(objs)
            # plot predict players
            objs, = ax.plot(attack_pre.loc[i]['Home_1_x'], attack_pre.loc[i]['Home_1_y'], color='coral',marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            objs, = ax.plot(defense_pre.loc[i]['Away_1_x'], defense_pre.loc[i]['Away_1_y'], color='skyblue', marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            objs, = ax.plot(defense_pre.loc[i]['Away_2_x'], defense_pre.loc[i]['Away_2_y'], color='skyblue', marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            # predict annotate
            if annotate:
                objs = ax.text(attack_pre.loc[i]['Home_1_x']+0.5, attack_pre.loc[i]['Home_1_y']+0.5, '1', fontsize=10, color='coral')
                figobjs.append(objs)
                objs = ax.text(defense_pre.loc[i]['Away_1_x']+0.5, defense_pre.loc[i]['Away_1_y']+0.5, '1', fontsize=10, color='skyblue')
                figobjs.append(objs)
                objs = ax.text(defense_pre.loc[i]['Away_2_x']+0.5, defense_pre.loc[i]['Away_2_y']+0.5, '2', fontsize=10, color='skyblue')
                figobjs.append(objs)
            # annotate using presentation
            else:
                if annotate_pre:
                    objs = ax.text(attack_true.loc[i]['Home_1_x']+0.5, attack_true.loc[i]['Home_1_y']+0.5, 'A1', fontsize=10, color='r')
                    figobjs.append(objs)
                    objs = ax.text(attack_true.loc[i]['Home_2_x']+0.5, attack_true.loc[i]['Home_2_y']+0.5, 'A2', fontsize=10, color='r')
                    figobjs.append(objs)
                    objs = ax.text(defense_true.loc[i]['Away_1_x']+0.5, defense_true.loc[i]['Away_1_y']+0.5, 'D1', fontsize=10, color='b')
                    figobjs.append(objs)
                    objs = ax.text(defense_true.loc[i]['Away_2_x']+0.5, defense_true.loc[i]['Away_2_y']+0.5, 'D2', fontsize=10, color='b')
                    figobjs.append(objs)
                    objs = ax.text(attack_pre.loc[i]['Home_1_x']+0.5, attack_pre.loc[i]['Home_1_y']+0.5, 'A1', fontsize=10, color='coral')
                    figobjs.append(objs)
                    objs = ax.text(defense_pre.loc[i]['Away_1_x']+0.5, defense_pre.loc[i]['Away_1_y']+0.5, 'D1', fontsize=10, color='skyblue')
                    figobjs.append(objs)
                    objs = ax.text(defense_pre.loc[i]['Away_2_x']+0.5, defense_pre.loc[i]['Away_2_y']+0.5, 'D2', fontsize=10, color='skyblue')
                    figobjs.append(objs)
                    objs = ax.text(1, field_dimen[1]/2.-4, 'A2 true obso:{:.4f}'.format(a2_true_ev), fontsize=18)
                    figobjs.append(objs)
                    objs = ax.text(1, field_dimen[1]/2.-8, 'A2 predict obso:{:.4f}'.format(a2_pre_ev), fontsize=18)
                    figobjs.append(objs)
                    objs = ax.text(1, field_dimen[1]/2.-12, 'A1 evaluation:{:.4f}'.format(a2_true_ev-a2_pre_ev), fontsize=18)
                    figobjs.append(objs)
            # plot ball
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
            figobjs.append(objs)
            # write obso of A1 and A2
            a1_true_pos = [attack_true.loc[i]['Home_1_x'], attack_true.loc[i]['Home_1_y']]
            a1_pre_pos = [attack_pre.loc[i]['Home_1_x'], attack_pre.loc[i]['Home_1_y']]
            a2_true_pos = [attack_true.loc[i]['Home_2_x'], attack_true.loc[i]['Home_2_y']]
            a2_pre_pos = [attack_pre.loc[i]['Home_2_x'], attack_pre.loc[i]['Home_2_y']]
            a1_true_ev = obs.calc_player_evaluate(a1_true_pos, OBSO_true[obso_frame])
            a1_pre_ev = obs.calc_player_evaluate(a1_pre_pos, OBSO_pre[obso_frame])
            a2_true_ev = obs.calc_player_evaluate(a2_true_pos, OBSO_true[obso_frame])
            a2_pre_ev = obs.calc_player_evaluate(a2_pre_pos, OBSO_pre[obso_frame])
            # objs = ax.text(-52, field_dimen[1]/2.+1, 'A1 true obso:{:.4f}'.format(a1_true_ev), fontsize=14)
            # figobjs.append(objs)
            # objs = ax.text(-52, field_dimen[1]/2.-2, 'A1 predict obso:{:.4f}'.format(a1_pre_ev), fontsize=14)
            # figobjs.append(objs)
            a2_pre_evs.append(a2_pre_ev)
            a2_true_evs.append(a2_true_ev)
            a1_pre_evs.append(a1_pre_ev)
            a1_true_evs.append(a1_true_ev)
            obso_frame += 1
            # include match time at the top
            # frame_minute =  int( team['Time [s]']/60. )
            # frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            # timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            # objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            # figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)    
    # save evaluation
    if pkl_save:
        ev_df = pd.DataFrame(columns=['a2_true_ev', 'a2_pre_ev', 'a1_true_ev', 'a1_pre_ev'])
        ev_df['a2_true_ev'] = a2_true_evs
        ev_df['a2_pre_ev'] = a2_pre_evs
        ev_df['a1_true_ev'] = a1_true_evs
        ev_df['a1_pre_ev'] = a1_pre_evs
        ev_df.to_pickle(pkl_path)


def save_match_clip_custom_only2(attack_true, attack_pre, defense_true, defense_pre, OBSO_true, OBSO_pre,fpath, fname='clip_test', figax=None, frames_per_second=25, 
                            team_colors=('r','b'), field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False,pkl_save=False,pkl_path='./'):
    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( attack_true.index==defense_true.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = attack_true.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    obso_frame = 0
    a2_pre_evs = []
    a2_true_evs = []
    a1_pre_evs = []
    a1_true_evs = []
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [attack_true.loc[i],defense_true.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
                objs, = ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
                # annotate player number
                if annotate:
                    for x,y in zip(x_columns,y_columns):
                        if np.isnan(team[x]) or np.isnan(team[y]):  
                            continue 
                        else:
                            objs = ax.text(team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color) 
                            figobjs.append(objs)
            # plot predict players
            objs, = ax.plot(attack_pre.loc[i]['Home_1_x'], attack_pre.loc[i]['Home_1_y'], color='coral',marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            objs, = ax.plot(defense_pre.loc[i]['Away_1_x'], defense_pre.loc[i]['Away_1_y'], color='skyblue', marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            # predict annotate
            if annotate:
                objs = ax.text(attack_pre.loc[i]['Home_1_x']+0.5, attack_pre.loc[i]['Home_1_y']+0.5, '1', fontsize=10, color='coral')
                figobjs.append(objs)
                objs = ax.text(defense_pre.loc[i]['Away_1_x']+0.5, defense_pre.loc[i]['Away_1_y']+0.5, '1', fontsize=10, color='skyblue')
                figobjs.append(objs)
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
            figobjs.append(objs)
            # write obso of A1 and A2
            a1_true_pos = [attack_true.loc[i]['Home_1_x'], attack_true.loc[i]['Home_1_y']]
            a1_pre_pos = [attack_pre.loc[i]['Home_1_x'], attack_pre.loc[i]['Home_1_y']]
            a2_true_pos = [attack_true.loc[i]['Home_2_x'], attack_true.loc[i]['Home_2_y']]
            a2_pre_pos = [attack_pre.loc[i]['Home_2_x'], attack_pre.loc[i]['Home_2_y']]
            a1_true_ev = obs.calc_player_evaluate(a1_true_pos, OBSO_true[obso_frame])
            a1_pre_ev = obs.calc_player_evaluate(a1_pre_pos, OBSO_pre[obso_frame])
            a2_true_ev = obs.calc_player_evaluate(a2_true_pos, OBSO_true[obso_frame])
            a2_pre_ev = obs.calc_player_evaluate(a2_pre_pos, OBSO_pre[obso_frame])
            objs = ax.text(0, field_dimen[1]/2., 'A2 true obso:{:.4f}'.format(a2_true_ev), fontsize=18)
            figobjs.append(objs)
            objs = ax.text(0, field_dimen[1]/2.-4, 'A2 predict obso:{:.4f}'.format(a2_pre_ev), fontsize=18)
            figobjs.append(objs)
            objs = ax.text(0, field_dimen[1]/2.-8, 'A1 evaluation:{:.4f}'.format(a2_true_ev-a2_pre_ev), fontsize=18)
            figobjs.append(objs)
            a2_pre_evs.append(a2_pre_ev)
            a2_true_evs.append(a2_true_ev)
            a1_pre_evs.append(a1_pre_ev)
            a1_true_evs.append(a1_true_ev)
            obso_frame += 1
            # include match time at the top
            # frame_minute =  int( team['Time [s]']/60. )
            # frame_second =  ( team['Time [s]']/60. - frame_minute ) * 60.
            # timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
            # objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
            # figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig)    
    if pkl_save:
        ev_df = pd.DataFrame(columns=['a2_true_ev', 'a2_pre_ev', 'a1_true_ev', 'a1_pre_ev'])
        ev_df['a2_true_ev'] = a2_true_evs
        ev_df['a2_pre_ev'] = a2_pre_evs
        ev_df['a1_true_ev'] = a1_true_evs
        ev_df['a1_pre_ev'] = a1_pre_evs
        ev_df.to_pickle(pkl_path)


def save_video_pre(attack_true, attack_pre, defense_true, defense_pre,fpath, fname='clip_test', figax=None, frames_per_second=8, team_colors=('r','b'), 
                        field_dimen = (54.0,20.0), include_player_velocities=False, PlayerMarkerSize=10, PlayerAlpha=0.7, annotate=False,annotate_pre=False):

    """ save_match_clip( hometeam, awayteam, fpath )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        hometeam: home team tracking data DataFrame. Movie will be created from all rows in the DataFrame
        awayteam: away team tracking data DataFrame. The indices *must* match those of the hometeam DataFrame
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)

    """
    # check that indices match first
    assert np.all( attack_true.index==defense_true.index ), "Home and away team Dataframe indices must be the same"
    # in which case use home team index
    index = attack_true.index
    # Set figure and movie settings
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='Tracking Data', artist='Matplotlib', comment='Metrica tracking data clip')
    writer = FFMpegWriter(fps=frames_per_second, metadata=metadata)
    fname = fpath + '/' +  fname + '.mp4' # path and filename
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_color='white', field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    with writer.saving(fig, fname, 100):
        for i in index:
            figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
            for team,color in zip( [attack_true.loc[i],defense_true.loc[i]], team_colors) :
                x_columns = [c for c in team.keys() if c[-2:].lower()=='_x' and c!='ball_x'] # column header for player x positions
                y_columns = [c for c in team.keys() if c[-2:].lower()=='_y' and c!='ball_y'] # column header for player y positions
                objs, = ax.plot( team[x_columns], team[y_columns], color+'o', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
                figobjs.append(objs)
                if include_player_velocities:
                    vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns] # column header for player x positions
                    vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns] # column header for player y positions
                    objs = ax.quiver( team[x_columns], team[y_columns], team[vx_columns], team[vy_columns], color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                    figobjs.append(objs)
                # annotate player number
                if annotate:
                    for x,y in zip(x_columns,y_columns):
                        if np.isnan(team[x]) or np.isnan(team[y]):  
                            continue 
                        else:
                            objs = ax.text(team[x]+0.5, team[y]+0.5, x.split('_')[1], fontsize=10, color=color) 
                            figobjs.append(objs)
            # plot predict players
            objs, = ax.plot(attack_pre.loc[i]['Home_1_x'], attack_pre.loc[i]['Home_1_y'], color='coral',marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            objs, = ax.plot(defense_pre.loc[i]['Away_1_x'], defense_pre.loc[i]['Away_1_y'], color='skyblue', marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            objs, = ax.plot(defense_pre.loc[i]['Away_2_x'], defense_pre.loc[i]['Away_2_y'], color='skyblue', marker='o', markersize=PlayerMarkerSize, alpha=PlayerAlpha)
            figobjs.append(objs)
            # predict annotate
            if annotate:
                objs = ax.text(attack_pre.loc[i]['Home_1_x']+0.5, attack_pre.loc[i]['Home_1_y']+0.5, '1', fontsize=10, color='coral')
                figobjs.append(objs)
                objs = ax.text(defense_pre.loc[i]['Away_1_x']+0.5, defense_pre.loc[i]['Away_1_y']+0.5, '1', fontsize=10, color='skyblue')
                figobjs.append(objs)
                objs = ax.text(defense_pre.loc[i]['Away_2_x']+0.5, defense_pre.loc[i]['Away_2_y']+0.5, '2', fontsize=10, color='skyblue')
                figobjs.append(objs)
            # annotate using presentation
            else:
                if annotate_pre:
                    objs = ax.text(attack_true.loc[i]['Home_1_x']+0.5, attack_true.loc[i]['Home_1_y']+0.5, 'A1', fontsize=12, color='r')
                    figobjs.append(objs)
                    objs = ax.text(attack_true.loc[i]['Home_2_x']-2.0, attack_true.loc[i]['Home_2_y']+0.5, 'A2', fontsize=12, color='r')
                    figobjs.append(objs)
                    objs = ax.text(defense_true.loc[i]['Away_1_x']+0.5, defense_true.loc[i]['Away_1_y']+0.5, 'D1', fontsize=12, color='b')
                    figobjs.append(objs)
                    objs = ax.text(defense_true.loc[i]['Away_2_x']+0.5, defense_true.loc[i]['Away_2_y']+0.5, 'D2', fontsize=12, color='b')
                    figobjs.append(objs)
                    objs = ax.text(attack_pre.loc[i]['Home_1_x']+0.5, attack_pre.loc[i]['Home_1_y']+0.5, 'A1', fontsize=12, color='coral')
                    figobjs.append(objs)
                    objs = ax.text(defense_pre.loc[i]['Away_1_x']+0.5, defense_pre.loc[i]['Away_1_y']+0.5, 'D1', fontsize=12, color='skyblue')
                    figobjs.append(objs)
                    objs = ax.text(defense_pre.loc[i]['Away_2_x']+0.5, defense_pre.loc[i]['Away_2_y']+0.5, 'D2', fontsize=12, color='skyblue')
                    figobjs.append(objs)
                    
            # plot ball
            objs, = ax.plot( team['ball_x'], team['ball_y'], 'ko', markersize=6, alpha=1.0, linewidth=0)
            figobjs.append(objs)
            writer.grab_frame()
            # Delete all axis objects (other than pitch lines) in preperation for next frame
            for figobj in figobjs:
                figobj.remove()
    print("done")
    plt.clf()
    plt.close(fig) 







