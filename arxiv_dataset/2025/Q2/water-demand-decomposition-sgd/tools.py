"""
----------------------------------------------------------------------
Project: water-demand-decomposition-sgd
Filename: tools.py
Author: Roy Elkayam
Created: 2025-05-23
Description: find significant peaks in the data.
----------------------------------------------------------------------
"""

def find_significant_peaks(df):

    df = df.copy()
    col = df.columns[0]
    df.loc[:, 'Hour'] = [val.hour for val in df.index]

    df['dflow'] = df[col].diff()
    df.loc[df.index[0], 'dflow'] = 0
    df['direction'] = df['dflow'].apply(lambda x: 1 if x > 0 else (0 if x <= 0 else None))

    df['peaks'] = df['direction'].diff(-1)
    df.loc[df.index[0], 'peaks'] = 0 if df[col].iloc[0] < df[col].iloc[1] else 1  # Fixed logic error
    df.loc[df.index[-1], 'peaks'] = 0 if df[col].iloc[-1] < df[col].iloc[-2] else 1

    plato = df.loc[df['dflow'] == 0]
    plato = plato.copy()
    plato['events'] = plato['Hour'].diff()
    plato_events = plato.loc[plato['events'] > 1].index
    df.loc[plato_events, 'peaks'] = 1
    peaks = df.loc[df['peaks'] == 1, 'Hour'].values
    return peaks

