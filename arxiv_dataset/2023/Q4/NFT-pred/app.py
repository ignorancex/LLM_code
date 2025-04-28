import streamlit as st
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px

st.title("NFT Trade Analysis & Price Prediction")

# dropdown to select file
c = st.selectbox(
    "Select NFT collection to analyze", 
    ["Bored Ape Yacht Club"]
    )

collection_dict = {
    "Bored Ape Yacht Club": "bayc_cleaned.csv",
}

# get data, clean, save in cache
@st.cache_data(ttl=600)
def get_data(collection_name):
    file = collection_dict[collection_name]
    data = pd.read_csv(f"data/{file}", index_col=0)
    data['block_date'] = pd.to_datetime(data['block_date'])
    data['block_date'] = data['block_date'].dt.date
    return data

df = get_data(c)

#st.write(df.head())

# plot all trades
def plot_all(x_col,y_col,data=df):
    """ scatter plot price over time"""
    fig = px.scatter(
        data, 
        x=x_col, 
        y=y_col, 
        hover_data=["token_id","rarity_rank","trade_price","block_date"],
        labels={
            "block_date": "Date",
            "trade_price": "Trade Price (ETH)",
            "token_id": "Token ID",
            "rarity_rank": "Rarity Rank"
        },
        #title=f"{c} historical trades"
        )
    return fig

# plot trade by day
def plot_trades(date, data=df):
    """scatter plot price and rarity of each trade on a given date"""
    data = data[data["block_date"] == date]
    fig = px.scatter(
        data, 
        x="rarity_rank", 
        y="trade_price", 
        hover_data=["token_id","rarity_rank","trade_price"],
        labels={
            "rarity_rank": "Rarity Rank",
            "trade_price": "Trade Price (ETH)",
            "token_id": "Token ID"
        },
        title=f"{c} trades on {date}"
        )
    # add line for min and max price
    if len(data) > 0:
        fig.add_hline(y=data["price_min_eth"].values[0], line_dash="dash", line_color="green")
        fig.add_hline(y=data["price_max_eth"].values[0], line_dash="dash", line_color="red")
    return fig

with st.container():
    c11, c12 = st.columns([1,3])
    c11.subheader("All Trades")

    with c11:
        # plot all trades
        x = st.selectbox(
            "Select X-axis", 
            ["block_date", "rarity_rank"]
            )
        y = st.selectbox(
            "Select Y-axis", 
            ["trade_price", "trade_price_log"]
            )
            
    with c12:
        all_trades_fig = plot_all(x,y,data=df)
        st.plotly_chart(all_trades_fig)

with st.container():
    c21,c22 = st.columns([1,3])
    c21.subheader("Daily Trades")

    # select date to plot
    with c21:
        d = st.date_input(
            "Select date to plot", 
            value=dt.datetime.strptime("2023-09-30", '%Y-%m-%d').date(), 
            min_value=df["block_date"].min(), 
            max_value=df["block_date"].max()
            )    
    with c22:
        daily_trades_fig = plot_trades(date=d,data=df)
        st.plotly_chart(daily_trades_fig)
