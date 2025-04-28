import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import itertools
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS

# Constants
SENTIMENT_COLUMN = 'sentiment_deepseek'
CHUNK_SIZE = 1000
TRAIN_START_DATE = '2013-01-01'
TRAIN_END_DATE = '2018-12-31'
TRADE_START_DATE = '2019-01-01'
TRADE_END_DATE = '2023-12-31'

def load_seagen_data():
    """Load and process Seagen stock data"""
    df_seagen = pd.read_csv('seagen.csv')
    
    # Rename columns
    df_seagen.rename(columns={
        'Date': 'date',
        'Price': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume',
        'Change %': 'change_percent'
    }, inplace=True)
    
    # Add tic column
    df_seagen['tic'] = 'SGEN'
    
    # Convert volume strings to numeric
    def convert_volume(vol_str):
        if isinstance(vol_str, str):
            if vol_str.endswith('M'):
                return float(vol_str.replace('M', '')) * 1e6
            elif vol_str.endswith('K'):
                return float(vol_str.replace('K', '')) * 1e3
        return float(vol_str)
    
    df_seagen['volume'] = df_seagen['volume'].apply(convert_volume)
    df_seagen['date'] = pd.to_datetime(df_seagen['date'], format='%m/%d/%Y')
    
    return df_seagen.drop(columns=['change_percent'])

def prepare_stock_data(nasdaq_tickers):
    """Prepare stock data including Seagen"""
    # Download data for other stocks
    df_raw = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=[tic for tic in nasdaq_tickers if tic != "SGEN"]
    ).fetch_data()
    
    # Load and combine Seagen data
    df_seagen = load_seagen_data()
    df_combined = pd.concat([df_raw, df_seagen], ignore_index=True)
    df_combined.sort_values(by='date', inplace=True)
    
    # Convert date to string format for consistency
    df_combined['date'] = df_combined['date'].dt.strftime('%Y-%m-%d')
    
    return df_combined

def process_data(df_raw):
    """Process raw data with feature engineering"""
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False
    )
    
    processed = fe.preprocess_data(df_raw)
    
    # Create full processed dataset
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date, list_ticker))
    
    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date', 'tic'])
    processed_full = processed_full.ffill()
    
    return processed_full

def process_with_sentiment_chunks(df, sentiment_file, output_file):
    """Process and save data with sentiment in chunks"""
    # Convert date to datetime for merging
    df['date'] = pd.to_datetime(df['date'])
    
    # Initialize chunk reader for sentiment data
    sentiment_chunks = pd.read_csv(
        sentiment_file,
        usecols=['Date', 'Stock_symbol', SENTIMENT_COLUMN],
        chunksize=CHUNK_SIZE
    )
    
    # Process first chunk to create output file
    first_chunk = True
    
    for chunk_num, sentiment_chunk in enumerate(sentiment_chunks, 1):
        print(f"Processing sentiment chunk {chunk_num}...")
        
        # Prepare sentiment chunk
        sentiment_chunk['Date'] = pd.to_datetime(sentiment_chunk['Date']).dt.tz_localize(None)
        sentiment_chunk.rename(
            columns={'Stock_symbol': 'tic', SENTIMENT_COLUMN: 'llm_sentiment'},
            inplace=True
        )
        
        # Merge chunk with main dataframe
        merged_chunk = df.merge(
            sentiment_chunk[['Date', 'tic', 'llm_sentiment']],
            left_on=['date', 'tic'],
            right_on=['Date', 'tic'],
            how='left'
        )
        
        # Clean up merged chunk
        merged_chunk = merged_chunk.drop(columns=['Date'])
        
        # Save to CSV
        if first_chunk:
            merged_chunk.to_csv(output_file, index=False)
            first_chunk = False
        else:
            # Append without header
            merged_chunk.to_csv(output_file, mode='a', header=False, index=False)
        
        # Clear memory
        del merged_chunk
        
def main():
    # Nasdaq 100 tickers (July 17, 2023)
    nasdaq_100_tickers = [
        "ADBE", "ADP", "ABNB", "ALGN", "GOOGL", "GOOG", "AMZN", "AMD", "AEP", "AMGN",
        "ADI", "ANSS", "AAPL", "AMAT", "ASML", "AZN", "TEAM", "ADSK", "BKR", "BIIB",
        "BKNG", "AVGO", "CDNS", "CHTR", "CTAS", "CSCO", "CTSH", "CMCSA", "CEG", "CPRT",
        "CSGP", "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DLTR", "EBAY", "EA",
        "ENPH", "EXC", "FAST", "FTNT", "GEHC", "GILD", "GFS", "HON", "IDXX", "ILMN",
        "INTC", "INTU", "ISRG", "JD", "KDP", "KLAC", "KHC", "LRCX", "LCID", "LULU",
        "MAR", "MRVL", "MELI", "META", "MCHP", "MU", "MSFT", "MRNA", "MDLZ", "MNST",
        "NFLX", "NVDA", "NXPI", "ORLY", "ODFL", "ON", "PCAR", "PANW", "PAYX", "PYPL",
        "PDD", "PEP", "QCOM", "REGN", "ROST", "SGEN", "SIRI", "SBUX", "SNPS", "TMUS",
        "TSLA", "TXN", "TTD", "VRSK", "VRTX", "WBA", "WBD", "WDAY", "XEL", "ZM", "ZS"
    ]
    
    # Prepare and process stock data
    print("Preparing stock data...")
    df_raw = prepare_stock_data(nasdaq_100_tickers)
    
    print("Processing data with technical indicators...")
    processed_full = process_data(df_raw)
    
    # Split data into train and trade periods
    print("Splitting data...")
    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    
    # Process and save with sentiment in chunks
    print("Processing training data with sentiment...")
    process_with_sentiment_chunks(
        train,
        'sentiment_deepseek.csv',
        'train_data_deepseek_sentiment_2013_2018.csv'
    )
    
    print("Processing trading data with sentiment...")
    process_with_sentiment_chunks(
        trade,
        'sentiment_deepseek.csv',
        'trade_data_deepseek_sentiment_2019_2023.csv'
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
