import yfinance as yf
import pandas as pd

def load_tables(symbol):
    stock = yf.Ticker(symbol)
    stock_df=stock.history(interval='1d',period = 'max',auto_adjust = False)
    return stock_df

def transform_table(table):
    table_close - table[['Close']].copy()
    cmg_df_close['std_5days'] = cmg_df_close['Close'].rolling(window=5).std().copy()
    return table_close