import yfinance as yf
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def load_tables(symbol):
    stock = yf.Ticker(symbol) # starting an ticker object
    stock_df = stock.history(interval='1d',period='max',auto_adjust=False)
    return stock_df

def up_down(today, tommorow):
    if today < tommorow:
        return 1
    else:
        return 0

# candle parts percentages
def candle_parts_pcts(o, c, h, l):
    full = h - l
    body = abs(o - c)
    if o > c:
        top_wick = h - o
        bottom_wick = c - l
    else:
        top_wick = h - c
        bottom_wick = o - l
    return top_wick / full, body / full, bottom_wick / full


# previous close and open gap % of pervious candle size
def gap_up_down_pct(o, pc, ph, pl):
    if o == pc:
        return 0
    else:
        return (o - pc) / (ph - pl)
    
    
# z-score calculation
def zscore(x, mu, stdev):
    return (x - mu) / stdev


# target calculation
def target_values(today_close,yesterday_close):
    pct_change = (today_close - yesterday_close) / yesterday_close

    if abs(pct_change) <= 0.007: #half a percent (1%)
        return 0
    elif pct_change > 0.007: # goes up more than .5%
        return 1
    else: #goes down more than -0.5%
        return 2
        



#rolling-makes function go x iterations

def transform_table(table):
    table_close = table[['Open','Close']].copy()
    table_close['std_23days'] = table_close['Close'].rolling(window=23).std().copy() #get standard deviation
    table_close['mu_23days']=table_close['Close'].rolling(window=23).mean().copy() #get mean
    table_close['z_23days'] = (table_close['Close'] - table_close['mu_23days']) / table_close['std_23days'] #get z-score
    
    table_close['Close_shift1'] = table_close['Close'].shift(-1).copy()

    #set target column
    table_close['target'] = table_close.apply(lambda row: target_values(row['Close'], row['Close_shift1']), axis=1).copy()

    table_close['month']=table_close.index.month.copy()
    table_close['dow']=table_close.index.dayofweek.copy()


    #update 1 day table: candle parts %'s
    table_close[['pct_top_wick', 'pct_body', 'pct_bottom_wick']] = table.apply(lambda row: candle_parts_pcts(row['Open'], row['Close'], row['High'],  row['Low']), axis=1, result_type='expand').copy()
    

    #update 1 day table: % gap btwn current open relative to previous candle size
    table_close['pc'] = table['Close'].shift(1).copy()
    table_close['ph'] = table['High'].shift(1).copy()
    table_close['pl'] = table['Low'].shift(1).copy()
    table_close['pct_gap_up_down'] = table_close.apply(lambda row: gap_up_down_pct(row['Open'], row['pc'], row['ph'], row['pl']), axis=1, result_type='expand').copy()


# get z-score of candle top
    table_close['std_top_30days'] = table_close['pct_top_wick'].rolling(window=30).std().copy() 
    
    table_close['mu_top_30days']=table_close['pct_top_wick'].rolling(window=30).mean().copy() 
    table_close['z_top_30days'] = (table_close['pct_top_wick'] - table_close['mu_top_30days'])/ table_close['std_top_30days']




# get z-score of candle body
    table_close['std_body_30days'] = table_close['pct_body'].rolling(window=30).std().copy() 
    
    table_close['mu_body_30days']=table_close['pct_body'].rolling(window=30).mean().copy() 
    table_close['z_body_30days'] = (table_close['pct_body'] - table_close['mu_body_30days']) / table_close['std_body_30days']



# get z-score of candle bottom
    table_close['std_bottom_30days'] = table_close['pct_bottom_wick'].rolling(window=30).std().copy() 
    table_close['mu_bottom_30days']=table_close['pct_bottom_wick'].rolling(window=30).mean().copy() 
    table_close['z_bottom_30days'] = (table_close['pct_bottom_wick'] - table_close['mu_bottom_30days']) / table_close['std_bottom_30days']



# get z-score of percent gap up/down
    table_close['std_gap_30days'] = table_close['pct_gap_up_down'].rolling(window=30).std().copy()  
    table_close['mu_gap_30days']=table_close['pct_gap_up_down'].rolling(window=30).mean().copy() 
    table_close['z_gap_30days'] = (table_close['pct_gap_up_down'] - table_close['mu_gap_30days']) / table_close['std_gap_30days']

    
    table_close = table_close[['Close', 'std_23days', 'z_23days', 'target', 'mu_23days', 'month', 'dow', 'pct_top_wick', 'pct_body', 'pct_bottom_wick', 'pct_gap_up_down','z_top_30days','z_body_30days','z_bottom_30days','z_gap_30days']].copy()


    last_day = table_close.iloc[-1].copy() #iloc-index location for rows; -1 is last row
    table_close.dropna(axis=0,inplace=True) #gets rid of all null values on table
    return table, last_day, table_close


def train_test_model(df):
    # Features and target
    X = df[['Close','std_23days','z_23days','mu_23days','month','dow', 'pct_top_wick', 'pct_body', 'pct_bottom_wick', 'pct_gap_up_down','z_top_30days','z_body_30days','z_bottom_30days','z_gap_30days']]
    y = df['target']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the XGBoost classifier
    model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss',    use_label_encoder=False)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # feature importances show which column are more important to model
    importances = model.feature_importances_

    return print(f'Accuracy: {accuracy:.2f}\nClassification Report: {report}'),model, importances




