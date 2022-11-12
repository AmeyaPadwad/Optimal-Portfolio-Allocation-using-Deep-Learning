#------------------ Libraries ----------------------#
import os
import quandl
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as spo
from scipy.stats import kurtosis, skew
from financial_data import *
import tensorflow as tf
mlp.style.use('seaborn')
quandl.save_key('HtwBLPt3k37yZHTvy15K')

# Importing S&P500 indexes:
sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500 = pd.read_html(sp_url, header=0)[0]


#------------------ Data Preprocessing ----------------------#


# Correct invalid dates:
sp500.loc[sp500[sp500['Date first added']=='1983-11-30 (1957-03-04)'].index,'Date first added'] = '1983-11-30'

# Filter firms that entered the index after December 2015:
sp500['Date first added'] = pd.to_datetime(sp500['Date first added'],format='%Y-%m-%d')
sp500 = sp500[sp500['Date first added']<'2007-01-01']

# Define a random seed:
n_stocks = 10
np.random.seed(1792)
universe_tickers = sp500['Symbol'].unique()
portfolio_tickers = list(np.random.choice(universe_tickers,replace=False,size=n_stocks))

# Historic price data of 10 randomly selected stocks:
start_date = '2007-01-04'
end_date = '2021-01-05'
my_portfolio = FinancialData(tickers = portfolio_tickers, cols = ['Adj Close','Volume'], start = start_date, end = end_date)
my_portfolio.plot_data(figsize=(10,5))
norm_prices =my_portfolio.get_normalized_prices(plot=True,figsize=(10,5),fontsize=10, title='Historical Adj Normalized Prices Data') #normalized prices

# Portfolio returns:
port_returns = my_portfolio.get_returns(plot=True,subplots=True,figsize=(10,5),kind='hist',bins=100, layout=(5,2), title='Portfolio Return Histograms')
port_returns = my_portfolio.get_returns(plot=True,subplots=True,figsize=(10,20), layout=(5,2), title='Portfolio Returns')

# Scatter plot of returns of asset vs returns of market, along with alpha and beta parameters:
market = yf.download('SPY',start=start_date,end=end_date)['Adj Close'].rename('Adj Close_SPY').pct_change()
portfolio_alphas_betas = my_portfolio.find_beta_alpha(market=market,plot=True,nrows=2,ncols=5,figsize=(10,5),color='teal', title = 'Returns of Asset vs. Returns of Market')


# Additional metrics about the returns:
def abs_mean(x):
    ab = x.abs()
    ab_m = ab.mean()
    return ab_m
print(port_returns.agg(['min','max','mean','std','kurtosis','skew',abs_mean]).transpose())


#------------------ Building the Model ----------------------#

# DEFINING DATASETS

# Extracting additional factors:
new_tickers = ['EIA/PET_RWTC_D','FRED/T10Y2Y','FRED/T10Y3M','FRED/DTB3','FRED/DLTIIT', 'FRED/TEDRATE']
names = ['wti_spot','10y2y_spread','10y3m_spread','3m_rate','ltiit','ted_spread']
add_factors = quandl.get(new_tickers, start_date=start_date, end_date=end_date, api_key=quandl.ApiConfig.api_key)
add_factors.columns = names
add_factors['var_wti'] = add_factors['wti_spot'].pct_change()

# Fill NaN values:
add_factors.fillna(method='ffill',inplace=True)
add_factors.fillna(method='bfill',inplace=True)

# Factors: momentum, simple moving average and bollinger bands:
factors = my_portfolio.rolling_statistics()

# Merge dataframes to get the full data for modeling:
total_df = port_returns.merge(factors,right_index=True,left_index=True,how='left')
add_factors.index = pd.to_datetime(add_factors.index)
add_factors.index = add_factors.index.tz_localize(None)
total_df.index = pd.to_datetime(total_df.index)
total_df.index = total_df.index.tz_localize(None)
total_df = total_df.merge(add_factors,right_index=True,left_index=True,how='left')

# Drop NaN that result from rolling functions:
total_df.dropna(subset=factors.columns,inplace=True)

# Define the label columns:
label_cols = total_df.columns[:n_stocks]

# Define train (70%), val (20%) and test (10%) dataframes: 
train_p, val_p, test_p = 0.7,0.2,0.1
window_size = 5
num_features = total_df.shape[1]
total_size = len(total_df)
train_size = int(total_size*train_p)
val_size = int(total_size*val_p)
test_size = int(total_size*test_p)
train_df = total_df.iloc[:train_size,:]
val_df = total_df.iloc[train_size-window_size:train_size+val_size,:]
test_df = total_df.iloc[train_size+val_size-window_size:,:]

# Define the batch size:
batch_size = 512

# Create an instance of the WindowGenerator object:
my_window = WindowGenerator(input_width=window_size, label_width=1, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=label_cols, batch_size=batch_size, shuffle=True)

# Print the shapes for one batch of each sub dataset:
for example_inputs, example_labels in my_window.train.take(1):
    print("Train input shape:",example_inputs.shape)
    print("Train target shape:",example_labels.shape)
for example_inputs, example_labels in my_window.val.take(1):
    print("Validation input shape:",example_inputs.shape)
    print("Validation target shape:",example_labels.shape)
for example_inputs, example_labels in my_window.test.take(1):
    print("Test input shape:",example_inputs.shape)
    print("Test target shape:",example_labels.shape)

# CREATING THE RNN ARCHITECTURE

from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

# Model architecture creation:
model_1 = Sequential([
    BatchNormalization(
        input_shape = (window_size,num_features),
        name = 'Batch_Norm_1'),
    LSTM(512, return_sequences=True, name='LSTM_1'), BatchNormalization(),
    LSTM(512, name='LSTM_2'), BatchNormalization(momentum=0.8),
    Dense(256, activation='relu', name='Dense_1'),
    Dense(n_stocks, name='Returns')
])

# Learning Rate Schedule: used to decide the learning rate:
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8*10**(epoch/20))

#Checkpoint callback to save the model:
checkpont_rnn = ModelCheckpoint(filepath='model_1_rnn', save_weights_only=False, save_freq = 'epoch', monitor = 'val_loss', save_best_only = True, verbose = 0)

# Define optimizer
optimizer = tf.keras.optimizers.Adam(lr=1e-3)

# Compile Model
model_1.compile(loss=tf.keras.losses.Huber(), metrics=[tf.metrics.RootMeanSquaredError(),'mae'],
optimizer=optimizer)

# Train model
history = model_1.fit(my_window.train, validation_data=my_window.val, epochs=100)
print(model_1.summary())

model_1 = load_model('rnn_model_1')
print(model_1.summary())

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])
plt.title('Huber Loss: Training and Validation')
plt.subplot(1,2,2)
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training','Validation'])
plt.title('RMSE Loss: Training and Validation')

print(model_1.evaluate(my_window.test))


#------------------ Training/Validation/Testing ----------------------#


my_window_2 = WindowGenerator(input_width=window_size, label_width=1, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=label_cols, batch_size=batch_size, shuffle=False)

plot_window(my_window_2.train,train_df,window_size,model_1) # Training Results
plot_window(my_window_2.val,val_df,window_size,model_1) # Validation Results
plot_window(my_window_2.test,test_df,window_size,model_1) # Test Results
plt.show()