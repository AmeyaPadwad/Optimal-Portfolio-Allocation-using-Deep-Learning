#------------------ Libraries ----------------------#
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as spo
import tensorflow as tf
mlp.style.use('seaborn-darkgrid')

#--------------- Financial Data Class ---------------#

class FinancialData(object):
    def __init__(self, tickers=['SPY'], fillna=True, cols=None,  **kwargs):

        # Prepare the data- Extract data from Yahoo Finance:
        if isinstance(tickers,list):
            t = ' '.join(tickers)
            df = yf.download(t, **kwargs)
        
        elif isinstance(tickers,str):
            df = yf.download(tickers,**kwargs)

        if not isinstance(cols,type(None)):
            df = df[cols]

        if fillna:
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)

        # Rename columns so that they are one level:
        df = one_lvl_colnames(df,cols,tickers)

        self.tickers = tickers
        self.df = df
        self.columns = cols
        
    def plot_data(self,tickers=None, cols='Adj Close', title='Historical Adj Close Price Data', ylabel='Adj Close Prices', xlabel='Date', fontsize=10, **kwargs):

        # Retrieve important information:
        df = self.get_data()
        if isinstance(tickers,type(None)):
            tickers = self.get_tickers()

        # Define data column, if multiple data is in the dataset:
        cols = return_names(cols,tickers)
        df = df[cols]

        # Define the axis, and plot the data:
        ax = df.plot(fontsize=fontsize,**kwargs)
        ax.set_title(title,fontsize=fontsize*1.3)
        ax.set_xlabel(xlabel,fontsize=fontsize*1.1)
        ax.set_ylabel(ylabel,fontsize=fontsize*1.1)

        return ax
    
    def rolling_statistics(self, cols='Adj Close', tickers=None, functions=None, window=20, bollinger=False,  roll_linewidth=1.5, **kwargs):

        # Define important varibles:
        df = self.df
        if isinstance(tickers,type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        if isinstance(functions,type(None)):
            functions = [momentum, simple_moving_average, bollinger_bands]
        elif not isinstance(functions,list):
            functions = [functions]

        # Define the actual dataframe analized
        df = df[col_names]

        # Compute the rolling statistics:
        rolling_stats = df.rolling(window).agg(functions)

        # Given one level names:
        rolling_stats = one_lvl_colnames(rolling_stats,col_names,functions)

        return rolling_stats
    
    def get_returns(self, cols='Adj Close', tickers=None, return_window=1, plot=False, **kwargs):

        # Define important variables:
        df = self.df
        if isinstance(tickers, type(None)):
            tickers = self.get_tickers()
        col_names = return_names(cols,tickers)
        
        # Compute the returns:
        returns = df[col_names].pct_change(return_window)

        # Plot returns:
        if plot:
            returns.plot(**kwargs)
        
        # Define attributes:
        self.returns = returns.dropna(how='all')
        self.return_window = return_window

        return self.returns
    
    def find_beta_alpha(self, market=None, plot=False, nrows=1, ncols=1, figsize=(10,5), fillna=True, title = None, **kwargs):

        # Define important variables:
        market = market.to_frame()
        market_name = market.columns[0]
        try:
            returns = self.returns
        except:
            returns = self.get_returns()
        
        # Merge data:
        df = market.merge(returns, left_index=True, right_index=True, how='left')

        # Fill NaNs
        if fillna:
            df.fillna(method='ffill',inplace=True)
            df.fillna(method='bfill',inplace=True)
        
        alpha_beta = {}
        stocks = [stock for stock in df.columns.values if stock != market_name]
        
        # Find the alpha, beta values for each stock in the object:
        for stock in stocks:
            beta, alpha = np.polyfit(df[market_name],df[stock],1)
            alpha_beta[stock] = (alpha,beta)
        if plot:
            fig = plt.figure(figsize=figsize)
            axs = {'ax'+str(i+1): fig.add_subplot(nrows,ncols,i+1) for i in range(len(stocks))}
            for i,stock in enumerate(stocks):
                alpha, beta = alpha_beta[stock]
                df.plot(kind='scatter', ax=axs['ax'+str(i+1)], x=market_name, y=stock, **kwargs)
                axs['ax'+str(i+1)].plot(df[market_name],df[market_name]*beta+alpha)
                axs['ax'+str(i+1)].text(df[market_name].min(), df[stock].max(), r'$\beta$ = {}  $\alpha$ = {}'.\
                    format(round(beta,2),round(alpha,2)), fontsize=8)
            fig.suptitle(title,fontsize=15)

        return alpha_beta
    
    def get_normalized_prices(self, start_date=None, plot=False, prices_col='Adj Close', title=None,x_label='Date',y_label='Normalized Prices', fontsize=15,**kwargs):

        # Define important variables:
        if not isinstance(prices_col,list):
            prices_col = [prices_col]
        prices_names = return_names(prices_col,self.get_tickers())
        prices = self.get_data()[prices_names]

        if isinstance(start_date,type(None)):
            start_date = prices.index.min()

        # Compute the normalized prices:
        base = prices.loc[prices.index==start_date].values
        norm_prices = prices/base*100

        # Plot normalized prices
        if plot:
            norm_prices.plot(fontsize=fontsize,**kwargs)
            
            plt.hlines(y = 100, xmin = norm_prices.index.min(), xmax = norm_prices.index.max(), color = 'black', linestyles = 'dashdot')

            if isinstance(title,type(None)):
                title = 'Precios Normalizados (100 = {}-{}-{})'.\
                        format(start_date.year,start_date.month, start_date.day)
            
            plt.title(title,fontsize=fontsize*1.3)
            plt.ylabel(y_label,fontsize=fontsize*1.1)
            plt.xlabel(x_label,fontsize=fontsize*1.1)

        return norm_prices

    def get_tickers(self):
        return self.tickers

    def get_data(self):
        return self.df


#---------------------- Portfolio Class --------------------------#
class Portfolio(FinancialData):

    def __init__(self,tickers=['SPY'], fillna=True, cols=None, weights=[1], column = 'Close', **kwargs):
        FinancialData.__init__(self, tickers, fillna, cols, **kwargs)
        columns = [column+'_'+ticker for ticker in tickers]
        prices = self.prepare_data(fillna=fillna)
        self.prices = prices.loc[:,columns]
        self.weights = weights
    
    def normalize_prices(self, start_date=None, end_date=None, tickers=None, column='Close'):

        prices = self.prices
        if tickers == None:
            tickers = self.get_tickers()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        columns = [column+'_'+ticker for ticker in tickers]
        norm_prices = prices.loc[start_date:end_date,columns]/prices.loc[start_date,columns]
        return norm_prices
    
    def get_portfolio_values(self,start_date=None,end_date=None,tickers=None,column='Close'):

        prices = self.get_prices()
        weights = self.get_weights()
        if start_date == None:
            start_date = prices.index.values.min()
        if end_date == None:
            end_date = prices.index.values.max()
        if tickers == None:
            tickers = self.get_tickers()
        norm_prices = self.normalize_prices(start_date,end_date,tickers,column)
        portfolio_values = norm_prices*weights
        portfolio_values['Portfolio'] = portfolio_values.sum(axis=1)
        return portfolio_values

    def get_prices(self):
        return self.prices
    
    def get_weights(self):
        return self.weights
    
    def change_weights(self,weights):

        assert len(self.weights) == len(weights), "Wrong length of weights"
        self.weights = weights
    
    def get_returns(self, start_date=None, end_date=None, tickers=None, column='Close', window=1, portfolio_returns=False):

        # Get the returns of each asset inside the portfolio:
        prices = self.get_prices()
        if start_date is None:
            start_date = prices.index.values.min()
        if end_date is None:
            end_date = prices.index.values.max()
        if tickers is None:
            tickers = self.get_tickers()
        columns = [column+'_'+ticker for ticker in tickers]
        prices = prices.loc[start_date:end_date,columns]
        returns = prices.pct_change(window).dropna(how='all')

        # Add the portfolio returns
        if portfolio_returns:
            weights = self.get_weights()
            returns['Portfolio'] = (returns*weights).sum(axis=1)

        return returns
    
    def get_performance_metrics(self, risk_free_rate=0, start_date=None, end_date=None, **kwargs):

        if start_date == None:
            start_date = self.get_prices().index.values.min()
        if end_date == None:
            end_date = self.get_prices().index.values.max()
        portfolio_values = self.get_portfolio_values(start_date, end_date, **kwargs)
        def compute_cum_return(series):
            mn = series.index.values.min()
            mx = series.index.values.max()
            cum_return = (series[mx]/series[mn])-1
            return cum_return  

        cum_return = compute_cum_return(portfolio_values['Portfolio'])
        returns = self.get_returns(start_date, end_date, portfolio_returns=True, **kwargs)
        sharpe_ratio = self.get_sharpe_ratio(risk_free_rate, start_date=start_date, end_date=end_date, **kwargs)
        metrics = returns['Portfolio'].agg(['mean','std'])
        metrics.loc['Cum Return'] = cum_return
        metrics.loc['Sharpe Ratio'] = sharpe_ratio
        return metrics

    def get_sharpe_ratio(self, weights=None, rfr=0, negative=False, **kwargs):

        # Get building blocks for the computation:
        returns = self.get_returns(**kwargs)
        if weights is None:
            weights = self.get_weights()

        # Get the portfolio returns:
        portfolio_returns = (returns*weights).sum(axis=1)
        portfolio_std = portfolio_returns.std()

        # Compute Sharpe Ratio formula:
        sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std
        if negative:
            sharpe_ratio *= -1
        return sharpe_ratio

    def optimize_portfolio(self, guess_weights=None, short=False, rfr=0, **kwargs):
        tickers = self.get_tickers()

        if guess_weights is None:
            guess_weights = [1/len(tickers) for i in range(len(tickers))]
        
        # Determine the bounds of the optimized weights (min=0, max=1):
        if not short:
            bounds = [(0,1) for i in range(len(tickers))]
        else:
            bounds = [(-1,1) for i in range(len(tickers))]

        # Determine the restrictions:
        weights_sum_to_1 = {'type':'eq', 'fun':lambda weights: np.sum(np.absolute( weights))-1}
        
        # Optimize:
        opt_weights = spo.minimize(self.get_sharpe_ratio, guess_weights, args=(rfr,True), method='SLSQP', options={'disp':False}, constraints=(weights_sum_to_1), bounds=bounds )

        # Update weights to optimized weights
        print(len(opt_weights.x))
        self.change_weights(opt_weights.x)

        return opt_weights.x

#----------------------------- WindowGenerator Class -------------------------------------#

class WindowGenerator():

    def __init__(self, input_width=5, label_width=1, shift=1, train_df=None, val_df=None, test_df=None,  label_columns=None, batch_size=None, shuffle=False):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_columns = label_columns
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Define information about columns:
        if isinstance(label_columns,type(None)):
            self.label_columns_indices = {name:i for i,name in enumerate(label_columns)}
        self.column_indices = {name:i for i,name in enumerate(train_df.columns)}

        # Define window information:
        self.total_window_size = input_width+shift
        self.input_slice = slice(0,input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size-self.label_width
        self.labels_slice = slice(self.label_start,None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    def __repr__(self):
        return '\n'.join([f'Total window size: {self.total_window_size}', f'Input indices: {self.input_indices}', f'Label indices: {self.label_indices}', f'Label column name(s): {self.label_columns}'])
    
    def split_window(self, features):

        inputs = features[:, self.input_slice,:]
        labels = features[:,self.labels_slice,:]
        if not isinstance(self.label_columns,type(None)):
            labels = tf.stack([labels[:,:,self.column_indices[name]] for name in self.label_columns], axis = -1)
        
        # Set the shapes of the informaiton:
        inputs.set_shape([None,self.input_width,None])
        labels.set_shape([None,self.label_width,None])

        return inputs,labels
    
    def make_dataset(self,data):

        data = np.array(data,dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(data = data, targets = None, sequence_length =  self.total_window_size, sequence_stride = 1, shuffle = self.shuffle, batch_size = self.batch_size)
        ds = ds.map(self.split_window)

        return ds

    # Adding properties for accessing the train, val and test as tf.data.Dataset objects
    @property
    def train(self):
        if isinstance(self.train_df,type(None)):
            return None
        else:
            return self.make_dataset(self.train_df)

    @property
    def val(self):
        if isinstance(self.val_df,type(None)):
            return None
        else:
            return self.make_dataset(self.val_df)

    @property
    def test(self):
        if isinstance(self.test_df,type(None)):
            return None
        else:
            return self.make_dataset(self.test_df)
    
#---------------------------------------------------------------------------------------#
# Complementary functions:

def one_lvl_colnames(df,cols,tickers):

    # Define important variables:
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(cols, str):
        cols = [cols]

    # For multi-level column indexing:
    if isinstance(df.columns.values[0], tuple):

        # Define important varibles
        columns = df.columns.values
        new_cols = []

        # Iterate through the multi-level column names and flatten them:
        for col in columns:
            temp = []
            for name in col:
                if name != '':
                    temp.append(name)
            new_temp = '_'.join(temp)
            new_cols.append(new_temp)
        
        df.columns = new_cols
    
    # For uni-level colum indexing:
    elif isinstance(df.columns.values[0], str):
        
        # Define new names:
        col_names = [column+'_'+ticker for column in cols for ticker in tickers]
        df.columns = col_names

    return df

def return_names(cols,tickers):

    # Give the correct type:
    if isinstance(cols, str):
        cols = [cols]
    if isinstance(tickers, str):
        tickers = [tickers]

    col_names = [col+'_'+ticker for col in cols for ticker in tickers]

    return col_names

def momentum(prices):

    first = prices.iloc[0]
    last = prices.iloc[-1]
    momentum_df = last/first

    return momentum_df

def simple_moving_average(prices):

    mean = prices.mean()
    sma = prices[-1]/mean-1

    return sma
    
def bollinger_bands(prices):

    ma = prices.mean()
    std = prices.std()
    bb = (prices[-1]-ma)/(2*std)

    return bb

def plot_window(window_dataset, pandas_dataset, window_size, model, figsize=(100,50)):

    # Determine the X-axis of the plot:
    plot_index = pandas_dataset.iloc[window_size:,:].index
    # Assign in the adequate format the values of the observed taget variable(s):
    y = np.concatenate([targets for inputs,targets in window_dataset],axis=0)
    # Use the model to predict the target variable(s):
    y_hat = model.predict(window_dataset)
    # Adjust the shapes:
    y = y.reshape(y_hat.shape)

    # Plot the data:
    fig = plt.figure(figsize=figsize, dpi=50)
    for n in range(y_hat.shape[1]):
        plt.subplot(y_hat.shape[1],2,n+1)
        plt.ylabel('Return')
        plt.plot(plot_index,y_hat[:,n],label='Predicted',color='maroon')
        plt.plot(plot_index,y[:,n],label='Observed',color='midnightblue',alpha=0.5)
    plt.legend()

def daily_rate(x, periods_year=252):
    dr = np.power(1+x,1/periods_year)-1
    return dr

def optimize_portfolio(returns, guess_weights=None, short=True, rfr=0):

    # Define important variables:
    num_assets = returns.shape[1]
    if isinstance(guess_weights,type(None)):
        guess_weights = [1/num_assets for i in range(num_assets)]

    # Define bound if short possitions are allowed or not:
    if not short:
        bounds = [(0,1) for i in range(num_assets)]
    else:
        bounds = [(-1,1) for i in range(num_assets)]

    # Define constraints, if there can or not be leverage
    weights_sum_to_1 = {'type':'eq', 'fun':lambda weights: np.sum(np.absolute(weights))-1}
    
    # Minimize the function:
    opt_weights = spo.minimize(sharpe_ratio, guess_weights, args = (rfr, True, returns), method = 'SLSQP', options = {'disp':False}, constraints = (weights_sum_to_1), bounds = bounds)

    return opt_weights 

def sharpe_ratio(weights=None, rfr=0, negative=False, returns=0):

    # Define important variables:
    num_assets = returns.shape[1]
    if isinstance(weights,type(None)):
        weights = [1/num_assets for i in range(num_assets)]

    # Get portfolio returns:
    portfolio_returns = (returns*weights).sum(axis=1)
    portfolio_std = portfolio_returns.std()

    # Compute Sharpe Ratio formula:
    sharpe_ratio = (portfolio_returns-rfr).mean()/portfolio_std

    # If used in a minization process:
    if negative:
        sharpe_ratio *= -1

    return sharpe_ratio    