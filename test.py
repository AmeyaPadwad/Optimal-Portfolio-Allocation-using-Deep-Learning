from tensorflow.keras.layers import Dense, Conv1D, LSTM, GRU, Input, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model

#------------------ Portfolio Allocation ----------------------#


# columns = ['FR_'+ticker for ticker in portfolio_tickers]
# y_train = model_1.predict(my_window_2.train)
# y_val = model_1.predict(my_window_2.val)
# y_test = model_1.predict(my_window_2.test)
# y_hat_total = np.concatenate([y_train,y_val,y_test],axis=0)
# ret_hat_df = pd.DataFrame(data=y_hat_total,index=total_df.index[5:],columns=columns)
# rfr = add_factors['3m_rate'].agg(daily_rate)
# ret_hat_df = ret_hat_df.merge(rfr.rename('rfr'),left_index=True,right_index=True,how='left')
# # ret_hat_df.rolling(40).agg(lambda x: optimize_portfolio(
# #     returns = ret_hat_df[ret_hat_df.columns[:-1]],
# #     rfr = ret_hat_df[ret_hat_df.columns[-1]]))
# opt_weights = np.array([optimize_portfolio(
#     returns = window[ret_hat_df.columns[:-1]],
#     rfr = window[ret_hat_df.columns[-1]]).x for window in ret_hat_df.rolling(40)])
# print(opt_weights.shape)
# print(opt_weights)
# plt.show()