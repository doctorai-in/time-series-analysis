
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[12]:


df_train = pd.read_csv('train.csv', index_col='Date', parse_dates=True).sort_values('Date')
df_test = pd.read_csv('test.csv', index_col='Date', parse_dates=True).sort_values('Date')
df_store = pd.read_csv('store.csv')


# In[13]:


print("Total train data points : ",len(df_train))
print("Total test data points : ",len(df_test))
print("Total store : ",len(df_store))
df_train.head(2)


# In[4]:


df_test.head(2)


# In[5]:


df_store.head(2)


# In[6]:


freq_data_per_store=[]
for i in range(1,1116):
    sales_data_wz = pd.DataFrame(df_train[ (df_train['Sales'] !=0) & (df_train['Store']==i) ].loc[:,'Sales'])
    sales_data_z = pd.DataFrame(df_train[ df_train['Store']==i ].loc[:,'Sales'])
    freq_data_per_store.append((i, len(sales_data_wz), len(sales_data_z)))


# In[7]:


# Construct DataFrame from freq_data_per_store 
order_df = pd.DataFrame(freq_data_per_store, 
                        columns=['store', 'data point_without_zero', 'data point_zero'])

order_df.sum()


# In[8]:


sales_data = pd.DataFrame(df_train[ (df_train['Sales'] !=0) & (df_train['Store']==i) ].loc[:,'Sales'])


# In[9]:


sales_data.head(5)


# In[10]:


train=sales_data
#test = sales_data.loc['2015']
print("Data points used for training : ",len(train))
#print("Data points used for test : ",len(test))
train.plot()
plt.show()


# ## DICKY-Fuller Test

# <img src="adfuller.png" />

# In[62]:


# Import augmented dicky-fuller test function
from statsmodels.tsa.stattools import adfuller


# In[63]:


# Run test
result = adfuller(train['Sales'])


# In[64]:


print(result)
# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Print critical values
print(result[4]) 


# ## If data is non-satationary

# which ARIMA model is the best fit for a dataset after trying different degrees of differencing and applying the Augmented Dicky-Fuller test on the differenced data.
# 1. ADF value
# 2. P-value

# In[ ]:


'''# Take the first difference of the data
train_diff = train.diff().dropna()

# Create ARMA(2,2) model
arma = SARIMAX(train_diff, order=(2,0,2))

# Fit model
arma_results = arma.fit()

# Print fit summary
print(arma_results.summary())'''


# In[65]:


# Plot the time series
fig, ax = plt.subplots()
train.plot(ax=ax)
plt.show()


# ## ACF VS PACF 

# 1. ACF - Autocorrelation Function
# 2. PACF - Partial autocorrelation function
# <img src="acf vs pacf.png" />

# In[66]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(train, lags=20, zero=False)
plot_pacf(train, lags=20, zero=False)
plt.show()


# # ARMA Model

# In[67]:


from statsmodels.tsa.arima_model import ARMA

model = ARMA(train, order=(1,1))
result_m = model.fit()


# In[68]:


print(result_m.summary())


# In[107]:


#Forecasting
result_m.plot_predict(start='2013-01-02', end='2014-12-31')
plt.show()


# # ARIMA Model

# In[135]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[149]:


# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
      	# create and fit ARMA(p,q) model
        model = SARIMAX(train, order=(p,0,q), trend='c')
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p,q,results.aic, results.bic))


# In[150]:


# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'q', 'AIC', 'BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))

# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))


# In[184]:


# Now after searching p and q apply to the model
model_s = SARIMAX(train, order=(2,0,2), trend='c')
result_s = model_s.fit()


# In[185]:


# Generate predictions
one_step_forecast = result_s.get_prediction(start=-100)

# Make ARIMA forecast of next 10 values
forecast = result_s.get_forecast(steps=10)


# In[186]:


# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean
mean_forecast_


# In[187]:


# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()


# In[188]:


# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower Sales']
upper_limits = confidence_intervals.loc[:,'upper Sales']

# Print best estimate  predictions
print(mean_forecast)


# In[189]:


plt.rcParams.update({'font.size': 15})
plt.figure(figsize=(20,5))
# plot the amazon data
plt.plot(train.index, train, label='observed')
# plot your mean predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits,upper_limits, color='grey')
plt.xlabel('Date', labelpad=10, fontsize=20)
plt.ylabel('No. of Sales', labelpad=10, fontsize=20 )
plt.legend()
plt.show()


# ## AIC - Akaike information criterion
# 

# 1. Lower AIC indicates a better model
# 2. AIC likes to choose simple models with lower order

# In[190]:


print(result_s.aic)


# ## BIC - Bayesian information criterion
# 

# 1. Very similar to AIC
# 2. Lower BIC indicates a better model
# 3. BIC likes to choose simple models with lower order

# In[191]:


print(result_s.bic)


# ## Model diagnostics

# In[192]:


# Assign residuals to variable
residuals = result_s.resid


# In[193]:


mae = np.mean(np.abs(residuals))
print(mae)


# ### Plot diagnostics
# 

# If the model ts well the residuals will be white
# Gaussian noise
# <img src="plot_diagnostics.png" />

# In[194]:


# Create the 4 diagostics plots
plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(10,2))

result_s.plot_diagnostics(figsize=(15, 8), lags=15)
plt.show()


# In[195]:


print(result_s.summary())


# <img src="prob(Q, JB).png" />

# ## Unrolling ARMA forecast

# In[ ]:


'''# Make arma forecast of next 10 differences
arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean

# Integrate the difference forecast
arma_int_forecast = np.cumsum(arma_diff_forecast)

# Make absolute value forecast
arma_value_forecast = arma_int_forecast + amazon.iloc[-1,0]

# Print forecast
print(arma_value_forecast)'''

