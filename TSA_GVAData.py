import eurostat

import matplotlib.pyplot as plt

from data_processing import process_import_data, process_ICT_labour_import_data
from utils import debug_print, info_print, error_print

import pandas as pd

from statsmodels.tsa.ar_model import AutoReg as AR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the raw data from Eurostat API
# GVA_data_import = eurostat.get_data_df('namq_10_a10')
# info_print("Got GVA data")                
# # Define the starting quarter for filtering data
# date_start = '2019Q4'

# # Process the raw data
# GVA_data = process_import_data(GVA_data_import, date_start)

# data_GVA=pd.DataFrame(GVA_data)
# data_GVA.to_parquet('TSA_Parquet_data/GVA_data.parquet', index=False)
# info_print("Saved GVA Parquet data")

GVA_data=pd.read_parquet('TSA_Parquet_data/GVA_data.parquet')
info_print("Completed GVA Parquet data read")

# Filter GVA data for the given country, where the sector is 'J', unit is 'PC_GDP', item is 'B1G', and data is not seasonally adjusted
filtered_data_GVA = GVA_data[(GVA_data['nace_r2'] == 'J') & 
                                 (GVA_data['unit'] == 'PC_GDP') & 
                                 (GVA_data['geo'] == 'IT') & 
                                 (GVA_data['na_item'] == 'B1G') & 
                                 (GVA_data['s_adj'] == 'NSA')].copy()

# Select only 'quarter' and 'value' columns, rename 'value' to 'GVA_value'
filtered_data_GVA = filtered_data_GVA[['quarter', 'value']]


#set index as the quarter values for filtered data gva
filtered_data_GVA=filtered_data_GVA.set_index('quarter')

# Convert PeriodIndex to DatetimeIndex for decomposition and plotting
if isinstance(filtered_data_GVA.index, pd.PeriodIndex):
    filtered_data_GVA = filtered_data_GVA.copy()
    filtered_data_GVA.index = filtered_data_GVA.index.to_timestamp()


#plot a graph of GVA values over time with quarterly values in format 'YYYY-Qn'
plt.plot(filtered_data_GVA.index, filtered_data_GVA['value'], marker='o')
plt.xlabel('Quarter')       
plt.ylabel('GVA Value')
plt.title('GVA Value over Time')
plt.show


#Decompose the GVA time series to detect trend,seasonality and residuals
decomposition_GVA = seasonal_decompose(filtered_data_GVA['value'], model='multiplicative', period=4)
decomposition_GVA.plot()
plt.show()


#Handling missing values for GVA
filtered_data_GVA['value']=filtered_data_GVA['value'].interpolate(method='spline',order=2)
    
filtered_data_GVA = filtered_data_GVA.rename(columns={'value': f'IT_GVA_value'})



#Checking for stationarity using ADF test
result = adfuller(filtered_data_GVA['IT_GVA_value'])
info_print('ADF Statistic: %f' % result[0])
info_print('p-value: %f' % result[1])

if result[1] <= 0.05:
    info_print("The time series is stationary (reject H0)")
else:
    info_print("The time series is non-stationary (fail to reject H0)")

for key, value in result[4].items():    
    info_print('Critical Values:')
    info_print(f'{key}, {value}')



#Plotting ACF and PACF-Check for seasonality
fig, ax = plt.subplots(2,1, figsize=(10,8))

#ACF shows slow decay
plot_acf(filtered_data_GVA['IT_GVA_value'], ax=ax[0])

#PACF cuts off after lag 1
plot_pacf(filtered_data_GVA['IT_GVA_value'], ax=ax[1])

#Posisible AR model for stationary data
plt.show()


#Use AR model to forecast and check MSE.
model = AR(filtered_data_GVA['IT_GVA_value'],1)
model_fit = model.fit()
predictions = model_fit.predict(start=len(filtered_data_GVA), end=len(filtered_data_GVA)+4, dynamic=False)
forecast = predictions
info_print(predictions)

plt.plot(filtered_data_GVA['IT_GVA_value'], label='Observed')
plt.plot(forecast, label='Forecast', color='red')   
plt.legend()
plt.show()

#Use SARIMAX Model to forecast and check if MSE is less compared to AR(1) model.
from statsmodels.tsa.statespace.sarimax import SARIMAX
model_sarimax = SARIMAX(filtered_data_GVA['IT_GVA_value'], order=(1,0,0), seasonal_order=(1,0,0,4))
model_sarimax_fit = model_sarimax.fit(disp=False)
predictions_sarimax = model_sarimax_fit.predict(start=len(filtered_data_GVA), end=len(filtered_data_GVA)+4, dynamic=False)
forecast_sarimax = predictions_sarimax
info_print(predictions_sarimax)

plt.plot(filtered_data_GVA['IT_GVA_value'], label='Observed')
plt.plot(forecast_sarimax, label='Forecast', color='red')   
plt.legend()
plt.show()

# Evaluation of AR model
mae = mean_absolute_error(filtered_data_GVA['IT_GVA_value'][-4:], forecast[-4:])
mse = mean_squared_error(filtered_data_GVA['IT_GVA_value'][-4:], forecast[-4:])
info_print(f"AR Model - MAE: {mae}, MSE: {mse}")

#Evaluation of SARIMAX model
mae = mean_absolute_error(filtered_data_GVA['IT_GVA_value'][-4:], forecast_sarimax[-4:])
mse = mean_squared_error(filtered_data_GVA['IT_GVA_value'][-4:], forecast_sarimax[-4:])
info_print(f"SARIMAX Model - MAE: {mae}, MSE: {mse}")

