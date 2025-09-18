
import eurostat

import matplotlib.pyplot as plt

from data_processing import process_import_data, process_ICT_labour_import_data
from utils import debug_print, info_print, error_print
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load the raw data from Eurostat API
        
# Labour_demand_ICT_data_import = eurostat.get_data_df('isoc_sk_oja1')
# info_print("Got Labour Demand data")
        
# # Define the starting quarter for filtering data
# date_start = '2019Q4'

# # Process the raw data

# Labour_demand_ICT_data = process_ICT_labour_import_data(Labour_demand_ICT_data_import, date_start)

# data_labour=pd.DataFrame(Labour_demand_ICT_data)
# data_labour.to_parquet('TSA_Parquet_data/labour_demand_data.parquet', index=False)
# info_print("Saved Labour Demand Parquet data")

Labour_demand_ICT_data=pd.read_parquet('TSA_Parquet_data/labour_demand_data.parquet')
info_print("Completed Labour Demand Parquet data read")


# Filter Labour Demand data for the given country, where the unit is 'PC'
filtered_data_labour_demand = Labour_demand_ICT_data[(Labour_demand_ICT_data['geo'] == 'IT') &
                                                         (Labour_demand_ICT_data['unit'] == 'PC')].copy()

# Select only 'quarter' and 'value' columns, rename 'value' to 'employment_value'
filtered_data_labour_demand = filtered_data_labour_demand[['quarter', 'value']]
#Decompose the labour time series to detect trend,seasonality and residuals
decomposition_labour_demand = seasonal_decompose(filtered_data_labour_demand['value'], model='multiplicative', period=4)
decomposition_labour_demand.plot()
plt.show()
#Handling missing values for Labour Demand
#filtered_data_labour_demand['value'].fillna(method='ffill', inplace=True)
filtered_data_labour_demand['value']=filtered_data_labour_demand['value'].interpolate(method='spline',order=2)
filtered_data_labour_demand = filtered_data_labour_demand.rename(columns={'value': f'IT_labour_demand_value'})

#Checking for stationarity using ADF test
result = adfuller(filtered_data_labour_demand['IT_labour_demand_value'])
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

info_print("ACF plot for Labour Demand")

#ACF shows slow decay
plot_acf(filtered_data_labour_demand['IT_labour_demand_value'], ax=ax[0])

#PACF cuts off after lag 1
info_print("PACF plot for Labour Demand")

plot_pacf(filtered_data_labour_demand['IT_labour_demand_value'], ax=ax[1])

#Posisible AR(1) model
plt.show()



