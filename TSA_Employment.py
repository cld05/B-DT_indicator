import eurostat

import matplotlib.pyplot as plt

from data_processing import process_import_data, process_ICT_labour_import_data
from utils import debug_print, info_print, error_print
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf



# Load the raw data from Eurostat API
        
# Employment_data_import = eurostat.get_data_df('namq_10_a10_e')
# info_print("Got Employment data")


# # Define the starting quarter for filtering data
# date_start = '2019Q4'

# # Process the raw data

# Employment_data = process_import_data(Employment_data_import, date_start)

# data_employment=pd.DataFrame(Employment_data)
# data_employment.to_parquet('TSA_Parquet_data/employment_data.parquet', index=False)
# info_print("Saved Employment Parquet data")

Employment_data=pd.read_parquet('TSA_Parquet_data/employment_data.parquet')
info_print("Completed Employment Parquet data read")

# Filter employment data for the given country, with similar filtering criteria (for employment)
filtered_data_employment = Employment_data[(Employment_data['nace_r2'] == 'J') & 
                                               (Employment_data['unit'] == 'PC_TOT_PER') & 
                                               (Employment_data['geo'] == 'IT') & 
                                               (Employment_data['na_item'] == 'EMP_DC') & 
                                               (Employment_data['s_adj'] == 'NSA')].copy()

# Select only 'quarter' and 'value' columns, rename 'value' to 'employment_value'
filtered_data_employment = filtered_data_employment[['quarter', 'value']]
#Decompose the Employment time series to detect trend,seasonality and residuals
decomposition_employment = seasonal_decompose(filtered_data_employment['value'], model='multiplicative', period=4)
decomposition_employment.plot()
plt.show()
#Handling missing values for Employment
#filtered_data_employment['value'].fillna(method='bfill', inplace=True)
filtered_data_employment['value']=filtered_data_employment['value'].interpolate(method='spline',order=2)
filtered_data_employment = filtered_data_employment.rename(columns={'value': f'IT_employment_value'})


#Checking for stationarity using ADF test
result = adfuller(filtered_data_employment['IT_employment_value'])
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

info_print("ACF plot for Employment Data")

fig, ax = plt.subplots(2,1, figsize=(10,8))

plot_acf(filtered_data_employment['IT_employment_value'], ax=ax[0])

info_print("PACF plot for Employment Data")

plot_pacf(filtered_data_employment['IT_employment_value'], ax=ax[1])

plt.show()




