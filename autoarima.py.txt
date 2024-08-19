import pandas as pd
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_error

# Loading the datasets
train_df = pd.read_csv('C:/Personal/DemandForecastingGit/trainingData.csv')
test_df = pd.read_csv('C:/Personal/DemandForecastingGit/testData.csv')

# Preprocessing data as before
# Convert date columns, set index, etc.

# Using auto_arima to find the best ARIMA model
auto_arima_model = auto_arima(train_df['Sales Quantity'], seasonal=True, m=12, stepwise=True,
                              suppress_warnings=True, error_action="ignore", max_order=None, trace=True)

print(auto_arima_model.summary())

# Forecasting using best model
n_periods = len(test_df)
forecast, conf_int = auto_arima_model.predict(n_periods=n_periods, return_conf_int=True)

# Calculation of MAE
mae = mean_absolute_error(test_df['Sales Quantity'], forecast)
print(f"Mean Absolute Error: {mae}")


