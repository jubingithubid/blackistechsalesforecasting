import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Loading the datasets
train_df = pd.read_csv('C:/Personal/AI/Sales-Forecasting/trainingData.csv')
test_df = pd.read_csv('C:/Personal/AI/Sales-Forecasting/testData.csv')

# Converting date columns and set as index, then explicitly set frequency
train_df['Transaction Date'] = pd.to_datetime(train_df['Transaction Date'])
test_df['Transaction Date'] = pd.to_datetime(test_df['Transaction Date'])
train_df.set_index('Transaction Date', inplace=True)
test_df.set_index('Transaction Date', inplace=True)
train_df.index.freq = 'D'
test_df.index.freq = 'D'

# Plotting the time series of the target variable
plt.figure(figsize=(12, 6))
plt.plot(train_df['Sales Quantity'], label='Sales Quantity')
plt.title('Time Series - Sales Quantity')
plt.xlabel('Date')
plt.ylabel('Sales Quantity')
plt.legend()
plt.show()

# Plotting histograms for all numeric variables to understand distributions
train_df.hist(bins=20, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Heatmap for correlation analysis
plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Align training and testing data to ensure they have the same columns
train_df, test_df = train_df.align(test_df, join='inner', axis=1)

# Separate the target variable
y_train = train_df.pop('Sales Quantity')
y_test = test_df.pop('Sales Quantity')
X_train = train_df
X_test = test_df

# Define models
models = {
    'ARIMA': ARIMA(y_train, order=(5,0,1)),
    'SARIMAX': SARIMAX(y_train, exog=X_train, order=(5,0,1), seasonal_order=(1,1,1,12)),  # Adding seasonal components
    'ExpSmoothing': ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12)
}

# Forecasting and evaluating
forecast_results = {}
mae_scores = {}
pearson_correlations = {}

for model_name, model in models.items():
    fitted_model = model.fit()
    if model_name == 'SARIMAX':
        forecast = fitted_model.get_forecast(steps=len(y_test), exog=X_test).predicted_mean
    else:
        forecast = fitted_model.forecast(steps=len(y_test))
    
    forecast_results[model_name] = forecast
    mae_scores[model_name] = mean_absolute_error(y_test, forecast)
    pearson_correlations[model_name] = pearsonr(y_test, forecast)[0]
    
    
# Print forecasted values
for model_name, forecast in forecast_results.items():
    print(f"Forecast from {model_name}:")
    print(forecast.to_string(), "\n")  # Using to_string() for nicer formatting

# Outputting the forecasts and errors
print("Mean Absolute Errors:")
for model, mae in mae_scores.items():
    print(f"{model}: {mae}")

print("\nPearson Correlation Coefficients:")
for model, corr in pearson_correlations.items():
    print(f"{model}: {corr}")

# Plotting MAE Scores
plt.figure(figsize=(10, 5))
plt.bar(mae_scores.keys(), mae_scores.values(), color='skyblue')
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting Pearson Correlation
plt.figure(figsize=(10, 5))
plt.bar(pearson_correlations.keys(), pearson_correlations.values(), color='lightgreen')
plt.title('Pearson Correlation Coefficient Comparison')
plt.ylabel('Pearson Correlation Coefficient')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
