# stock-price-predictor
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
data = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
print(data.head())
data = data[['Close']]
data['Target'] = data['Close'].shift(-1)

# Drop the last row (since it has NaN target)
data.dropna(inplace=True)

# Features and labels
X = data[['Close']]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)  # no shuffle because it's time-series
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Compare predictions to actual
df_result = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(df_result.head())
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual Price')
plt.plot(predictions, label='Predicted Price')
plt.legend()
plt.title("Apple Stock Price Prediction vs Actual")
plt.xlabel("Days")
plt.ylabel("Price ($)")
plt.grid(True)
plt.show()

