import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


# STEP 1: Load dataset
data = pd.read_csv("stock_data.csv")

print("Historical Stock Data:")
print(data)


# STEP 2: Convert Date column
data["Date"] = pd.to_datetime(data["Date"])


# STEP 3: Create numerical day column
data["Day"] = np.arange(len(data))


# Input and output
X = data[["Day"]]
y = data["Close"]


# STEP 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# STEP 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)


# STEP 6: Test model
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\\nModel Performance")
print("----------------------")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# STEP 7: Predict next day
next_day = np.array([[len(data)]])

next_price = model.predict(next_day)

print("\\nPredicted Stock Price for Next Day:")
print(round(next_price[0], 2))


# STEP 8: Plot historical prices
plt.figure(figsize=(10,5))

plt.plot(
    data["Date"],
    data["Close"],
    marker="o"
)

plt.title("Historical Stock Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# STEP 9: Predict future prices
future_days = np.arange(
    len(data),
    len(data) + 5
).reshape(-1,1)

future_prices = model.predict(future_days)


# Plot prediction
plt.figure(figsize=(10,5))

plt.plot(
    data["Day"],
    data["Close"],
    marker="o",
    label="Actual Prices"
)

plt.plot(
    future_days,
    future_prices,
    marker="x",
    linestyle="--",
    label="Predicted Prices"
)

plt.title("Stock Price Prediction")
plt.xlabel("Day")
plt.ylabel("Price")

plt.legend()

plt.tight_layout()

plt.show()