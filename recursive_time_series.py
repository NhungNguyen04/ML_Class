import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("Datasets/Time-series-datasets/web-traffic.csv")


def create_recursive_data(df, window_size, target):
    i = 1
    while i < window_size:
        df[f"{target}_{i}"] = df[f"{target}"].shift(-i)
        i += 1
    df["target"] = df[f"{target}"].shift(-i)
    df = df.dropna(axis=0)
    return df


# FILL IN MISSING VALUES
data["users"] = data["users"].interpolate()

# VISUALIZATION
data["date"] = pd.to_datetime(data["date"], infer_datetime_format=True, format="%d/%m/%y")
# fig, ax = plt.subplots()
# ax.plot(data["date"], data["users"])
# ax.set_xlabel("time")
# ax.set_ylabel("CO2")
# plt.show()

target = "users"
window_size = 5
train_size = int(len(data)*0.8)
data = create_recursive_data(data, window_size, target)
x = data.drop(["date", "target"], axis=1)
y = data["target"]

# poly = PolynomialFeatures(2)
# x = poly.fit_transform(x)

x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

# reg = LinearRegression()
reg = RandomForestRegressor()
reg.fit(x_train, y_train)
y_predict = reg.predict(x_test)
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# LINEAR REGRESSION

# MSE: 287902.66418643325
# MAE: 409.44993743794714
# R2: 0.5268133920297756

# MSE: 264528.0763374954
# MAE: 373.04319750474684
# R2: 0.558478100490603

# RANDOM FOREST REGRESSION
# MSE: 153430.9961925
# MAE: 270.38925
# R2: 0.7478262563113764

