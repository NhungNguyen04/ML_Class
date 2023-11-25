import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_direct_data(df, window_size, target, target_size):
    i = 1
    while (i < window_size):
        df[f"{target}_{i}"] = df[target].shift(-i)
        i += 1
    j = 0
    while (j < target_size):
        df[f"target_{j}"] = df[target].shift(-j-window_size)
        j += 1
    df = df.dropna(axis=0)
    return df


data = pd.read_csv("Datasets/Time-series-datasets/bike-sharing-dataset.csv")
target = "users"
data = data.get(["date_time", target])
data[target] = data[target].interpolate()

# fig, ax = plt.subplots()
# ax.plot(data["date_time"], data[target])
# ax.set_xlabel("date_time")
# ax.set_ylabel(target)
# plt.show()

window_size = 5
target_size = 3
train_size = int(0.8*len(data))
data = create_direct_data(data, window_size, target, target_size)

target = ["target_{}".format(i) for i in range(target_size)]
x = data.drop(["date_time"]+target, axis=1)
y = data[target]
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]

r2 = []
mse = []
mae = []
# regs = [LinearRegression() for _ in range(target_size)]
regs = [RandomForestRegressor() for _ in range(target_size)]

for i, reg in enumerate(regs):
    reg.fit(x_train, y_train["target_{}".format(i)])
    y_predict = reg.predict(x_test)
    mse.append(mean_squared_error(y_true=y_test["target_{}".format(i)], y_pred=y_predict))
    mae.append(mean_absolute_error(y_true=y_test["target_{}".format(i)], y_pred=y_predict))
    r2.append(r2_score(y_true=y_test["target_{}".format(i)], y_pred=y_predict))

print("MSE: {}".format(mse))
print("MAE: {}".format(mae))
print("R2: {}".format(r2))

# LINEAR REGRESSION
# MSE: [12099.196116462235, 31255.567453575884, 42035.29937145821]
# MAE: [70.0270537157487, 121.01916799327027, 147.9486568670997]
# R2: [0.7507948321686305, 0.3562299481878003, 0.134144580665289]

# RANDOM FOREST REGRESSION
# MSE: [3853.911668460831, 10330.405504231754, 15990.524750182662]
# MAE: [38.47640251668251, 61.71035476436303, 77.781949436659]
# R2: [0.9206216103201015, 0.7872249257167317, 0.6706224834846185]
