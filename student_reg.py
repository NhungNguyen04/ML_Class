import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


data = pd.read_csv("./Datasets/StudentScore.xls")
# print(data.corr())
# report = ProfileReport(data)
# report.to_file("report.html")

# Split data
target = "reading score"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Data preprocessing
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

education_values = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree",
                    "master's degree"]
gender_values = ["female", "male"]
lunch_values = x_train["lunch"].unique()
test_values = x_train["test preparation course"].unique()

ord_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[education_values, gender_values, lunch_values, test_values]))
])

nom_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num_features", num_transformer, ["math score", "writing score"]),
    ("ord_features", ord_transformer, ["parental level of education", "gender", "lunch", "test preparation course"]),
    ("nom_features", nom_transformer, ["race/ethnicity"]),
])

# Applying model
reg = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

params = {
    "regressor__n_estimators": [50, 100],
    "regressor__criterion": ["squared_error", "absolute_error", "friedman_mse"],
    "preprocessor__num_features__imputer__strategy": ["median", "mean"]
}

grid_search = GridSearchCV(reg, param_grid=params, cv=5, scoring="r2", verbose=1, n_jobs=2)
grid_search.fit(x_train, y_train)

print(grid_search.best_score_)
print(grid_search.best_params_)

# Predict
y_pred = grid_search.predict(x_test)

# Report
print("MSE: {}".format(mean_squared_error(y_test, y_pred)))
print("MAE: {}".format(mean_absolute_error(y_test, y_pred)))
print("R2: {}".format(r2_score(y_test, y_pred)))

# MSE: 19.077372
# MAE: 3.6113999999999997
# R2: 0.9076109789548723

