import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lazypredict.Supervised import LazyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Read data
data = pd.read_csv("Datasets/diabetes.csv")
# report = ProfileReport(data)
# report.to_file("report.html")

# Split data
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Preprocess data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define a model
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(x_train, x_test, y_train, y_test)
# print(models)
cls = LogisticRegression()
# cls = RandomForestClassifier()
cls.fit(x_train, y_train)
prediction = cls.predict(x_test)
print("_________ Logistic Regression _________________________")
print(classification_report(y_true=y_test, y_pred=prediction))

params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [None, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=5, scoring="f1", verbose=2, n_jobs=2)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
prediction = grid_search.predict(x_test)
print("________________ Random Forest _________________________")
print(classification_report(y_true=y_test, y_pred=prediction))
