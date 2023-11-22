import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

with np.load("Datasets/mnist.npz", "r") as f:
    x_train = f["x_train"]
    x_test = f["x_test"]
    y_train = f["y_train"]
    y_test = f["y_test"]

x_train = np.reshape(x_train, (x_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))

cls = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier())
])

cls.fit(x_train, y_train)
y_predicted = cls.predict(x_test)
print(classification_report(y_test, y_predicted))

# precision    recall  f1-score   support
#
#            0       0.97      0.99      0.98       980
#            1       0.99      0.99      0.99      1135
#            2       0.96      0.96      0.96      1032
#            3       0.96      0.96      0.96      1010
#            4       0.98      0.98      0.98       982
#            5       0.98      0.96      0.97       892
#            6       0.97      0.98      0.98       958
#            7       0.97      0.96      0.97      1028
#            8       0.96      0.96      0.96       974
#            9       0.96      0.95      0.96      1009
#
#     accuracy                           0.97     10000
#    macro avg       0.97      0.97      0.97     10000
# weighted avg       0.97      0.97      0.97     10000
