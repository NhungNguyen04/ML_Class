import pandas as pd
import re
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.metrics import classification_report

def take_location_code(string):
    result = re.findall("\,\s[A=Z]{2}$", string)
    if len(result):
        return result[0][2:]
    else:
        return string


data = pd.read_excel("./Datasets/final_project.ods", engine="odf", dtype="str")
data["location"] = data["location"].apply(take_location_code)
target = "career_level"
x = data.drop(target, axis=1)
y = data[target]
# Stratify: make sure each set has the same ratio of target class as the initial set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

# Handling imbalanced data
ros = SMOTEN(random_state=100, k_neighbors=2,
             sampling_strategy={"director_business_unit_leader": 500, "specialist": 500,
                                "managing_director_small_medium_company": 500, "bereichsleiter": 1000})
# print(y_train.value_counts())
x_train, y_train = ros.fit_resample(x_train, y_train)
# print("=======================================================")
# print(y_train.value_counts())

# Data preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("title", TfidfVectorizer(stop_words="english"), "title"),
    ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("description", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry", TfidfVectorizer(stop_words="english"), "industry"),
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectKBest(chi2, k=200)),
    ("classifier", RandomForestClassifier())
])

params = {
    "preprocessor__description__min_df": [0.01, 0.02],
    "feature_selection__k": [100, 500, 1000]
}

grid_search = GridSearchCV(cls, param_grid=params, cv=5, scoring="f1_weighted", verbose=2, n_jobs=2)
grid_search.fit(x_train, y_train)
y_predict = grid_search.predict(x_test)
print(classification_report(y_test, y_predict))

#                                         precision    recall  f1-score   support
#
#                         bereichsleiter       0.67      0.14      0.23       192
#          director_business_unit_leader       0.67      0.14      0.24        14
#                    manager_team_leader       0.65      0.74      0.69       534
# managing_director_small_medium_company       0.00      0.00      0.00         1
#   senior_specialist_or_project_manager       0.83      0.92      0.88       868
#                             specialist       0.00      0.00      0.00         6
#
#                               accuracy                           0.76      1615
#                              macro avg       0.47      0.32      0.34      1615
#                           weighted avg       0.75      0.76      0.73      1615

