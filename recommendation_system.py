import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def convert_genres(genres):
    result = genres.replace("|", " ").replace("-", "")
    return result


movies = pd.read_csv("Datasets/movie_data/movies.csv", encoding="latin-1", sep="\t", usecols=["title", "genres"])
movies["genres"] = movies["genres"].apply(convert_genres)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(movies["genres"])
cs_matrix = cosine_similarity(tfidf_matrix)
cs_df = pd.DataFrame(cs_matrix, index=movies["title"], columns=movies["title"])

title = "Lawnmower Man 2: Beyond Cyberspace (1996)"
top_n = 30
data = cs_df.loc[title, :]
data = data.sort_values(ascending=False)
result = data[:top_n].to_frame(name="score")
