import pandas as pd

from isolation_forest import IsolationForest


def score(X, n_trees, sub_sample_size):
    model = IsolationForest(n_trees=n_trees, sub_sample_size=sub_sample_size)
    model.fit(X)
    average_path_length, anomaly_scores = model.anomaly_score(X)
    prediction = model.predict(anomaly_scores)
    return average_path_length, anomaly_scores, prediction


if __name__ == "__main__":

    df = pd.read_csv("data/data.csv")

    average_path_length, anomaly_scores, prediction = score(df, n_trees=100, sub_sample_size=256)

    df["score"] = prediction

    print("Done")
