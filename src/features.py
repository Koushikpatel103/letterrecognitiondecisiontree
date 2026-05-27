import numpy as np

def extract_features(X):
    X = X.values.reshape(-1, 28, 28)

    features = []

    for img in X:
        row_mean = img.mean(axis=1)
        col_mean = img.mean(axis=0)

        extra = [
            img.mean(),
            img.std(),
            img.sum()
        ]

        features.append(np.concatenate([row_mean, col_mean, extra]))

    return np.array(features)