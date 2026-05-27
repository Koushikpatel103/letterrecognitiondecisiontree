from sklearn.datasets import fetch_openml

def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    return X, y