from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from src.preprocess import preprocess_data
from src.utils import save_object


def train_model():

    file_path = "data/letter-recognition.csv"

    (
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        feature_names
    ) = preprocess_data(file_path)

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    save_object("model.pkl", model)
    save_object("label_encoder.pkl", label_encoder)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()