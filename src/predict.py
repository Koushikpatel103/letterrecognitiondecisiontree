import numpy as np
from src.utils import load_object

model = load_object("model.pkl")
label_encoder = load_object("label_encoder.pkl")


def predict_letter(features):

    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    result = label_encoder.inverse_transform(prediction)

    return result[0]