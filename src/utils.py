import os
import joblib

ARTIFACTS_DIR = "artifacts"

def save_object(file_name, obj):
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    path = os.path.join(ARTIFACTS_DIR, file_name)
    joblib.dump(obj, path)

def load_object(file_name):
    path = os.path.join(ARTIFACTS_DIR, file_name)
    return joblib.load(path)