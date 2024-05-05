import os
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

class CustomModel:
    def __init__(self, model_dir):
        self.model = self.load_model(model_dir)
        self.vectorizer = self.load_vectorizer(model_dir)

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, "model.pkl")
        return joblib.load(model_path)

    def load_vectorizer(self, model_dir):
        vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        return joblib.load(vectorizer_path)

    def preprocess_input(self, input_data):
        X = input_data['url']
        X_vect = self.vectorizer.transform(X)
        return X_vect

    def predict(self, input_data):
        processed_data = self.preprocess_input(input_data)
        return self.model.predict(processed_data)

    def serialize_output(self, prediction):
        return json.dumps({"type": prediction.tolist()})

def model_fn(model_dir):
    return CustomModel(model_dir)

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        return json.loads(request_body)
    else:
        raise ValueError("This model only supports application/json input")

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    return model.serialize_output(prediction)
