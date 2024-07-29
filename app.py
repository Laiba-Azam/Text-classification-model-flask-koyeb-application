from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import pickle
from processing1 import preprocess_text  # Ensure this is correctly imported

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model
model = load_model('fyp_model.h5', compile=False)

# Load the tokenizer
with open('fyp_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the scaler
with open('fyp_scaler.pickle', 'rb') as handle:
    scaler = pickle.load(handle)

# Define the prediction function
def predict_sentiment(tweet):
    # Preprocess the tweet
    processed_tweet = preprocess_text(tweet)
    # Tokenize the preprocessed tweet
    seque = tokenizer.texts_to_sequences([processed_tweet])
    padded_seque = tf.keras.preprocessing.sequence.pad_sequences(seque, maxlen=50, padding='post', truncating='post')
    # Scale the confidence score (constant value of 0.5)
    confidence_score = 0.5
    scaled_confidence_score = scaler.transform(np.array([confidence_score]).reshape(-1, 1))
    # Make prediction
    prediction = model.predict([padded_seque, scaled_confidence_score])
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    # Decode the predicted label using label mapping
    label_mapping = {0: 'mild', 1: 'moderate', 2: 'severe', 3: 'non-depressed'}
    predicted_label = label_mapping.get(predicted_label_index, "Unknown")
    return predicted_label

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    predicted_sentiment = predict_sentiment(text)
    response = {
        'predicted_sentiment': predicted_sentiment
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()
