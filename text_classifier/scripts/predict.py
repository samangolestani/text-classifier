import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

from config import MODEL_PATH

# Load model and tokenizers
model = tf.keras.models.load_model(MODEL_PATH)

with open('models/text_tokenizer.pkl', 'rb') as f:
    text_tokenizer = pickle.load(f)
with open('models/label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)

# Prediction function
def predict_type(word):
    seq = text_tokenizer.texts_to_sequences([word])
    pad = pad_sequences(seq, maxlen=model.input_shape[1], padding='post')
    pred = model.predict(pad)
    class_index = np.argmax(pred)
    label = list(label_tokenizer.word_index.keys())[list(label_tokenizer.word_index.values()).index(class_index + 1)]
    return label

# Example usage
if __name__ == "__main__":
    while True:
        word = input("Enter a word (or 'exit' to quit): ")
        if word.lower() == 'exit':
            break
        label = predict_type(word)
        print(f"Predicted type: {label}")