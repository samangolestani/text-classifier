import pandas as pd
import tensorflow as tf
import os

from config import *
from utils.preprocessing import prepare_tokenizers, prepare_data

# Load dataset
df = pd.read_csv(DATASET_PATH)
texts = df['word'].tolist()
labels = df['type'].tolist()

# Prepare tokenizers and data
text_tokenizer, label_tokenizer = prepare_tokenizers(texts, labels, VOCAB_SIZE, OOV_TOKEN)
padded_sequences, y = prepare_data(texts, labels, text_tokenizer, label_tokenizer)

# Build model
input_layer = tf.keras.layers.Input(shape=(padded_sequences.shape[1],))
embedding_layer = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM)(input_layer)
x = tf.keras.layers.GlobalAveragePooling1D()(embedding_layer)
x = tf.keras.layers.Dense(32, activation='relu')(x)
output_layer = tf.keras.layers.Dense(len(label_tokenizer.word_index), activation='softmax')(x)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.summary()
model.fit(padded_sequences, y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT)

# Save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"Model saved at {MODEL_PATH}")

# Save tokenizers
import pickle
with open('models/text_tokenizer.pkl', 'wb') as f:
    pickle.dump(text_tokenizer, f)
with open('models/label_tokenizer.pkl', 'wb') as f:
    pickle.dump(label_tokenizer, f)