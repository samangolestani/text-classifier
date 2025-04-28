from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def prepare_tokenizers(texts, labels, vocab_size, oov_token):
    text_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    text_tokenizer.fit_on_texts(texts)
    
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    
    return text_tokenizer, label_tokenizer

def prepare_data(texts, labels, text_tokenizer, label_tokenizer):
    sequences = text_tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, padding='post')
    
    y = np.array(label_tokenizer.texts_to_sequences(labels)) - 1
    
    return padded_sequences, y