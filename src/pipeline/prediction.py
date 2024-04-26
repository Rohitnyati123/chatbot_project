import numpy as np
import pandas as pd
import urllib
from bs4 import BeautifulSoup
import requests
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
import pickle


"""this function is used for clean the data and  train the model on given document"""
def train_model():
    paragraphs=[]
    url = "https://en.wikipedia.org/wiki/Chatbot"

    # Send a GET request to the URL
    response = requests.get(url)
    soup1 = BeautifulSoup(response.content, 'html.parser')
    for i in range(1,55):
        paragraphs.append(soup1('p')[i].text)

    # Split paragraphs into sentences
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))

    # Create input-output pairs
    
    inputs = []
    outputs = []
    for i in range(len(sentences) - 1):
        inputs.append(sentences[i])
        outputs.append(sentences[i + 1])

    # Tokenization
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(inputs + outputs)
    input_sequences = tokenizer.texts_to_sequences(inputs)
    output_sequences = tokenizer.texts_to_sequences(outputs)

    # Padding
    max_seq_length = max([len(seq) for seq in input_sequences + output_sequences])
    padded_input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
    padded_output_sequences = pad_sequences(output_sequences, maxlen=max_seq_length, padding='post')

    # Build the seq2seq model
    vocab_size = len(tokenizer.word_index) + 1
    embedding_size = 64

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size, input_length=max_seq_length))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(vocab_size, activation='softmax'))

    # # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_input_sequences, padded_output_sequences, epochs=50, verbose=1)
    
    return model


"""this is used for creating pickle file of model"""
def create_pickle(model):
    with open('model2.pkl','wb') as f:
        pickle.dump(model,f)


"""its used for generate the response or output of the model"""
def generate_response(input_text,model):
    tokenizer=Tokenizer()
    input_seq = tokenizer.texts_to_sequences([input_text])
    padded_input_seq = pad_sequences(input_seq, maxlen=48, padding='post')
    predicted_seq = model.predict(padded_input_seq)
    predicted_seq = np.argmax(predicted_seq, axis=-1)
    predicted_text = tokenizer.sequences_to_texts(predicted_seq)
    return predicted_text[0]



if __name__=="__main__":
    d1=train_model()
    create_pickle(d1)