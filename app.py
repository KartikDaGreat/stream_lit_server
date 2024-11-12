import streamlit as st
import numpy as np
from sklearn.decomposition import DictionaryLearning
from sklearn.metrics import mean_squared_error
import time
from fastapi import FastAPI
from pydantic import BaseModel
from streamlit.server import Server

# Initialize FastAPI
app = FastAPI()

# Define data model
class EEGData(BaseModel):
    compressed_signal: list
    original_signal: list
    n_components: int

# Function to decompress the signal
def decompress_signal(compressed_signal, dictionary_learner):
    return np.dot(dictionary_learner.components_.T, np.array(compressed_signal).T).T.flatten()

# FastAPI endpoint for decompression
@app.post("/decompress")
async def decompress(data: EEGData):
    original_signal = np.array(data.original_signal)
    compressed_signal = np.array(data.compressed_signal)
    n_components = data.n_components

    # Set up Dictionary Learning for decompression
    dictionary_learner = DictionaryLearning(n_components=n_components)
    dictionary_learner.fit(original_signal.reshape(-1, 1))

    # Decompress and time the process
    start_time = time.time()
    reconstructed_signal = decompress_signal(compressed_signal, dictionary_learner)
    process_time = time.time() - start_time

    # Calculate metrics
    mse = mean_squared_error(original_signal, reconstructed_signal)
    compression_ratio = len(original_signal) / len(compressed_signal)

    # Return the results
    return {
        "decompression_time": process_time,
        "mse": mse,
        "compression_ratio": compression_ratio,
    }

# Embed FastAPI into Streamlit
if 'server' not in Server._singleton:
    Server._singleton['server'] = app

# Display Streamlit frontend
st.title("IoMT Framework: EEG Compression and Decompression Test")
st.write("Use the provided API endpoint to test compression and decompression.")
