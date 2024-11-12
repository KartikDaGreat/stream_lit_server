import streamlit as st
import numpy as np
from sklearn.decomposition import DictionaryLearning
import time
from sklearn.metrics import mean_squared_error

st.title("IoMT Framework: EEG Compression and Decompression Test")

def decompress_signal(compressed_signal, dictionary_learner):
    # Decompress the signal
    return np.dot(dictionary_learner.components_.T, compressed_signal.T).T.flatten()

# Receive the compressed data
if "compressed_signal" in st.experimental_get_query_params():
    compressed_signal = np.array(st.experimental_get_query_params().get("compressed_signal")[0].split(","), dtype=float)
    original_signal = np.array(st.experimental_get_query_params().get("original_signal")[0].split(","), dtype=float)
    n_components = int(st.experimental_get_query_params().get("n_components")[0])

    # Reconstruct dictionary learning model for decompression
    dictionary_learner = DictionaryLearning(n_components=n_components, fit_algorithm="lars")
    dictionary_learner.fit(original_signal.reshape(-1, 1))  # Fit model to reconstruct components

    # Decompress signal
    start_time = time.time()
    reconstructed_signal = decompress_signal(compressed_signal, dictionary_learner)
    process_time = time.time() - start_time

    # Calculate metrics
    mse = mean_squared_error(original_signal, reconstructed_signal)
    compression_ratio = len(original_signal) / len(compressed_signal)

    # Display results
    st.write("Decompression Process Time:", process_time, "seconds")
    st.write("Mean Squared Error:", mse)
    st.write("Compression Ratio:", compression_ratio)
