import streamlit as st
import numpy as np
import time
from sklearn.decomposition import DictionaryLearning
import json
import matplotlib.pyplot as plt

# Set up Streamlit title and instructions
st.title("IoMT EEG Data Compression and Decompression Server")
st.write("Upload compressed EEG data to measure decompression and processing time.")

# File uploader to receive compressed EEG data in JSON format
uploaded_file = st.file_uploader("Upload compressed EEG data (JSON format)", type="json")

if uploaded_file is not None:
    # Load the JSON data
    data = json.load(uploaded_file)
    compressed_signal = np.array(data["compressed_signal"])
    sample_duration_seconds = data["sample_duration_seconds"]
    n_components = data["n_components"]

    # Start decompression timer
    start_time = time.time()

    # Reconstruct signal using Dictionary Learning
    dictionary_learner = DictionaryLearning(n_components=n_components)
    dictionary_learner.fit(compressed_signal.reshape(-1, 1))  # Fit model on compressed data
    decompressed_signal = np.dot(dictionary_learner.components_.T, compressed_signal.reshape(-1, 1)).flatten()

    # End timer and calculate the decompression process time
    end_time = time.time()
    process_time = end_time - start_time

    # Display decompressed signal plot
    st.write("Decompressed Signal Visualization")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, sample_duration_seconds, len(decompressed_signal)), decompressed_signal)
    ax.set_title("Decompressed EEG Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # Display processing time
    st.write(f"Process Time: {process_time:.2f} seconds")
