# import streamlit as st
# import numpy as np
# import time
# from sklearn.decomposition import DictionaryLearning
# import json
# import matplotlib.pyplot as plt

# # Set up Streamlit title and instructions
# st.title("IoMT EEG Data Compression and Decompression Server")
# st.write("Upload compressed EEG data to measure decompression and processing time.")

# # File uploader to receive compressed EEG data in JSON format
# uploaded_file = st.file_uploader("Upload compressed EEG data (JSON format)", type="json")

# if uploaded_file is not None:
#     # Load the JSON data
#     data = json.load(uploaded_file)
#     compressed_signal = np.array(data["compressed_signal"])
#     sample_duration_seconds = data["sample_duration_seconds"]
#     n_components = data["n_components"]

#     # Start decompression timer
#     start_time = time.time()

#     # Reconstruct signal using Dictionary Learning
#     dictionary_learner = DictionaryLearning(n_components=n_components)
#     dictionary_learner.fit(compressed_signal.reshape(-1, 1))  # Fit model on compressed data
#     decompressed_signal = np.dot(dictionary_learner.components_.T, compressed_signal.reshape(-1, 1)).flatten()

#     # End timer and calculate the decompression process time
#     end_time = time.time()
#     process_time = end_time - start_time

#     # Display decompressed signal plot
#     st.write("Decompressed Signal Visualization")
#     fig, ax = plt.subplots()
#     ax.plot(np.linspace(0, sample_duration_seconds, len(decompressed_signal)), decompressed_signal)
#     ax.set_title("Decompressed EEG Signal")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude")
#     st.pyplot(fig)

#     # Display processing time
#     st.write(f"Process Time: {process_time:.2f} seconds")


import streamlit as st
import numpy as np
from sklearn.decomposition import DictionaryLearning
import matplotlib.pyplot as plt

# Title and description
st.title("EEG Seizure Detection - IoMT Framework")
st.write("Upload EEG data to test compression, decompression, and visualization within Streamlit.")

# Sidebar configuration
n_components = st.sidebar.slider("Number of Components for Compression", 1, 20, 10)
alpha = st.sidebar.slider("Sparsity Level (Alpha)", 0.0, 1.0, 0.2)
sample_duration_seconds = st.sidebar.slider("Sample Duration (seconds)", 10, 60, 30)

# File upload
uploaded_file = st.file_uploader("Upload EEG data file", type="edf")

if uploaded_file:
    # Load the EEG data from the uploaded EDF file
    import pyedflib
    edf_reader = pyedflib.EdfReader(uploaded_file)
    signal_index = 0
    sampling_frequency = edf_reader.getSampleFrequency(signal_index)
    signal = edf_reader.readSignal(signal_index)

    # Sample selection
    sample_points = int(sampling_frequency * sample_duration_seconds)
    sample_signal = signal[:sample_points]

    # Display original signal
    st.subheader("Original EEG Signal")
    plt.figure(figsize=(10, 4))
    time = np.linspace(0, sample_duration_seconds, sample_points)
    plt.plot(time, sample_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Original EEG Signal")
    st.pyplot()

    # Apply Dictionary Learning for compression
    sample_signal_reshaped = sample_signal.reshape(-1, 1)
    dictionary_learner = DictionaryLearning(n_components=n_components, alpha=alpha, max_iter=1000)
    compressed_signal = dictionary_learner.fit_transform(sample_signal_reshaped).flatten()

    # Display compressed signal
    st.subheader("Compressed Signal")
    plt.figure(figsize=(10, 4))
    compressed_time = np.linspace(0, sample_duration_seconds, len(compressed_signal))
    plt.plot(compressed_time, compressed_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Compressed EEG Signal")
    st.pyplot()

    # Decompress/Reconstruct the signal
    reconstructed_signal = np.dot(dictionary_learner.components_.T, dictionary_learner.transform(sample_signal_reshaped).T).T.flatten()

    # Display reconstructed signal
    st.subheader("Reconstructed EEG Signal")
    plt.figure(figsize=(10, 4))
    plt.plot(time, reconstructed_signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Reconstructed EEG Signal")
    st.pyplot()

    # Performance metrics
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(sample_signal, reconstructed_signal)
    st.write(f"Mean Squared Error between Original and Reconstructed Signal: {mse}")
