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


import os
import pyedflib
import numpy as np
import streamlit as st
from sklearn.decomposition import DictionaryLearning
import time

st.title("EEG Data Compression and Transmission")

uploaded_file = st.file_uploader("Upload EDF file", type=["edf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.edf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the EDF file
    edf_reader = pyedflib.EdfReader("temp.edf")

    # Signal selection and sampling parameters
    signal_index = 0  # Adjust if needed
    sampling_frequency = edf_reader.getSampleFrequency(signal_index)
    signal = edf_reader.readSignal(signal_index)

    # Select a sample of the signal
    sample_duration_seconds = 30
    sample_points = int(sampling_frequency * sample_duration_seconds)
    sample_signal = signal[:sample_points]
    start_time = time.time()
    # Apply Dictionary Learning for compression
    n_components = 10
    dictionary_learner = DictionaryLearning(n_components=n_components, alpha=0.2, max_iter=1000)
    sample_signal_reshaped = sample_signal.reshape(-1, 1)
    compressed_signal = dictionary_learner.fit_transform(sample_signal_reshaped).flatten()

    # Display compression results
    st.write("Original Signal Length:", len(sample_signal))
    st.write("Compressed Signal Length:", len(compressed_signal))

    # Process timing example
    # (Your compression/transmission logic)
    process_time = time.time() - start_time
    st.write("Process Time:", process_time, "seconds")

    # Clean up the temporary file
    edf_reader._close()
    del edf_reader
    os.remove("temp.edf")
