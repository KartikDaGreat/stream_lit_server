from flask import Flask, request, jsonify
import numpy as np
from sklearn.decomposition import DictionaryLearning

app = Flask(__name__)

@app.route('/process', methods=['POST'])
def process_data():
    # Get data from request
    data = request.get_json()
    compressed_signal = np.array(data.get("compressed_signal"))
    sample_duration_seconds = data.get("sample_duration_seconds")
    n_components = data.get("n_components")

    # Perform decompression using the provided Dictionary Learning parameters
    dictionary_learner = DictionaryLearning(n_components=n_components, alpha=0.2, max_iter=1000)
    reconstructed_signal = dictionary_learner.inverse_transform(compressed_signal.reshape(-1, 1)).flatten()

    # Send back reconstructed signal
    return jsonify({
        "reconstructed_signal": reconstructed_signal.tolist(),
        "processing_time_seconds": sample_duration_seconds
    })

if __name__ == '__main__':
    app.run(debug=True)
