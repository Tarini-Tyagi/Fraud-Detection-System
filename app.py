from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 415

            data = request.get_json()
            print("Received Data:", data)

            if "features" not in data:
                return jsonify({"error": "Missing 'features' key"}), 400

            # Print expected features
            print("Model expects", model.n_features_in_, "features")

            features = np.array(data["features"]).reshape(1, -1)

            # Check if input matches model expectation
            if features.shape[1] != model.n_features_in_:
                return jsonify({
                    "error": f"Expected {model.n_features_in_} features, but got {features.shape[1]}"
                }), 400

            prediction = model.predict(features)[0]
            return jsonify({"Fraud": bool(prediction)})

        except Exception as e:
            print("Server Error:", str(e))
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

