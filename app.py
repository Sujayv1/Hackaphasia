from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Load the ONNX model
session = ort.InferenceSession('disease_predictor.onnx')

# Get the input name for the model
input_name = session.get_inputs()[0].name

@app.route('/')
def home():
    return "Disease Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the front-end form
        data = request.get_json()

        # Convert the input data to a numpy array (assume 14 features in this case)
        input_features = np.array([
            data['fever'], 
            1 if data['cough_type'].lower() == 'dry' else 0,  # Convert text to numeric
            data['fatigue'],
            data['wbc_count'],
            data['neutrophils'],
            data['lymphocytes'],
            data['procalcitonin'],
            data['crp'],
            1 if data['pcr_result'].lower() == 'positive' else 0,  # Convert text to numeric
            1 if data['sputum_culture'].lower() == 'positive' else 0,  # Convert text to numeric
            1 if data['xray_pattern'].lower() == 'abnormal' else 0,  # Convert text to numeric
            1 if data['ct_pattern'].lower() == 'abnormal' else 0,  # Convert text to numeric
            data['oxygen_level'],
            data['il6_level']
        ], dtype=np.float32).reshape(1, -1)

        # Run inference
        output = session.run(None, {input_name: input_features})

        # Get the predicted disease and confidence (this depends on your model output)
        predicted_disease = output[0][0]  # Assuming output[0] has the prediction
        confidence = output[1][0]  # Assuming output[1] has the confidence score
        
        # You can customize this according to your model’s actual output format
        disease_label = 'Disease A' if predicted_disease == 1 else 'Disease B'
        confidence_percentage = confidence * 100  # Convert to percentage

        return jsonify({
            'predicted_disease': disease_label,
            'confidence': confidence_percentage
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
