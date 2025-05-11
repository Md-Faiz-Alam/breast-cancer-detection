from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#Load the Saved models
model_logistic = pickle.load(open('./models/model_logistic.pkl', 'rb'))
model_random_forest = pickle.load(open('./models/model_random_forest.pkl', 'rb'))
scaler = pickle.load(open('./models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST']) 
def predict():
    if request.method == 'POST':
        # Get the values from the form
        features = [
            float(request.form['mean_radius']),
            float(request.form['mean_texture']),
            float(request.form['mean_perimeter']),
            float(request.form['mean_area']),
            float(request.form['mean_smoothness']),
            float(request.form['mean_compactness']),
            float(request.form['mean_concavity']),
            float(request.form['mean_concave_points']),
            float(request.form['mean_symmetry']),
            float(request.form['mean_fractal_dimension']),
            float(request.form['radius_error']),
            float(request.form['texture_error']),
            float(request.form['perimeter_error']),
            float(request.form['area_error']),
            float(request.form['smoothness_error']),
            float(request.form['compactness_error']),
            float(request.form['concavity_error']),
            float(request.form['concave_points_error']),
            float(request.form['symmetry_error']),
            float(request.form['fractal_dimension_error']),
            float(request.form['worst_radius']),
            float(request.form['worst_texture']),
            float(request.form['worst_perimeter']),
            float(request.form['worst_area']),
            float(request.form['worst_smoothness']),
            float(request.form['worst_compactness']),
            float(request.form['worst_concavity']),
            float(request.form['worst_concave_points']),
            float(request.form['worst_symmetry']),
            float(request.form['worst_fractal_dimension'])
        ]

        # Reshape and scale the features
        features_array = np.array(features).reshape(1, -1)
        scaled_features = scaler.transform(features_array)

        # Get the model choice
        model_choice = request.form['model_choice']

        if model_choice == 'Logistic Regression':
            model = model_logistic
        elif model_choice == 'Random Forest':
            model = model_random_forest
        else:
            return render_template('index.html', prediction_text='Invalid model selected. Please choose Logistic Regression or Random Forest.')

        # Make prediction and get confidence
        prediction = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0]  
        
        print("Prediction:", prediction) 
        print("Probabilities:", proba) 

        if prediction == 0:  
            confidence = round(proba[0] * 100, 2)  
        else:
            confidence = round(proba[1] * 100, 2)  

        print("Confidence:", confidence)  

        # Determine risk level based on confidence
        if confidence < 50:
            confidence_level = "Low"
        elif 50 <= confidence < 75:
            confidence_level = "Moderate"
        else:
            confidence_level = "High"

        
        if prediction == 0:
            label = "Malignant - Cancer Detected ❗"
        else:
            label = "Benign - No Cancer Detected ✅"

        result = f"<b>Prediction:</b> {label}<br><b>Confidence in Diagnosis:</b> {confidence}%<br><b>Confidence Level:</b> {confidence_level}"

        return result

if __name__ == "__main__":
    app.run(debug=True)