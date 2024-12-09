from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract features from the form
    try:
        features = [
            float(request.form.get('longitude')),
            float(request.form.get('latitude')),
            float(request.form.get('housing_median_age')),
            float(request.form.get('total_rooms')),
            float(request.form.get('total_bedrooms')),
            float(request.form.get('population')),
            float(request.form.get('households')),
            float(request.form.get('median_income'))
        ]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)[0]
        
        # Return result
        return render_template('index.html', prediction_text=f'Predicted House Price: ${prediction:,.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
