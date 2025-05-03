from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert to float
        glucose = float(request.form['Glucose'])
        bp = float(request.form['BloodPressure'])
        skin = float(request.form['SkinThickness'])
        insulin = float(request.form['Insulin'])
        bmi = float(request.form['BMI'])

        # Create input array for prediction
        data = np.array([[glucose, bp, skin, insulin, bmi]])

        # Predict using the loaded model
        prediction = model.predict(data)

        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'
        return render_template('index.html', prediction_text=f'Result: {result}')

    except ValueError:
        return render_template('index.html', prediction_text='Invalid input. Please enter numeric values only.')

if __name__ == '__main__':
    app.run(debug=True)
