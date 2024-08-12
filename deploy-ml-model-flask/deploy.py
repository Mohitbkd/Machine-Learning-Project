from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and label encoder
with open('savemodel.sav', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)

@app.route('/predict', methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Predict using the model
    encoded_result = model.predict(features)[0]

    # Decode the prediction
    result = label_encoder.inverse_transform([encoded_result])[0]

    return render_template('index.html', result=result, sepal_length=sepal_length,
                           sepal_width=sepal_width, petal_length=petal_length, petal_width=petal_width)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
