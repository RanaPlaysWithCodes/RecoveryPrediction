from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template',static_folder='static')
# Load your trained model
model = pickle.load(open('lrg_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_features = np.array(data['features']).reshape(1, -1) # type: ignore
        prediction = model.predict(input_features)
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ =="__main__":
    app.run(debug=True)
