from flask import Flask, request, jsonify
import pandas as pd 
from src.predict import make_predictions


app = Flask(__name__)


# Define the predict route
@app.route("/predict", methods=["POST"])
def predict():

    # Get the input data from the JSON payload
    data_json = request.get_json(force=True) 

    # Convert the JSON object to a Pandas DataFrame
    data = pd.DataFrame([data_json], index=[0])

    # Save the permalink for later
    perma = data['permalink']
    data = data.drop(['permalink'],axis=1)

    # Make a prediction using the loaded model
    prediction = make_predictions(data)

    # Return the prediction as a JSON response
    return jsonify({"prediction": int(prediction[0]), "permalink": str(perma.iloc[0])})

if __name__ == '__main__':
    # Run the app with Gunicorn as the web server
    gunicorn_app = 'app:app'
    workers = 2
    bind_address = '0.0.0.0:8080'
    timeout = 600
    loglevel = 'debug'
    command = f'gunicorn {gunicorn_app} -w {workers} -b {bind_address} --timeout {timeout} --log-level {loglevel}'
    os.system(command)

    
