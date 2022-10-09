import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

#create flask app
app = Flask(__name__, template_folder="template")

#Load the pickle files
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    # whenever the predict_api is been click, then the input to be given is a json file
    data = request.json['data']
    print(data)

    # The data we will received will be in key value, this will give the dictionary value,and give the list
    # we reshape, since the transformation expect a single record with many number of features
    print(np.array(list(data.values())).reshape(1,-1))

    # get this data transform into pickle file, scaling.pickle will trans standardized the entire data
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    
    #make prediction on the new data
    output = regmodel.predict(new_data)

    print(output[0])
    return jsonify(output[0])

if __name__ == '__main__':
    app.run(debug=True)
    