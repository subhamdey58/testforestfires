from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

ridgecv = pickle.load(open('models/ridgecv.pkl','rb'))
scaler = pickle.load(open('models/scaler.pkl','rb'))

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/predictdata',methods=['GET','POST'])
def predict_data():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridgecv.predict(new_data_scaled)
        return render_template('home.html',results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    application.run(host='0.0.0.0')