import numpy as np
import os
import pickle
from flask import Flask , request, render_template
from sklearn.preprocessing import StandardScaler
from joblib import load

app = Flask(__name__)

with open('heart.pkl', 'rb') as file:  
    model = pickle.load(file)
                 
@app.route('/')
def index():
    return render_template('original.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])

        pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope]
        
        sc=load('std_scaler.bin')
        new_pred_args = sc.transform([pred_args])
        print(new_pred_args)

        model_predcition = model.predict(new_pred_args)
        if model_predcition == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        #return res
    return render_template('predict.html', prediction = res)


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)  ## For AWS cloud deployment we need to mention ip(host) as '0.0.0.0' and port as 8080