from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)
model = joblib.load('model.pkl')
sc = joblib.load('sc.pkl')
columns = ['Age (yrs)', 'Weight (Kg)',
       'Height(Cm)', 'BMI', 'Blood Group', 'Pulse rate(bpm)',
       'RR (breaths/min)', 'Hb(g/dl)', 'Cycle(R/I)', 'Cycle length(days)',
       'Marraige Status (Yrs)', 'Pregnant(Y/N)', 'No. of abortions',
       'I   beta-HCG(mIU/mL)', 'II    beta-HCG(mIU/mL)', 'FSH(mIU/mL)',
       'LH(mIU/mL)', 'FSH/LH', 'Hip(inch)', 'Waist(inch)', 'Waist:Hip Ratio',
       'TSH (mIU/L)', 'AMH(ng/mL)', 'PRL(ng/mL)', 'Vit D3 (ng/mL)',
       'PRG(ng/mL)', 'RBS(mg/dl)', 'Weight gain(Y/N)', 'hair growth(Y/N)',
       'Skin darkening (Y/N)', 'Hair loss(Y/N)', 'Pimples(Y/N)',
       'Fast food (Y/N)', 'Reg.Exercise(Y/N)', 'BP _Systolic (mmHg)',
       'BP _Diastolic (mmHg)', 'Follicle No. (L)', 'Follicle No. (R)',
       'Avg. F size (L) (mm)', 'Avg. F size (R) (mm)', 'Endometrium (mm)']

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    try:
        json_data = request.json
        data = pd.DataFrame([json_data])
        data["FSH/LH"] = data["FSH(mIU/mL)"]/data["LH(mIU/mL)"]
        data["Waist:Hip Ratio"] = data["Waist(inch)"]/data["Hip(inch)"]
        data = data[columns]

        prediction = model.predict(sc.transform(data.to_numpy()))
        if prediction[0] == 0:
            output = "You are predicted to have PCOS."
        elif prediction[0] == 1:
            output = "You are predicted not to have PCOS."
            
        return output
    except:
        return "Wrong Input Data"

if __name__ == '__main__':
    app.run()
