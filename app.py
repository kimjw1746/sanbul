import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

# 1. 저장된 파이프라인(전처리 도구) 로드
# (이제 서버가 켜질 때마다 CSV를 읽고 새로 학습할 필요가 없습니다)
full_pipeline = joblib.load('./models/pipeline.pkl')

# 2. 저장된 Keras 모델 로드

model = keras.models.load_model('./models/fires_model.h5')

class LabForm(FlaskForm):
    longitude = StringField('longitude (1-7)', validators=[DataRequired()])
    latitude = StringField('latitude (1-7)', validators=[DataRequired()])
    month = StringField('month(01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day (00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # 1. 폼 입력 데이터를 DataFrame으로 변환
        input_data = {
            'longitude': [float(request.form.get('longitude'))],
            'latitude': [float(request.form.get('latitude'))],
            'month': [request.form.get('month')],
            'day': [request.form.get('day')],
            'avg_temp': [float(request.form.get('avg_temp'))],
            'max_temp': [float(request.form.get('max_temp'))],
            'max_wind_speed': [float(request.form.get('max_wind_speed'))],
            'avg_wind': [float(request.form.get('avg_wind'))]
        }
        input_df = pd.DataFrame(input_data)

        # 2. 파이프라인을 통한 데이터 전처리
        X_new_prepared = full_pipeline.transform(input_df)

        # 3. 예측 수행 (결과값은 로그 변환된 값)
        prediction_log = model.predict(X_new_prepared)[0][0]

        # 4. 로그 역변환: 학습 단계의 수식에 맞추어 지수 함수로 면적 복원
        prediction = np.exp(prediction_log) - 1
        prediction_rounded = round(prediction, 2)

        return render_template('result.html', 
                               burned_area_pred=prediction_rounded,
                               longitude=input_data['longitude'][0],
                               latitude=input_data['latitude'][0],
                               month=input_data['month'][0],
                               day=input_data['day'][0],
                               avg_temp=input_data['avg_temp'][0],
                               max_temp=input_data['max_temp'][0],
                               max_wind_speed=input_data['max_wind_speed'][0],
                               avg_wind=input_data['avg_wind'][0])

    return render_template('prediction.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)