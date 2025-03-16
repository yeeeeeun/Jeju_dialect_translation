import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib

# 모델과 스케일러 로드
model = load_model('rnn_model.keras')  # 변경된 로드 형식
scaler = joblib.load('scaler.pkl')

# Streamlit 애플리케이션 설정
st.title('Crop Growth Prediction')

# 사용자 입력 받기
stem_length = st.number_input('줄기 길이', min_value=0.0, value=0.0)
leaf_cnt = st.number_input('잎 개수', min_value=0, value=0)
leaf_width = st.number_input('잎 너비', min_value=0.0, value=0.0)
leaf_length = st.number_input('잎 길이', min_value=0.0, value=0.0)
stem_thick = st.number_input('줄기 굵기', min_value=0.0, value=0.0)
fr1_cnt = st.number_input('과실 개수', min_value=0, value=0)
ei_value = st.number_input('양액 EC', min_value=0.0, value=0.0)
pl_value = st.number_input('양액 PH', min_value=0.0, value=0.0)
pi_value = st.number_input('배지 PH', min_value=0.0, value=0.0)
el_value = st.number_input('배지 EC', min_value=0.0, value=0.0)

# 예측 버튼
if st.button('Predict'):
    # 입력 데이터를 DataFrame으로 변환
    input_data = pd.DataFrame([[stem_length, leaf_cnt, leaf_width, leaf_length, stem_thick, fr1_cnt, pl_value, ei_value, pi_value, el_value]],
                              columns=['stem_length', 'leaf_cnt', 'leaf_width', 'leaf_length', 'stem_thick', 'fr1_cnt', 'pl_value', 'ei_value', 'pi_value', 'el_value'])

    # 데이터 스케일링
    input_data_scaled = scaler.transform(input_data)

    # 모델 예측
    prediction = model.predict(input_data_scaled.reshape(-1, 1, 10))

    # 예측 결과 출력
    st.write("Predicted Crop Growth in a Week:")
    st.write(f"- 줄기 길이: {prediction[0][0]:.2f}")
    st.write(f"- 잎 개수: {prediction[0][1]:.2f}")
    st.write(f"- 잎 너비: {prediction[0][2]:.2f}")
    st.write(f"- 잎 길이: {prediction[0][3]:.2f}")
    st.write(f"- 줄기 굵기: {prediction[0][4]:.2f}")
