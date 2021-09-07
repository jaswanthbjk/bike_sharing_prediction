import joblib
import pandas as pd
import streamlit as st

from PIL import Image

from preprocess import preprocess_inputs


@st.cache(allow_output_mutation=True)
def load(model_path, scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def inference(data, scaler, model, coloumn_names):
    X = scaler.transform(data)
    input_features = pd.DataFrame(X, columns=coloumn_names)
    prediction = model.predict(input_features)
    output = "The number of Bikes rented are {}".format(int(prediction[0]))
    return output


st.title('Bike-sharing count Prediction App')
st.write('Historical data for bike sharing in London Powered by TfL Open Data')
image = Image.open('data/bike-share.jpg')
st.image(image, use_column_width=True)
st.write('Please fill in the details of the day and its conditions')

datetime = st.text_input("Timestamp", '2011-01-01 00:00:00',
                         help='Year-Month-Day Hours:Minutes:Seconds')
season = st.sidebar.slider("Season", 1, 4, 1, 1)
weather = st.sidebar.slider("Weather", 1, 4, 1, 1)
workingday = st.sidebar.number_input("Workingday or Not", 0, 1, 0, 1)
holiday = st.sidebar.number_input("Holiday or Not", 0, 1, 0, 1)

temp = st.sidebar.number_input("Temperature", 0, 100, 0, 1)
atemp = st.sidebar.number_input("Air Temperature", 0, 100, 0, 1)
humidity = st.sidebar.number_input("Humidity", 0, 100, 0, 1)
windspeed = st.sidebar.number_input("Windspeed", 0, 100, 0, 1)
casual = st.sidebar.number_input("Casual usage count", 0, 400, 0, 1)
registered = st.sidebar.number_input("Registered usage count", 0, 10000, 0, 1)

row = [datetime, season, holiday, workingday, weather, temp, atemp, humidity,
       windspeed, casual, registered]

if st.button('Find Bike Sharing Count'):
    feat_cols = ['datetime', 'season', 'holiday', 'workingday', 'weather',
                 'temp', 'atemp', 'humidity', 'windspeed', 'casual',
                 'registered']
    data = pd.DataFrame([row], columns=feat_cols)
    data = preprocess_inputs(data, dump_flag=False, dump_path=None, test=True)

    coloumn_names = ['month', 'day', 'hour', 'season', 'holiday', 'workingday',
                     'temp', 'atemp', 'humidity', 'windspeed', 'casual',
                     'registered', 'weather']
    scaler, model = load('models/scaler.joblib', 'models/model.joblib')
    result = inference(data, scaler, model, coloumn_names)
    st.write(result)
