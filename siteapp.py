import streamlit as st
import numpy as np
import pandas as pd

import pickle
model = pickle.load(open("l.pkl",'rb'))

st.header("Site Energy Prediction")
st.write("Give the input features :")
def ip_features():
    april_max_temp = st.number_input('april_max_temp')
    april_min_temp = st.number_input('april_min_temp')
    august_avg_temp = st.number_input('august_avg_temp')
    august_max_temp = st.number_input('august_max_temp')
    days_above_90F = st.number_input('days_above_90F')
    january_avg_temp = st.number_input('january_avg_temp')
    july_avg_temp = st.number_input('july_avg_temp')
    july_min_temp = st.number_input('july_min_temp')
    june_avg_temp = st.number_input('june_avg_temp')
    september_min_temp = st.number_input('september_min_temp')
    data = {
            'april_max_temp':april_max_temp,
            'april_min_temp':april_min_temp,
            'august_avg_temp':august_avg_temp,
            'august_max_temp':august_max_temp,
            'days_above_90F':days_above_90F,
            'january_avg_temp':january_avg_temp,
            'july_avg_temp':july_avg_temp,
            'july_min_temp':july_min_temp,
            'june_avg_temp':june_avg_temp,
            'september_min_temp':september_min_temp}
    features = pd.DataFrame(data, index=[0])
    return features

def main():
    df = ip_features()
    st.write("The input features you've given are :") 
    st.write(df)
    DF = pd.read_csv("x_test.csv")
    x = DF[['april_max_temp','april_min_temp','august_avg_temp','august_max_temp','days_above_90F','january_avg_temp','july_avg_temp','july_min_temp','june_avg_temp','september_min_temp']]
    y = DF['site_eui']
    model.fit(x,y)
    if st.button('Predict'):
        pred = model.predict(df)
        st.success("The site energy for given input data is {}:".format(pred))

if __name__ == '__main__':
    main()

