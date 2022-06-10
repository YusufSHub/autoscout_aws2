import streamlit as st
import pandas as pd
import numpy as np


#Project Example

df = pd.read_csv("final_scout_not_dummy.csv")
df_new = df[["make_model","hp_kW","km","Fuel","age","Gearing_Type","price"]]
df_new=pd.get_dummies(df_new)
X = df_new.drop("price", axis =1)



import pickle
filename = 'xgb_model'
model = pickle.load(open(filename, 'rb'))


KM = st.sidebar.number_input("KM:",min_value=0, max_value=500000)
Age = st.sidebar.number_input("Age:",min_value=0, max_value=50)
Hp = st.sidebar.number_input("Hp:",min_value=0, max_value=1000)
Model = st.sidebar.selectbox("Make Model", ('Audi A1','Audi A2', 'Audi A3', 'Opel Astra', "Opel Corsa","Opel Insignia","Renault Clio","Renault Duster","Renault Espace"))
Fuel = st.sidebar.selectbox("Fuel", ('Benzine','Diesel', 'Electric', 'LPG/CNG'))
Gears = st.sidebar.selectbox("Gear", ('Automatic','Manual', 'Semi-automatic'))



my_dict = {
    "km": KM,
    "age": Age,
    "make_model": Model,
    "hp_kW": Hp,
    "Fuel": Fuel,
    "Gearing_Type": Gears,
}

#image
#from PIL import Image
#img = Image.open("images.png")
#st.image(img,width=40,caption="cattie") 




html_temp = """
<div style="background-color:black;padding:1.5px">
<h1 style="color:white;text-align:center;">Guess the price of your car!! </h1>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

html_temp = """
<div style="background-white:black;padding: 15px 1px 1px 1px">
<h6 style="color:red;text-align:center;">Select your vehicle's information from the left menu.</h6>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)

html_temp = """
<div style="background-white:black;padding: 15px 1px 1px 1px">
<h6 style="color:black;text-align:center;">Your choices are here.!! </h6>
</div><br>"""
st.markdown(html_temp,unsafe_allow_html=True)
df=pd.DataFrame.from_dict([my_dict])
st.table(df)

_, _, _,  col, _, _, _ = st.columns([1]*6+[0.30])




if col.button("Predict"): 
    

    my_dict = pd.get_dummies(df) 
    my_dict = my_dict.reindex(columns = X.columns, fill_value=0) # yeni sunacağımız veriyi modeldeki sütun düzenine göre ayarlıyoruz
    pred = model.predict(my_dict)
    st.success('You can post your car at this price:{} $'.format(pred[0].round(0)))

