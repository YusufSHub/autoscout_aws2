import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("/Users/emine/Desktop/autoscout_project/autoscout.pkl", "rb"))
enc = pickle.load(open("autoscout_encoder.pkl", "rb"))
df = pd.read_csv("/Users/emine/Desktop/autoscout_project/final_model.csv")

st.markdown("# <center>Car Price Prediction For AutoScout</center>",unsafe_allow_html=True )
col1, col2= st.columns(2)
user_make_model = st.slider(col1.selectbox("Select your car's Make&Model", df.make_model.unique()))
user_body_type = col2.selectbox("Select your car's Body Type", df.body_type.unique())
user_gear = col1.selectbox("Select your car's Gearing Type", df["Gearing Type"].unique())
user_fuel = col2.selectbox("Select your car's Fuel Type", df.Fuel.unique())
user_km = col1.number_input("KM", 0, 300000,step=10000)
user_age = int(col2.selectbox("Age", (0,1,2,3)))
user_cc = col1.number_input("Displacement (cc)", 900,2967,1200,100)
user_hp = col2.number_input("HP", 55,390,90,10)

option1 = st.slider("Select a number", min_value=5, max_value=70, value=30, step=5)
option2 = st.slider("Select a number", min_value=0.2, max_value=30.2, value=5.2, step=0.2)
result=option1*option2
st.write("Multiplication of two option is: ",result)

#mak baslik cent ort calismasinicin de unsafe kisim 

#iki colum yapti
# select box col 1 iicndeki yazi virgulden sonra de neler sececegi df make modelden uniguleri al
#col2 icinde benzeri
# ilk 4 kategorik kullanici select boxdan sececek
#km number input artis 10000
# age num input olabilirdi o yuzden select box hata vermesn diye int yaptik
# motor hacmi icinde number input kullanildu veri setinden bakildi bunlara describeden 
#     

#bak brda borwserea gectik adam isteklerini secti onlari dataframe yapiyoruz biz

car = pd.DataFrame({"make_model" : [user_make_model],
                    "body_type" : [user_body_type],
                    "km" : [user_km],
                    "hp" : [user_hp],
                    "Gearing Type" : [user_gear],
                    "Displacement_cc" : [user_cc],
                    "Fuel": [user_fuel],
                    "Age" : [user_age]})

cat = car.select_dtypes("object").columns
car[cat] = enc.transform(car[cat])# burda encoder kullandik bu katregorik verileri num veirye ceviyor
#bizim olusturdugumuz en codeera gore yapiyor
#burda transform cikartigi kaliba gore degistiriyor
#

c1, c2, c3, c4, c5,c6,c7,c8,c9 = st.columns(9) #dugme ortadan olsun diye cler
if c5.button('Predict'):
    result = model.predict(car)[0]# kullanici degerlerine gore prediction yapcak
    st.info(f"Predicted value of your car : {round(result)}$")
    #
