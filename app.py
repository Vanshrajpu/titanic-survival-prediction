import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #1e3a8a);
    background-size: 400% 400%;
    animation: bgMove 10s ease infinite;
}
@keyframes bgMove {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
@keyframes shipFloat {
    0%{transform: translateY(0px) rotate(-3deg);}
    50%{transform: translateY(-22px) rotate(3deg) scale(1.08);}
    100%{transform: translateY(0px) rotate(-3deg);}
}
.ship {
    font-size: 90px;
    text-align: center;
    animation: shipFloat 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 30px #38bdf8);
}
.card {
    background: rgba(30,41,59,0.92);
    backdrop-filter: blur(15px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(56,189,248,0.25);
}
.badge {
    background: #38bdf8;
    color: black;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
    display: inline-block;
    margin-bottom: 6px;
}
.result-green{background: linear-gradient(135deg,#10b981,#059669); padding:25px; border-radius:16px; text-align:center; color:white;}
.result-red{background: linear-gradient(135deg,#ef4444,#991b1b); padding:25px; border-radius:16px; text-align:center; color:white;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare"]]
    y = df["Survived"]
    model = RandomForest
