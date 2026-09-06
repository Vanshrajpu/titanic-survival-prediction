import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0E1117, #1E293B, #0F172A, #1E1B4B);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes float-ship {
  0% { transform: translateY(0px) rotate(-3deg); }
  50% { transform: translateY(-18px) rotate(3deg) scale(1.1); }
  100% { transform: translateY(0px) rotate(-3deg); }
}
.ship-anim {
  font-size: 80px;
  text-align: center;
  animation: float-ship 2.8s ease-in-out infinite;
  filter: drop-shadow(0 0 30px #3B82F6);
}
@keyframes twinkle {
  0%,100% { opacity: 0.2; }
  50% { opacity: 0.8; }
}
.card {
    background: rgba(30, 35, 47, 0.9);
    backdrop-filter: blur(12px);
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(59,130,246,0.2);
}
.badge {
    background: linear-gradient(135deg, #3B82F6, #60A5FA);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 6px;
}
.result-red { background: linear-gradient(135deg,#FF6B6B,#EF4444); padding: 25px; border-radius: 15px; text-align: center; color: white; }
.result-green { background: linear-gradient(135deg,#10B981,#059669); padding: 25px; border-radius: 15px; text-align: center; color: white; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df = pd.get_dummies(df, columns=["Embarked"])
    for col in ["Embarked_C", "Embarked_Q", "Embarked_S"]:
        if col not in df.columns:
