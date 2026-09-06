import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# --- CSS for Exact Screenshot UI ---
st.markdown("""
<style>
.stApp { background-color: #0E1117; }
.card {
    background-color: #1E232F;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #2D3748;
}
.badge {
    background-color: #3B82F6;
    color: white;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 6px;
}
.result-red {
    background-color: #FF6B6B;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.result-green {
    background-color: #2ECC71;
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
}
.prob-card {
    background-color: #1E232F;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'])
    for col in ['Embarked_C','Embarked_Q','Embarked_S']:
        if col not in df.columns:
            df[col] = 0
    X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
    y = df['Survived']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_model()

# --- HEADER ---
st.markdown("<div style='text-align:center; font-size:70px;'>🚢</div>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center; margin:0;'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF;'>🤖 Machine Learning Powered Passenger Survival Predictor<br>Built by Vansh Rajput</p>", unsafe_allow_html=True)
st.write("")

# --- PASSENGER INFO CARD (Aapka Screenshot) ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Passenger Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="badge">🎫 Passenger Class</div>', unsafe_allow_html=True)
    pclass = st.selectbox("", [1,2,3], index=2, label_visibility="collapsed", key="pclass")

    st.markdown('<div class="badge" style="margin-top:15px;">🎂 Age</div>', unsafe_allow_html=True)
    age = st.number_input("", 0, 100, 25, label_visibility="collapsed", key="age")

    st.markdown('<div class="badge" style="margin-top:15px;">👥 Siblings / Spouse</div>', unsafe_allow_html=True)
    sibsp = st.number_input("", 0, 10, 0, label_visibility="collapsed", key="sibsp")

with col2:
    st.markdown('<div class="badge">👤 Gender</div>', unsafe_allow_html=True)
    gender = st.selectbox("", ["male","female"], label_visibility="collapsed", key="gender")

    st.markdown('<div class="badge" style="margin-top:15px;">💰 Fare</div>', unsafe_allow_html=True)
    fare = st.number_input("", 0.0, 600.0, 10.0, label_visibility="collapsed", key="fare")

    st.markdown('<div class="badge" style="margin-top:15px;">👶 Parents / Children</div>', unsafe_allow_html=True)
    parch = st.number_input("", 0, 10, 0, label_visibility="collapsed", key="parch")

st.markdown('</div>', unsafe_allow_html=True)
st.write("")

#
