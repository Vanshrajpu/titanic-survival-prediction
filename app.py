import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic AI - Survival Predictor", page_icon="🚢", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;700;900&display=swap');
* { font-family: 'Outfit', sans-serif; }
.stApp { background: #070A12; }

/* HERO */
.hero {
  text-align: center;
  padding: 40px 20px 20px 20px;
}
@keyframes float {
  0% { transform: translateY(0px) rotate(-2deg); }
  50% { transform: translateY(-20px) rotate(2deg) scale(1.1); }
  100% { transform: translateY(0px) rotate(-2deg); }
}
@keyframes wave {
  0% { transform: translateX(-10%); }
  100% { transform: translateX(10%); }
}
.ship {
  font-size: 90px;
  animation: float 3s ease-in-out infinite;
  display: inline-block;
  filter: drop-shadow(0 0 30px #3B82F6);
}
.gradient-text {
  background: linear-gradient(90deg, #60A5FA, #A78BFA, #F472B6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  font-size: 55px;
  font-weight: 900;
  line-height: 1.1;
}
.hero-sub {
  color: #94A3B8;
  font-size: 18px;
  margin-top: 15px;
}

/* CARDS */
.glass-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02));
  backdrop-filter: blur(25px);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 24px;
  padding: 28px;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}
.badge {
  background: linear-gradient(135deg, #3B82F6, #8B5CF6);
  color: white;
  padding: 6px 14px;
  border-radius: 100px;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.5px;
  display: inline-block;
  margin-bottom: 8px;
}

/* STATS */
.stat-card {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
  padding: 18px;
  text-align: center;
}
.stat-num { font-size: 28px; font-weight: 900; color: #60A5FA; }
.stat-label { color: #94A3B8; font-size: 12px; }

/* BUTTON */
div[data-testid="stButton"] > button {
  background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
  border: none;
  border-radius: 100px;
  padding: 15px;
  font-weight: 800;
  font-size: 16px;
  letter-spacing: 0.5px;
  transition: 0.3s;
}
div[data-testid="stButton"] > button:hover {
  transform: scale(1.03);
  box-shadow: 0 10px 40px rgba(139,92,246,0.5);
}

/* INPUTS */
div[data-baseweb="select"] > div, input {
  background: rgba(0,0,0,0.4)!important;
  border-radius: 12px!important;
  border: 1px solid rgba(255,255,255,0.1)!important;
  color: white!important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'])
    for c in ['Embarked_C','Embarked_Q','Embarked_S']:
        if c not in df.columns: df[c]=0
    X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
    y = df['Survived']
    m = RandomForestClassifier(n_estimators=150, random_state=42)
    m.fit(X,y)
    return m

model = get_model()

# ===== HERO SECTION (Website jaisa) =====
st.markdown("""
<div class="hero">
  <div class="ship">🚢</div>
  <div class="gradient-text">
