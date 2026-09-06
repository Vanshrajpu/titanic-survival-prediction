import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic AI Predictor", page_icon="🚢", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700;800&display=swap');
* {font-family: 'Poppins', sans-serif;}

.stApp {
    background: radial-gradient(circle at 20% 30%, #1e3a8a 0%, #0f172a 40%, #020617 100%);
    overflow: hidden;
}
.stApp::before {
    content: "✦ ✦ ✦ ✦ ✦ ✦";
    position: fixed;
    top: 10%;
    left: 0;
    width: 200%;
    font-size: 18px;
    letter-spacing: 50px;
    color: rgba(56,189,248,0.15);
    animation: starsMove 60s linear infinite;
    pointer-events: none;
}
@keyframes starsMove {
    0%{transform: translateX(0);}
    100%{transform: translateX(-50%);}
}
@keyframes bgPulse {
    0%,100%{background-size: 100% 100%;}
    50%{background-size: 120% 120%;}
}
@keyframes floatShip {
    0%{transform: translateY(0px) rotate(-4deg) scale(1);}
    25%{transform: translateY(-18px) rotate(2deg) scale(1.08);}
    50%{transform: translateY(-28px) rotate(4deg) scale(1.12);}
    75%{transform: translateY(-12px) rotate(-2deg) scale(1.05);}
    100%{transform: translateY(0px) rotate(-4deg) scale(1);}
}
@keyframes wave {
    0%{transform: translateX(-50%) translateY(0) rotate(0deg);}
    50%{transform: translateX(-50%) translateY(-10px) rotate(1deg);}
    100%{transform: translateX(-50%) translateY(0) rotate(0deg);}
}
.ship-box {
    position: relative;
    text-align: center;
    margin-top: 20px;
}
.ship {
    font-size: 95px;
    animation: floatShip 3.5s ease-in-out infinite;
    display: inline-block;
    filter: drop-shadow(0 0 25px #38bdf8) drop-shadow(0 0 50px #0ea5e9);
}
.wave {
    width: 180px;
    height: 20px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
    border-radius: 50%;
    margin: -15px auto 10px auto;
    opacity: 0.6;
    animation: wave 2.5s ease-in-out infinite;
    filter: blur(1px);
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    background: linear-gradient(90deg, #fff, #38bdf8, #fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% 100%;
    animation: shine 3s linear infinite;
}
@keyframes shine {
    0%{background-position: -200% 0;}
    100%{background-position: 200% 0;}
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 14px;
    letter-spacing: 1px;
    margin-bottom: 25px;
}
.card {
    background: rgba(15,23,42,0.75);
    backdrop-filter: blur(18px);
    padding: 28px;
    border-radius: 24px;
    border: 1px solid rgba(56,189,248,0.2);
    box-shadow: 0 0 0 1px rgba(255,255,255,0.05) inset, 0 20px 60px rgba(0,0,0,0.5), 0 0 40px rgba(56,189,248,0.1);
    animation: cardEntry 0.8s ease-out;
}
@keyframes cardEntry {
    from{opacity:0; transform: translateY(30px) scale(0.95);}
    to{opacity:1; transform: translateY(0) scale(1);}
}
.badge {
    background: linear-gradient(90deg, #38bdf8, #60a5fa);
    color: #020617;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.5px;
    display: inline-block;
    margin-bottom: 8px;
    box-shadow: 0 2px 10px rgba(56,189,248,0.4);
}
.stSelectbox > div > div,.stNumberInput > div > div > input {
    background: rgba(30,41,59,0.8)!important;
    border: 1px solid rgba(56,189,248,0.2)!important;
    border-radius: 12px!important;
    color: white!important;
}
.result-green{
    background: linear-gradient(135deg,#10b981,#059669);
    padding:28px;
    border-radius:18px;
    text-align:center;
    color:white;
    animation: pop 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 0 30px rgba(16,185,129,0.6);
}
.result-red{
    background: linear-gradient(135deg,#ef4444,#7f1d1d);
    padding:28px;
    border-radius:18px;
    text-align:center;
    color:white;
    animation: pop 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 0 30px rgba(239,68,68,0.6);
}
@keyframes pop {
    0%{transform: scale(0.5); opacity:0;}
    100%{transform: scale(1); opacity:1;}
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

st.markdown('<div class="ship-box"><div class="ship">🚢</div><div class="wave"></div></div>', unsafe_allow_html=True)
st.markdown('<div class="title">TITANIC AI VOYAGE</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">✨ Will You Survive The Legendary Journey? ✨<br>Built with ❤️ by Vansh Rajput</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 🧭 Passenger Details")

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="badge">🎫 CLASS</div>', unsafe_allow_html=True)
    pclass = st.selectbox("pclass", [1, 2, 3], index=2, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:14px;">🎂 AGE</div>', unsafe_allow_html=True)
    age = st.number_input("age", 0, 100, 25, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:14px;">👥 SIBLINGS</div>', unsafe_allow_html=True)
    sibsp = st.number_input("sibsp", 0, 10, 0, label_visibility="collapsed")
with c2:
    st.markdown('<div class="badge">👤 GENDER</div>', unsafe_allow_html=True)
    gender = st.selectbox("gender", ["male", "female"], label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:14px;">💰 FARE ($)</div>', unsafe_allow_html=True)
    fare = st.number_input("fare", 0.0, 600.0, 32.0, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:14px;">👶 PARENTS</div>', unsafe_allow_html=True)
    parch = st.number_input("parch", 0, 10, 0, label_visibility="collapsed")

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
if st.button("🔮 PREDICT MY FATE", use_container_width=True, type="primary"):
    sex_val = 1 if gender == "female" else 0
    data = np.array([[pclass, sex_val, age, sibsp, parch, fare]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    if pred == 1:
        st.markdown(f'<div class="result-green"><div style="font-size:50px;">🏆</div><h2>YOU SURVIVED!</h2><p>Captain saved you! {prob:.1f}% survival chance</p><p style="font-size:12px; opacity:0.9;">Jack would be proud!</p></div>', unsafe_allow_html=True)
        st.balloons()
        st.snow()
    else:
        st.markdown(f'<div class="result-red"><div style="font-size:50px;">🌊</div><h2>LOST AT SEA</h2><p>{100-prob:.1f}% risk - Ocean was cruel</p><p style="font-size:12px; opacity:0.9;">But your story lives on...</p></div>', unsafe_allow_html=True)

    st.progress(int(prob))
