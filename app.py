import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")

# --- PROFESSIONAL ANIMATION CSS ---
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #334155);
    background-size: 400% 400%;
    animation: bgMove 12s ease infinite;
}
@keyframes bgMove {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
@keyframes shipFloat {
    0%{transform: translateY(0px) rotate(-4deg);}
    50%{transform: translateY(-20px) rotate(4deg) scale(1.1);}
    100%{transform: translateY(0px) rotate(-4deg);}
}
.ship {
    font-size: 85px;
    text-align: center;
    animation: shipFloat 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 25px #38bdf8);
}
.card {
    background: rgba(30,41,59,0.85);
    backdrop-filter: blur(14px);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid rgba(56,189,248,0.2);
}
.badge {
    background: #38bdf8;
    color: black;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 800;
    display: inline-block;
    margin-bottom: 5px;
}
.result-green{background: linear-gradient(135deg,#10b981,#059669); padding:25px; border-radius:15px; text-align:center; color:white;}
.result-red{background: linear-gradient(135deg,#ef4444,#dc2626); padding:25px; border-radius:15px; text-align:center; color:white;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df["Sex"] = df["Sex"].map({"male":0,"female":1})
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("S")
    df = pd.get_dummies(df, columns=["Embarked"])
    if "Embarked_C" not in df.columns:
        df["Embarked_C"] = 0
    if "Embarked_Q" not in df.columns:
        df["Embarked_Q"] = 0
    if "Embarked_S" not in df.columns:
        df["Embarked_S"] = 0
    X = df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked_Q","Embarked_S"]]
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model

model = load_model()

st.markdown('<div class="ship">🚢</div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8;'>Professional AI App by Vansh Rajput</p>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Passenger Details")
c1,c2 = st.columns(2)
with c1:
    st.markdown('<div class="badge">CLASS</div>', unsafe_allow_html=True)
    pclass = st.selectbox("class",[1,2,3],index=2,label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">AGE</div>', unsafe_allow_html=True)
    age = st.number_input("age",0,100,25,label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">SIBLINGS</div>', unsafe_allow_html=True)
    sibsp = st.number_input("sibsp",0,10,0,label_visibility="collapsed")
with c2:
    st.markdown('<div class="badge">GENDER</div>', unsafe_allow_html=True)
    gender = st.selectbox("gender",["male","female"],label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">FARE</div>', unsafe_allow_html=True)
    fare = st.number_input("fare",0.0,600.0,32.0,label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">PARENTS</div>', unsafe_allow_html=True)
    parch = st.number_input("parch",0,10,0,label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")
if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    s = 1 if gender=="female" else 0
    data = np.array([[pclass,s,age,sibsp,parch,fare,0,1]])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]*100
    if pred==1:
        st.markdown(f'<div class="result-green"><h2>YOU SURVIVED! 🎉</h2><p>{prob:.1f}% chance</p></div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f'<div class="result-red"><h2>DID NOT SURVIVE 💔</h2><p>{100-prob:.1f}% risk</p></div>', unsafe_allow_html=True)
    st.progress(int(prob))
