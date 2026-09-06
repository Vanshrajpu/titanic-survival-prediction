import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

# CSS - Fixed, no triple quote issue
st.markdown(
    """
<style>
.stApp { background-color: #0E1117; }

@keyframes float-ship {
  0% { transform: translateY(0px) rotate(-3deg); }
  50% { transform: translateY(-15px) rotate(3deg) scale(1.08); }
  100% { transform: translateY(0px) rotate(-3deg); }
}
.ship-anim {
  font-size: 75px;
  text-align: center;
  animation: float-ship 2.5s ease-in-out infinite;
  filter: drop-shadow(0 0 20px #3B82F6);
}

.card {
    background-color: #1E232F;
    padding: 25px;
    border-radius: 18px;
    border: 1px solid #2D3748;
}
.badge {
    background-color: #3B82F6;
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
""",
    unsafe_allow_html=True,
)

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
            df[col] = 0
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked_Q", "Embarked_S"]]
    y = df["Survived"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = get_model()

# Header
st.markdown('<div class="ship-anim">🚢</div>', unsafe_allow_html=True)
st.markdown("<h1 style='text-align:center;'>Titanic Survival Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF;'>Machine Learning Powered Predictor<br>Built by Vansh Rajput</p>", unsafe_allow_html=True)

# Card
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Passenger Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="badge">🎫 Passenger Class</div>', unsafe_allow_html=True)
    pclass = st.selectbox("pclass", [1, 2, 3], index=2, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">🎂 Age</div>', unsafe_allow_html=True)
    age = st.number_input("age", 0, 100, 25, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">👥 Siblings / Spouse</div>', unsafe_allow_html=True)
    sibsp = st.number_input("sibsp", 0, 10, 0, label_visibility="collapsed")

with col2:
    st.markdown('<div class="badge">👤 Gender</div>', unsafe_allow_html=True)
    gender = st.selectbox("gender", ["male", "female"], label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">💰 Fare</div>', unsafe_allow_html=True)
    fare = st.number_input("fare", 0.0, 600.0, 10.0, label_visibility="collapsed")
    st.markdown('<div class="badge" style="margin-top:12px;">👶 Parents / Children</div>', unsafe_allow_html=True)
    parch = st.number_input("parch", 0, 10, 0, label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)
st.write("")

if st.button("🔮 Predict Survival", use_container_width=True, type="primary"):
    sex_val = 1 if gender == "female" else 0
    data = np.array([[pclass, sex_val, age, sibsp, parch, fare, 0, 1]])
    pred = model.predict(data)[0]
    proba = model.predict_proba(data)[0]
    survived = proba[1] * 100
    not_survived = proba[0] * 100

    st.markdown("#### ✨ Prediction Result")
    if pred == 1:
        st.markdown(f'<div class="result-green"><div style="font-size:45px;">✓</div><h2>SURVIVED</h2><p>{survived:.1f}% chance</p></div>', unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f'<div class="result-red"><div style="font-size:45px;">X</div><h2>DID NOT SURVIVE</h2><p>{not_survived:.1f}% risk</p></div>', unsafe_allow_html=True)

    st.write(f"Survived {survived:.1f}%")
    st.progress(int(survived))
