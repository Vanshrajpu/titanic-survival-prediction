import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    # Model training inside app - no pkl needed
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    df['Sex'] = df['Sex'].map({'male':0, 'female':1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna('S', inplace=True)
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
    if 'Embarked_C' not in df.columns:
        df['Embarked_C'] = 0
    X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']]
    y = df['Survived']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

model = load_model()

st.title("🚢 Titanic Survival Prediction")
st.write("Passenger details daal ke predict kijiye")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1,2,3], index=2)
    sex = st.selectbox("👤 Gender", ["male", "female"])
    age = st.slider("🎂 Age", 0, 100, 25)
with col2:
    sibsp = st.number_input("Siblings / Spouses", 0, 10, 0)
    parch = st.number_input("Parents / Children", 0, 10, 0)
    fare = st.number_input("💰 Fare", 0.0, 600.0, 32.0)
    embarked = st.selectbox("📍 Embarked", ["S","C","Q"])

if st.button("🔮 Predict Survival", use_container_width=True):
    sex_val = 1 if sex == "female" else 0
    embarked_Q = 1 if embarked == "Q" else 0
    embarked_S = 1 if embarked == "S" else 0
    import numpy as np
    input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_Q, embarked_S]])
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.success(f"✅ SURVIVED - {prob*100:.1f}% chance")
        st.balloons()
    else:
        st.error(f"❌ DID NOT SURVIVE - {prob*100:.1f}% chance")
