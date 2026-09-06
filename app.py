import streamlit as st
import pickle
import pandas as pd

# Page config
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="wide")

# CSS - aapka wahi glass wala design
st.markdown("""
<style>
.glass-card {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

# Model load
@st.cache_resource
def load_model():
    return pickle.load(open('titanic_model.pkl', 'rb'))

model = load_model()

st.title("🚢 Titanic Survival Prediction")

# --- INPUTS - Aapke wahi inputs ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    pclass = st.selectbox("🎫 Passenger Class", [1,2,3], index=2)
with col2:
    sex = st.selectbox("👤 Gender", ["male", "female"])
with col3:
    age = st.number_input("🎂 Age", value=25)
with col4:
    fare = st.number_input("💰 Fare", value=10.0)

col5, col6, col7 = st.columns(3)
with col5:
    sibsp = st.number_input("👨‍👩‍👧‍👦 Siblings / Spouses", value=0)
with col6:
    parch = st.number_input("👨‍👩‍👧 Parents / Children", value=0)
with col7:
    embarked = st.selectbox("📍 Embarked", ["S","C","Q"])

# --- PREDICT BUTTON ---
if st.button("🔮 Predict Survival", use_container_width=True):
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])

    # Yaha aapka encoding logic ayega
    # input_data['Sex'] =...

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ SURVIVED - Probability: {prob:.2f}")
        st.balloons()
    else:
        st.markdown(f"""
        <div style='background:#ff6b6b; color:white; padding:20px; border-radius:15px; text-align:center'>
            <h2>❌ DID NOT SURVIVE</h2>
            <p>The model predicts that this passenger did not survive. (Prob: {prob:.2f})</p>
        </div>
        """, unsafe_allow_html=True)
