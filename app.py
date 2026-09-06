import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #0c4a6e55, transparent 30%),
        radial-gradient(circle at 90% 90%, #1d4ed855, transparent 30%),
        linear-gradient(135deg, #020617, #071426, #020617);
    color: white;
}

/* Hide Streamlit default elements */

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Main width */

.block-container {
    max-width: 900px;
    padding-top: 25px;
    padding-bottom: 20px;
}

/* =========================================================
   HERO
   ========================================================= */

.hero {
    text-align: center;
    padding: 10px 0 25px;
    animation: fadeDown 1s ease;
}

@keyframes fadeDown {
    from {
        opacity: 0;
        transform: translateY(-25px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.ship {
    font-size: 70px;
    animation: floatShip 3s ease-in-out infinite;
    filter: drop-shadow(0 0 20px #38bdf855);
}

@keyframes floatShip {

    0%, 100% {
        transform: translateY(0) rotate(-2deg);
    }

    50% {
        transform: translateY(-15px) rotate(2deg);
    }

}

.badge {
    display: inline-block;
    padding: 7px 15px;
    margin-top: 5px;

    border: 1px solid #38bdf844;
    border-radius: 30px;

    background: #0ea5e91a;
    color: #7dd3fc;

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}

.title {
    font-size: 42px;
    font-weight: 800;

    margin-top: 13px;
    margin-bottom: 5px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #38bdf8,
        #60a5fa,
        #ffffff
    );

    background-size: 250% auto;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: titleMove 5s linear infinite;
}

@keyframes titleMove {

    to {
        background-position: 250% center;
    }

}

.subtitle {
    color: #94a3b8;
    font-size: 13px;
}

/* =========================================================
   INFO CARDS
   ========================================================= */

.cards {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 15px;

    margin-bottom: 22px;
}

.card {
    padding: 18px;
    text-align: center;

    background: #071426cc;
    border: 1px solid #94a3b81c;
    border-radius: 18px;

    box-shadow:
        0 15px 40px #00000033,
        inset 0 1px 0 #ffffff0a;

    backdrop-filter: blur(15px);

    transition: 0.3s ease;

    animation: cardUp 0.8s ease;
}

@keyframes cardUp {

    from {
        opacity: 0;
        transform: translateY(25px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

.card:hover {
    transform: translateY(-6px);

    border-color: #38bdf844;

    box-shadow:
        0 20px 45px #0284c733,
        0 0 20px #38bdf811;
}

.card-icon {
    font-size: 26px;
}

.card-title {
    color: #64748b;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;

    margin-top: 6px;
}

.card-value {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;

    margin-top: 5px;
}

/* =========================================================
   FORM
   ========================================================= */

.form-box {
    padding: 27px;

    background: #071426d9;

    border: 1px solid #94a3b81c;
    border-radius: 22px;

    box-shadow:
        0 25px 60px #00000044,
        inset 0 1px 0 #ffffff0a;

    backdrop-filter: blur(20px);

    animation: formShow 1s ease;
}

@keyframes formShow {

    from {
        opacity: 0;
        transform: translateY(30px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }

}

.form-heading {
    color: #f1f5f9;
    font-size: 18px;
    font-weight: 700;

    margin-bottom: 20px;
}

/* Input labels */

label {
    color: #cbd5e1 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

/* Number inputs */

input {
    background: #0f2035 !important;
    color: white !important;

    border: 1px solid #94a3b81f !important;
    border-radius: 11px !important;

    transition: 0.25s ease !important;
}

input:focus {
    border-color: #38bdf8 !important;

    box-shadow:
        0 0 18px #38bdf81c !important;
}

/* Selectbox */

div[data-baseweb="select"] > div {

    background: #0f2035 !important;

    border: 1px solid #94a3b81f !important;

    border-radius: 11px !important;

    color: white !important;

    transition: 0.25s ease;
}

div[data-baseweb="select"] > div:hover {

    border-color: #38bdf855 !important;

}

/* =========================================================
   BUTTON
   ========================================================= */

.stButton {
    margin-top: 15px;
}

.stButton > button {

    width: 100%;
    height: 52px;

    border: none;
    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            #0284c7,
            #2563eb
        );

    color: white;

    font-size: 14px;
    font-weight: 700;

    box-shadow:
        0 12px 30px #2563eb44;

    transition: all 0.3s ease;
}

.stButton > button:hover {

    transform: translateY(-3px);

    box-shadow:
        0 18px 40px #2563eb66,
        0 0 25px #38bdf822;
}

.stButton > button:active {

    transform: scale(0.98);

}

/* =========================================================
   RESULT
   ========================================================= */

.result {

    margin-top: 22px;
    padding: 25px;

    text-align: center;

    border-radius: 18px;

    animation:
        resultShow 0.6s cubic-bezier(.17,.67,.32,1.3);

}

@keyframes resultShow {

    from {
        opacity: 0;
        transform: scale(0.85) translateY(20px);
    }

    to {
        opacity: 1;
        transform: scale(1) translateY(0);
    }

}

.success {

    background:
        linear-gradient(
            135deg,
            #10b9811c,
            #064e3b33
        );

    border: 1px solid #34d39944;

    box-shadow:
        0 20px 50px #10b98114;
}

.danger {

    background:
        linear-gradient(
            135deg,
            #ef44441a,
            #7f1d1d33
        );

    border: 1px solid #f8717140;

    box-shadow:
        0 20px 50px #ef444414;
}

.result-icon {

    font-size: 48px;

    animation:
        resultFloat 1.5s ease-in-out infinite;
}

@keyframes resultFloat {

    0%, 100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }

}

.result-title {

    font-size: 22px;
    font-weight: 800;

    margin-top: 8px;
}

.success .result-title {
    color: #6ee7b7;
}

.danger .result-title {
    color: #fca5a5;
}

.result-label {

    color: #94a3b8;

    font-size: 10px;

    margin-top: 18px;

    letter-spacing: 1px;
}

.result-value {

    color: white;

    font-size: 29px;

    font-weight: 800;

    margin-top: 3px;
}

/* Progress */

.progress-background {

    width: 100%;
    height: 7px;

    background: #94a3b81c;

    border-radius: 20px;

    overflow: hidden;

    margin-top: 12px;
}

.progress {

    height: 100%;

    border-radius: 20px;

    animation: progress 1.2s ease;
}

.success .progress {

    background:
        linear-gradient(
            90deg,
            #10b981,
            #34d399
        );
}

.danger .progress {

    background:
        linear-gradient(
            90deg,
            #ef4444,
            #fb7185
        );
}

@keyframes progress {

    from {
        width: 0;
    }

}

/* =========================================================
   FOOTER
   ========================================================= */

.dev-footer {

    text-align: center;

    margin-top: 30px;
    padding-top: 20px;

    border-top: 1px solid #94a3b814;

    animation: footerFade 1.5s ease;
}

@keyframes footerFade {

    from {
        opacity: 0;
    }

    to {
        opacity: 1;
    }

}

.dev-name {

    color: #f1f5f9;

    font-size: 15px;

    font-weight: 700;
}

.dev-role {

    color: #64748b;

    font-size: 11px;

    margin-top: 5px;
}

.dev-stack {

    color: #38bdf8;

    font-size: 9px;

    letter-spacing: 1px;

    margin-top: 9px;
}

/* =========================================================
   MOBILE
   ========================================================= */

@media (max-width: 650px) {

    .block-container {
        padding: 15px 13px;
    }

    .title {
        font-size: 31px;
    }

    .ship {
        font-size: 58px;
    }

    .cards {
        grid-template-columns: 1fr;
    }

    .card {
        padding: 14px;
    }

    .form-box {
        padding: 20px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

df["Age"] = df["Age"].fillna(
    df["Age"].median()
)

df["Embarked"] = df["Embarked"].fillna("S")

df = pd.get_dummies(
    df,
    columns=["Embarked"],
    dtype=int
)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_Q",
    "Embarked_S"
]

X = df[features]
y = df["Survived"]


# =========================================================
# TRAIN MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X, y)


# =========================================================
# HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

    <div class="ship">🚢</div>

    <div class="badge">
        ✦ AI POWERED PREDICTION
    </div>

    <div class="title">
        Titanic Survival Predictor
    </div>

    <div class="subtitle">
        Machine Learning based passenger survival prediction
    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INFO CARDS
# =========================================================

st.markdown("""
<div class="cards">

    <div class="card">

        <div class="card-icon">
            🤖
        </div>

        <div class="card-title">
            Model
        </div>

        <div class="card-value">
            Random Forest
        </div>

    </div>


    <div class="card">

        <div class="card-icon">
            📊
        </div>

        <div class="card-title">
            Dataset
        </div>

        <div class="card-value">
            Titanic Dataset
        </div>

    </div>


    <div class="card">

        <div class="card-icon">
            🧠
        </div>

        <div class="card-title">
            Features
        </div>

        <div class="card-value">
            8 Input Features
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# PASSENGER FORM
# =========================================================

st.markdown("""
<div class="form-box">

    <div class="form-heading">
        👤 Passenger Information
    </div>

""", unsafe_allow_html=True)


col1, col2 = st.columns(2)


with col1:

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Siblings / Spouses",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


with col2:

    sex = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0
    )

    parch = st.number_input(
        "Parents / Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )


predict = st.button(
    "🔮  Predict Survival"
)

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if predict:

    sex_value = 1 if sex == "Female" else 0

    input_data = pd.DataFrame(
        [[
            pclass,
            sex_value,
            age,
            sibsp,
            parch,
            fare,
            0,
            1
        ]],
        columns=features
    )

    prediction = model.predict(input_data)[0]

    probability = (
        model.predict_proba(input_data)[0][prediction] * 100
    )

    # -----------------------------------------------------
    # SURVIVED
    # -----------------------------------------------------

    if prediction == 1:

        st.markdown(f"""
        <div class="result success">

            <div class="result-icon">
                🎉
            </div>

            <div class="result-title">
                Passenger Survived
            </div>

            <div class="result-label">
                SURVIVAL PROBABILITY
            </div>

            <div class="result-value">
                {probability:.1f}%
            </div>

            <div class="progress-background">

                <div
                    class="progress"
                    style="width:{probability}%;">
                </div>

            </div>

        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    # -----------------------------------------------------
    # DID NOT SURVIVE
    # -----------------------------------------------------

    else:

        st.markdown(f"""
        <div class="result danger">

            <div class="result-icon">
                💔
            </div>

            <div class="result-title">
                Passenger Did Not Survive
            </div>

            <div class="result-label">
                PREDICTION CONFIDENCE
            </div>

            <div class="result-value">
                {probability:.1f}%
            </div>

            <div class="progress-background">

                <div
                    class="progress"
                    style="width:{probability}%;">
                </div>

            </div>

        </div>
        """, unsafe_allow_html=True)


# =========================================================
# DEVELOPER FOOTER
# =========================================================

st.markdown("""
<div class="dev-footer">

    <div class="dev-name">
        👨‍💻 Vansh Rajput
    </div>

    <div class="dev-role">
        Machine Learning & AI Developer
    </div>

    <div class="dev-stack">
        PYTHON  •  SCIKIT-LEARN  •  STREAMLIT  •  MACHINE LEARNING
    </div>

</div>
""", unsafe_allow_html=True)
