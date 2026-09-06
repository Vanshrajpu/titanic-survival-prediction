import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PREMIUM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(14,165,233,.16), transparent 30%),
        radial-gradient(circle at 90% 85%, rgba(37,99,235,.15), transparent 30%),
        linear-gradient(135deg, #020617 0%, #071426 50%, #020617 100%);
    color: white;
}

/* Hide Streamlit default UI */
#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

.block-container {
    max-width: 920px;
    padding-top: 25px;
    padding-bottom: 20px;
}

/* ============================================================
   BACKGROUND ANIMATION
   ============================================================ */

.stApp::before {
    content: "";
    position: fixed;
    width: 450px;
    height: 450px;
    border-radius: 50%;
    background: rgba(14,165,233,.08);
    filter: blur(110px);
    top: -180px;
    left: -150px;
    animation: bgMove 9s ease-in-out infinite alternate;
    pointer-events: none;
}

.stApp::after {
    content: "";
    position: fixed;
    width: 400px;
    height: 400px;
    border-radius: 50%;
    background: rgba(59,130,246,.07);
    filter: blur(110px);
    right: -150px;
    bottom: -150px;
    animation: bgMove2 11s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes bgMove {
    from {
        transform: translate(0,0);
    }
    to {
        transform: translate(120px,100px);
    }
}

@keyframes bgMove2 {
    from {
        transform: translate(0,0);
    }
    to {
        transform: translate(-100px,-90px);
    }
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    text-align: center;
    padding: 10px 0 24px;
    animation: heroAppear .9s ease;
}

@keyframes heroAppear {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.ship {
    font-size: 72px;
    display: inline-block;
    filter: drop-shadow(0 0 25px rgba(56,189,248,.55));
    animation: shipFloat 3.5s ease-in-out infinite;
}

@keyframes shipFloat {
    0%,100% {
        transform: translateY(0) rotate(-2deg);
    }
    50% {
        transform: translateY(-14px) rotate(2deg);
    }
}

.badge {
    display: inline-block;
    margin-top: 7px;
    padding: 7px 16px;
    border-radius: 30px;
    background: rgba(14,165,233,.08);
    border: 1px solid rgba(56,189,248,.28);
    color: #7dd3fc;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 2px;
}

.title {
    font-size: 42px;
    font-weight: 800;
    margin: 13px 0 5px;
    letter-spacing: -1.5px;

    background: linear-gradient(
        90deg,
        #ffffff,
        #7dd3fc,
        #60a5fa,
        #ffffff
    );

    background-size: 250% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: titleGlow 5s linear infinite;
}

@keyframes titleGlow {
    to {
        background-position: 250% center;
    }
}

.subtitle {
    color: #94a3b8;
    font-size: 13px;
}

/* ============================================================
   INFORMATION CARDS
   ============================================================ */

.info-grid {
    display: grid;
    grid-template-columns: repeat(3,1fr);
    gap: 13px;
    margin: 5px 0 22px;
}

.info-card {
    position: relative;
    padding: 18px 15px;
    text-align: center;

    background: rgba(7,20,38,.68);
    border: 1px solid rgba(148,163,184,.12);
    border-radius: 17px;

    backdrop-filter: blur(18px);
    box-shadow:
        0 12px 35px rgba(0,0,0,.22),
        inset 0 1px 0 rgba(255,255,255,.04);

    transition: all .3s ease;
    animation: cardIn .7s ease both;
}

.info-card:nth-child(2) {
    animation-delay: .12s;
}

.info-card:nth-child(3) {
    animation-delay: .24s;
}

@keyframes cardIn {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.info-card:hover {
    transform: translateY(-5px);
    border-color: rgba(56,189,248,.28);
    box-shadow:
        0 18px 45px rgba(14,165,233,.10),
        0 0 25px rgba(56,189,248,.06);
}

.info-icon {
    font-size: 25px;
    margin-bottom: 7px;
}

.info-title {
    color: #94a3b8;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
}

.info-value {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 700;
    margin-top: 5px;
}

/* ============================================================
   FORM CARD
   ============================================================ */

.form-card {
    background: rgba(7,20,38,.76);
    border: 1px solid rgba(148,163,184,.13);
    border-radius: 22px;
    padding: 27px;

    backdrop-filter: blur(22px);

    box-shadow:
        0 25px 65px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.04);

    animation: formAppear .8s ease;
}

@keyframes formAppear {
    from {
        opacity: 0;
        transform: translateY(25px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.form-title {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 20px;
}

label {
    color: #cbd5e1 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
}

/* Inputs */

input {
    background: rgba(15,32,53,.92) !important;
    color: white !important;
    border: 1px solid rgba(148,163,184,.14) !important;
    border-radius: 11px !important;
    transition: all .25s ease !important;
}

input:hover {
    border-color: rgba(56,189,248,.35) !important;
}

input:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 18px rgba(56,189,248,.12) !important;
}

/* Select */

div[data-baseweb="select"] > div {
    background: rgba(15,32,53,.92) !important;
    color: white !important;
    border: 1px solid rgba(148,163,184,.14) !important;
    border-radius: 11px !important;
    transition: all .25s ease;
}

div[data-baseweb="select"] > div:hover {
    border-color: rgba(56,189,248,.35) !important;
}

/* ============================================================
   PREDICT BUTTON
   ============================================================ */

.stButton {
    margin-top: 15px;
}

.stButton > button {
    width: 100%;
    height: 53px;

    border: 1px solid rgba(125,211,252,.25);
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
        0 12px 30px rgba(37,99,235,.28),
        inset 0 1px 0 rgba(255,255,255,.18);

    transition: all .25s ease;
    overflow: hidden;
}

.stButton > button:hover {
    transform: translateY(-3px);
    box-shadow:
        0 17px 40px rgba(37,99,235,.40),
        0 0 25px rgba(56,189,248,.15);
}

.stButton > button:active {
    transform: scale(.98);
}

/* ============================================================
   RESULT
   ============================================================ */

.result {
    margin-top: 20px;
    padding: 25px;
    text-align: center;
    border-radius: 18px;

    animation:
        resultAppear .65s cubic-bezier(.17,.67,.32,1.3);
}

@keyframes resultAppear {
    0% {
        opacity: 0;
        transform: scale(.86) translateY(20px);
    }

    100% {
        opacity: 1;
        transform: scale(1) translateY(0);
    }
}

.success {
    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,.12),
            rgba(6,78,59,.18)
        );

    border: 1px solid rgba(52,211,153,.28);

    box-shadow:
        0 15px 50px rgba(16,185,129,.08);
}

.danger {
    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,.11),
            rgba(127,29,29,.18)
        );

    border: 1px solid rgba(248,113,113,.25);

    box-shadow:
        0 15px 50px rgba(239,68,68,.07);
}

.result-icon {
    font-size: 46px;
    animation: resultIcon 1.5s ease-in-out infinite;
}

@keyframes resultIcon {
    0%,100% {
        transform: translateY(0) scale(1);
    }

    50% {
        transform: translateY(-7px) scale(1.05);
    }
}

.result-title {
    font-size: 21px;
    font-weight: 800;
    margin-top: 7px;
}

.success .result-title {
    color: #6ee7b7;
}

.danger .result-title {
    color: #fca5a5;
}

.probability-label {
    margin-top: 17px;
    color: #94a3b8;
    font-size: 11px;
}

.probability-value {
    color: white;
    font-size: 28px;
    font-weight: 800;
    margin-top: 3px;
}

.progress-bg {
    width: 100%;
    height: 7px;
    background: rgba(148,163,184,.12);
    border-radius: 20px;
    overflow: hidden;
    margin-top: 11px;
}

.progress {
    height: 100%;
    border-radius: 20px;
    animation: progressAnimation 1.2s ease;
}

.success .progress {
    background: linear-gradient(90deg,#10b981,#34d399);
}

.danger .progress {
    background: linear-gradient(90deg,#ef4444,#fb7185);
}

@keyframes progressAnimation {
    from {
        width: 0;
    }
}

/* ============================================================
   FOOTER
   ============================================================ */

.custom-footer {
    text-align: center;
    margin-top: 25px;
    padding: 22px 10px 5px;

    border-top: 1px solid rgba(148,163,184,.10);

    animation: footerAppear 1s ease;
}

@keyframes footerAppear {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
    }
}

.footer-name {
    font-size: 15px;
    font-weight: 700;
    color: #e2e8f0;
}

.footer-role {
    margin-top: 5px;
    color: #64748b;
    font-size: 11px;
}

.footer-tech {
    margin-top: 9px;
    color: #38bdf8;
    font-size: 10px;
    letter-spacing: .8px;
}

/* ============================================================
   MOBILE
   ============================================================ */

@media (max-width: 650px) {

    .block-container {
        padding: 15px 13px 20px;
    }

    .title {
        font-size: 31px;
    }

    .ship {
        font-size: 58px;
    }

    .info-grid {
        grid-template-columns: 1fr;
        gap: 9px;
    }

    .info-card {
        padding: 14px;
    }

    .form-card {
        padding: 20px;
        border-radius: 18px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TITANIC DATASET
# ============================================================

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


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    random_state=42
)

model.fit(X, y)


# ============================================================
# HERO SECTION
# ============================================================

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


# ============================================================
# INFORMATION CARDS
# ============================================================

st.markdown("""
<div class="info-grid">

    <div class="info-card">
        <div class="info-icon">🤖</div>
        <div class="info-title">Model</div>
        <div class="info-value">Random Forest</div>
    </div>

    <div class="info-card">
        <div class="info-icon">📊</div>
        <div class="info-title">Dataset</div>
        <div class="info-value">Titanic Dataset</div>
    </div>

    <div class="info-card">
        <div class="info-icon">🧠</div>
        <div class="info-title">Features</div>
        <div class="info-value">8 Input Features</div>
    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT FORM
# ============================================================

st.markdown("""
<div class="form-card">

    <div class="form-title">
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


# ============================================================
# PREDICTION
# ============================================================

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

    # ========================================================
    # SURVIVED
    # ========================================================

    if prediction == 1:

        st.markdown(f"""
        <div class="result success">

            <div class="result-icon">
                🎉
            </div>

            <div class="result-title">
                Passenger Survived
            </div>

            <div class="probability-label">
                SURVIVAL PROBABILITY
            </div>

            <div class="probability-value">
                {probability:.1f}%
            </div>

            <div class="progress-bg">
                <div class="progress"
                     style="width:{probability}%;">
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.balloons()

    # ========================================================
    # DID NOT SURVIVE
    # ========================================================

    else:

        st.markdown(f"""
        <div class="result danger">

            <div class="result-icon">
                💔
            </div>

            <div class="result-title">
                Passenger Did Not Survive
            </div>

            <div class="probability-label">
                PREDICTION CONFIDENCE
            </div>

            <div class="probability-value">
                {probability:.1f}%
            </div>

            <div class="progress-bg">
                <div class="progress"
                     style="width:{probability}%;">
                </div>
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PROFESSIONAL FOOTER
# ============================================================

st.markdown("""
<div class="custom-footer">

    <div class="footer-name">
        👨‍💻 Vansh Rajput
    </div>

    <div class="footer-role">
        Machine Learning & AI Developer
    </div>

    <div class="footer-tech">
        PYTHON • SCIKIT-LEARN • STREAMLIT • MACHINE LEARNING
    </div>

</div>
""", unsafe_allow_html=True)
