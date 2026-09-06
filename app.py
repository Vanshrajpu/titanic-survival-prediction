import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Titanic AI Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown("""
<style>

/* ================= BACKGROUND ================= */

.stApp {
    background:
        radial-gradient(circle at 10% 20%, rgba(14,165,233,.15), transparent 25%),
        radial-gradient(circle at 90% 80%, rgba(6,182,212,.12), transparent 25%),
        linear-gradient(135deg,#020617,#0f172a,#082f49,#020617);

    background-size: 200% 200%;
    animation: backgroundMove 15s ease infinite;
}

@keyframes backgroundMove {
    0%   {background-position:0% 50%;}
    50%  {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

/* ================= HIDE STREAMLIT ================= */

#MainMenu {
    visibility:hidden;
}

footer {
    visibility:hidden;
}

header {
    visibility:hidden;
}

/* ================= MAIN CONTAINER ================= */

.block-container {
    max-width: 900px;
    padding-top: 30px;
}

/* ================= SHIP ================= */

.ship {
    font-size:95px;
    text-align:center;

    animation:
        shipFloat 3s ease-in-out infinite,
        shipGlow 2s ease-in-out infinite alternate;

    margin-bottom:-10px;
}

@keyframes shipFloat {

    0% {
        transform:translateY(0px) rotate(-3deg);
    }

    50% {
        transform:translateY(-18px) rotate(3deg) scale(1.08);
    }

    100% {
        transform:translateY(0px) rotate(-3deg);
    }
}

@keyframes shipGlow {

    from {
        filter:drop-shadow(0 0 8px #38bdf8);
    }

    to {
        filter:drop-shadow(0 0 35px #0ea5e9);
    }
}

/* ================= TITLE ================= */

.title {

    text-align:center;

    font-size:48px;

    font-weight:900;

    background:
        linear-gradient(
            90deg,
            #38bdf8,
            #22d3ee,
            #ffffff,
            #38bdf8
        );

    background-size:300%;

    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;

    animation:titleMove 5s linear infinite;

    letter-spacing:1px;
}

@keyframes titleMove {

    0% {
        background-position:0%;
    }

    100% {
        background-position:300%;
    }
}

.subtitle {

    text-align:center;

    color:#94a3b8;

    font-size:17px;

    margin-bottom:30px;
}

/* ================= AI BADGE ================= */

.ai-badge {

    width:max-content;

    margin:auto;

    padding:7px 18px;

    border-radius:30px;

    background:rgba(14,165,233,.12);

    border:1px solid rgba(56,189,248,.4);

    color:#7dd3fc;

    font-size:13px;

    font-weight:bold;

    box-shadow:0 0 20px rgba(14,165,233,.15);
}

/* ================= CARD ================= */

.card {

    background:
        linear-gradient(
            135deg,
            rgba(15,23,42,.90),
            rgba(30,41,59,.75)
        );

    backdrop-filter:blur(20px);

    border:1px solid rgba(148,163,184,.15);

    border-radius:28px;

    padding:30px;

    box-shadow:
        0 25px 70px rgba(0,0,0,.45),
        inset 0 1px 0 rgba(255,255,255,.05);

    margin-top:25px;
}

/* ================= SECTION TITLE ================= */

.section-title {

    font-size:24px;

    font-weight:800;

    color:#f8fafc;

    margin-bottom:20px;
}

/* ================= LABEL ================= */

.custom-label {

    color:#7dd3fc;

    font-size:12px;

    font-weight:800;

    letter-spacing:1px;

    margin-bottom:6px;
}

/* ================= INPUTS ================= */

.stSelectbox > div > div,
.stNumberInput > div > div {

    background:rgba(15,23,42,.75) !important;

    border:1px solid rgba(56,189,248,.20) !important;

    border-radius:13px !important;

    color:white !important;
}

/* ================= BUTTON ================= */

.stButton > button {

    height:58px;

    border-radius:17px;

    border:none;

    font-size:18px;

    font-weight:800;

    color:white;

    background:
        linear-gradient(
            90deg,
            #0284c7,
            #06b6d4,
            #0284c7
        );

    background-size:200%;

    box-shadow:
        0 8px 30px rgba(14,165,233,.30);

    transition:.3s;

}

.stButton > button:hover {

    transform:translateY(-3px) scale(1.01);

    background-position:100%;

    box-shadow:
        0 12px 40px rgba(14,165,233,.55);

}

/* ================= RESULT ================= */

.result {

    margin-top:25px;

    padding:30px;

    border-radius:24px;

    text-align:center;

    animation:resultAppear .6s ease;

}

@keyframes resultAppear {

    from {
        opacity:0;
        transform:translateY(25px) scale(.95);
    }

    to {
        opacity:1;
        transform:translateY(0) scale(1);
    }
}

.survived {

    background:
        linear-gradient(
            135deg,
            rgba(16,185,129,.95),
            rgba(5,150,105,.85)
        );

    box-shadow:
        0 15px 50px rgba(16,185,129,.25);
}

.not-survived {

    background:
        linear-gradient(
            135deg,
            rgba(239,68,68,.95),
            rgba(185,28,28,.85)
        );

    box-shadow:
        0 15px 50px rgba(239,68,68,.25);
}

.result-icon {

    font-size:65px;

    animation:iconPop .7s ease;
}

@keyframes iconPop {

    0% {
        transform:scale(.3) rotate(-20deg);
    }

    70% {
        transform:scale(1.2) rotate(5deg);
    }

    100% {
        transform:scale(1);
    }
}

.result h2 {

    font-size:30px;

    margin:10px 0;

    color:white;
}

.probability {

    font-size:22px;

    font-weight:bold;

    color:white;
}

/* ================= INFO CARDS ================= */

.info-box {

    background:rgba(15,23,42,.55);

    border:1px solid rgba(56,189,248,.12);

    border-radius:18px;

    padding:18px;

    text-align:center;

    transition:.3s;
}

.info-box:hover {

    transform:translateY(-5px);

    border-color:rgba(56,189,248,.5);

    box-shadow:0 10px 30px rgba(14,165,233,.12);
}

.info-number {

    font-size:25px;

    font-weight:900;

    color:#38bdf8;
}

.info-text {

    color:#94a3b8;

    font-size:12px;

}

/* ================= DIVIDER ================= */

.divider {

    height:1px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(56,189,248,.4),
            transparent
        );

    margin:30px 0;
}

/* ================= FOOTER ================= */

.footer {

    text-align:center;

    color:#64748b;

    font-size:13px;

    margin-top:35px;

    padding-bottom:20px;
}

.footer span {

    color:#38bdf8;

    font-weight:bold;
}

/* ================= WAVES ================= */

.wave {

    position:fixed;

    bottom:0;

    left:0;

    width:100%;

    height:100px;

    opacity:.07;

    background:
        radial-gradient(
            ellipse at center,
            #38bdf8 0%,
            transparent 70%
        );

    animation:waveMove 5s ease-in-out infinite;

    pointer-events:none;
}

@keyframes waveMove {

    0%,100% {
        transform:translateX(-30px);
    }

    50% {
        transform:translateX(30px);
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    df = pd.read_csv(url)

    # Gender
    df["Sex"] = df["Sex"].map({
        "male": 0,
        "female": 1
    })

    # Age
    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    # Embarked
    df["Embarked"] = df["Embarked"].fillna("S")

    df = pd.get_dummies(
        df,
        columns=["Embarked"]
    )

    # Make sure columns exist
    for col in [
        "Embarked_C",
        "Embarked_Q",
        "Embarked_S"
    ]:

        if col not in df.columns:
            df[col] = 0

    # Features
    X = df[
        [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_Q",
            "Embarked_S"
        ]
    ]

    y = df["Survived"]

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X, y)

    return model


model = load_model()


# =========================================================
# HERO
# =========================================================

st.markdown(
    '<div class="wave"></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ai-badge">🤖 POWERED BY MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="ship">🚢</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="title">Titanic AI Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="subtitle">
    Predict passenger survival using a Random Forest Machine Learning model
    </div>
    ''',
    unsafe_allow_html=True
)


# =========================================================
# INFO CARDS
# =========================================================

a, b, c = st.columns(3)

with a:
    st.markdown(
        '''
        <div class="info-box">
            <div class="info-number">🚢</div>
            <div class="info-text">TITANIC DATASET</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with b:
    st.markdown(
        '''
        <div class="info-box">
            <div class="info-number">🌲</div>
            <div class="info-text">RANDOM FOREST</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

with c:
    st.markdown(
        '''
        <div class="info-box">
            <div class="info-number">🎯</div>
            <div class="info-text">SURVIVAL PREDICTION</div>
        </div>
        ''',
        unsafe_allow_html=True
    )


# =========================================================
# INPUT CARD
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">👤 Passenger Information</div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)


# =========================================================
# LEFT COLUMN
# =========================================================

with c1:

    st.markdown(
        '<div class="custom-label">🎫 PASSENGER CLASS</div>',
        unsafe_allow_html=True
    )

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        index=2,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="custom-label">🎂 AGE</div>',
        unsafe_allow_html=True
    )

    age = st.number_input(
        "Age",
        min_value=0,
        max_value=100,
        value=25,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="custom-label">👨‍👩‍👧 SIBLINGS / SPOUSES</div>',
        unsafe_allow_html=True
    )

    sibsp = st.number_input(
        "SibSp",
        min_value=0,
        max_value=10,
        value=0,
        label_visibility="collapsed"
    )


# =========================================================
# RIGHT COLUMN
# =========================================================

with c2:

    st.markdown(
        '<div class="custom-label">⚧ GENDER</div>',
        unsafe_allow_html=True
    )

    gender = st.selectbox(
        "Gender",
        ["male", "female"],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="custom-label">💰 TICKET FARE</div>',
        unsafe_allow_html=True
    )

    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=32.0,
        step=1.0,
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="custom-label">👨‍👩‍👧 PARENTS / CHILDREN</div>',
        unsafe_allow_html=True
    )

    parch = st.number_input(
        "Parch",
        min_value=0,
        max_value=10,
        value=0,
        label_visibility="collapsed"
    )


st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PREDICT BUTTON
# =========================================================

st.write("")

if st.button(
    "🔮  PREDICT SURVIVAL",
    use_container_width=True,
    type="primary"
):

    # Gender encoding
    s = 1 if gender == "female" else 0

    # Input data
    data = np.array([
        [
            pclass,
            s,
            age,
            sibsp,
            parch,
            fare,
            0,
            1
        ]
    ])

    # Prediction
    pred = model.predict(data)[0]

    # Probability
    prob = model.predict_proba(data)[0][1] * 100


    # =====================================================
    # SURVIVED
    # =====================================================

    if pred == 1:

        st.markdown(
            f'''
            <div class="result survived">

                <div class="result-icon">
                    🎉
                </div>

                <h2>
                    YOU SURVIVED!
                </h2>

                <div class="probability">
                    Survival Probability: {prob:.1f}%
                </div>

                <p>
                    🌊 The model predicts a high chance of survival.
                </p>

            </div>
            ''',
            unsafe_allow_html=True
        )

        st.balloons()

    # =====================================================
    # DID NOT SURVIVE
    # =====================================================

    else:

        st.markdown(
            f'''
            <div class="result not-survived">

                <div class="result-icon">
                    💔
                </div>

                <h2>
                    DID NOT SURVIVE
                </h2>

                <div class="probability">
                    Survival Probability: {prob:.1f}%
                </div>

                <p>
                    ⚠️ The model predicts a higher risk of non-survival.
                </p>

            </div>
            ''',
            unsafe_allow_html=True
        )


    # =====================================================
    # PROBABILITY
    # =====================================================

    st.write("")

    st.markdown(
        "<p style='text-align:center;color:#94a3b8;'>"
        "AI Confidence Level"
        "</p>",
        unsafe_allow_html=True
    )

    st.progress(
        int(prob)
    )


    # =====================================================
    # RESULT STATS
    # =====================================================

    st.write("")

    x, y = st.columns(2)

    with x:

        st.markdown(
            f'''
            <div class="info-box">

                <div class="info-number">
                    {prob:.1f}%
                </div>

                <div class="info-text">
                    SURVIVAL CHANCE
                </div>

            </div>
            ''',
            unsafe_allow_html=True
        )

    with y:

        st.markdown(
            f'''
            <div class="info-box">

                <div class="info-number">
                    {100-prob:.1f}%
                </div>

                <div class="info-text">
                    NON-SURVIVAL RISK
                </div>

            </div>
            ''',
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    '''
    <div class="footer">

        🚢 Titanic AI Predictor

        <br><br>

        Built with ❤️ using
        <span>Python • Scikit-Learn • Streamlit</span>

        <br><br>

        © 2026 <span>Kavya Rajput</span>

    </div>
    ''',
    unsafe_allow_html=True
)
