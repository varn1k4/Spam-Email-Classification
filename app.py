
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)


# --------------------------------------------------
# LOAD DATASET AND TRAIN MODEL
# --------------------------------------------------

@st.cache_resource
def train_model():

    dataset_url = (
        "https://raw.githubusercontent.com/"
        "justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    )

    df = pd.read_csv(
        dataset_url,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df = df.drop_duplicates()
    df = df.dropna(subset=["label", "message"])

    df["label_num"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    df = df.dropna(subset=["label_num"])

    X = df["message"].astype(str)
    y = df["label_num"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words="english",
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                n_jobs=1
            )
        )
    ])

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    return model, accuracy, f1, len(df)


# --------------------------------------------------
# APPLICATION INTERFACE
# --------------------------------------------------

st.title("📩 Spam Message Classification")

st.write(
    "Enter a text message to check whether it is "
    "likely to be spam or a normal message."
)

st.info(
    "Machine learning model: XGBoost\n\n"
    "Feature extraction: TF-IDF\n\n"
    "Dataset: SMS Spam Collection"
)


# Train model on first load
try:
    with st.spinner("Loading dataset and training the model..."):
        model, accuracy, f1, total_messages = train_model()

except Exception as error:
    st.error(
        "The model could not be loaded. "
        "Please refresh the application and try again."
    )
    st.caption(f"Technical details: {error}")
    st.stop()


# --------------------------------------------------
# DISPLAY MODEL INFORMATION
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Test Accuracy", f"{accuracy:.1%}")
col2.metric("Test F1-Score", f"{f1:.1%}")
col3.metric("Dataset Messages", f"{total_messages:,}")

st.divider()


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

message = st.text_area(
    "Enter your message",
    placeholder=(
        "Example: Congratulations! "
        "You have won a free prize. Claim now!"
    ),
    height=150
)


if st.button("🔍 Classify Message", use_container_width=True):

    if not message.strip():

        st.warning("Please enter a message first.")

    else:

        prediction = int(model.predict([message])[0])

        probabilities = model.predict_proba([message])[0]

        spam_probability = float(probabilities[1])

        if prediction == 1:

            st.error("🚨 Prediction: SPAM")

            st.write(
                "The model classified this message as spam."
            )

        else:

            st.success("✅ Prediction: HAM")

            st.write(
                "The model classified this message as a normal message."
            )

        st.metric(
            "Estimated Spam Probability",
            f"{spam_probability:.1%}"
        )

        st.caption(
            "This probability is a model estimate, not a guarantee "
            "that the message is spam. Do not use it as the sole "
            "basis for security or financial decisions."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic machine learning project | "
    "TF-IDF + XGBoost"
)

