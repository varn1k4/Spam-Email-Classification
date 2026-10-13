
import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="wide"
)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
)


@st.cache_resource
def train_models():

    df = pd.read_csv(
        DATASET_URL,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df = df.drop_duplicates()
    df = df.dropna(subset=["label", "message"])

    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(["ham", "spam"])]

    df["label_num"] = df["label"].map({
        "ham": 0,
        "spam": 1
    })

    X = df["message"].astype(str)
    y = df["label_num"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Multinomial Naive Bayes": MultinomialNB(
            alpha=0.3
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )
    }

    trained_models = {}
    results = []
    predictions = {}

    for name, classifier in models.items():

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents="unicode"
                )
            ),
            ("classifier", classifier)
        ])

        pipeline.fit(X_train, y_train)

        predicted = pipeline.predict(X_test)

        trained_models[name] = pipeline
        predictions[name] = predicted

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predicted
            ),
            "Precision": precision_score(
                y_test, predicted, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predicted, zero_division=0
            ),
            "F1-Score": f1_score(
                y_test, predicted, zero_division=0
            )
        })

    results_df = pd.DataFrame(results)

    return (
        trained_models,
        results_df,
        X_test,
        y_test,
        predictions,
        len(df)
    )


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title("📩 Spam Message Classification")

st.write(
    "Analyze a message using machine learning to estimate "
    "whether it is spam or legitimate."
)

st.info(
    "Dataset: SMS Spam Collection | "
    "Feature extraction: TF-IDF | "
    "Algorithms: Logistic Regression, "
    "Multinomial Naive Bayes and XGBoost"
)


# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

try:

    with st.spinner("Loading data and training models..."):

        (
            trained_models,
            results_df,
            X_test,
            y_test,
            predictions,
            total_messages
        ) = train_models()

except Exception as error:

    st.error("Unable to load the dataset or train the models.")

    st.code(str(error))

    st.stop()


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.header("📊 Model Performance Comparison")

display_results = results_df.copy()

for metric in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]:
    display_results[metric] = (
        display_results[metric] * 100
    ).round(2)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)

st.caption(
    "Metrics are measured on the held-out test portion "
    "of the SMS dataset. They do not guarantee performance "
    "on full-length emails."
)


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Messages",
    f"{total_messages:,}"
)

col2.metric(
    "Spam Messages",
    f"{int((y_test == 1).sum()):,} in test set"
)

col3.metric(
    "Legitimate Messages",
    f"{int((y_test == 0).sum()):,} in test set"
)


# --------------------------------------------------
# CHOOSE MODEL
# --------------------------------------------------

st.divider()

st.header("🔍 Test a Message")

selected_model = st.selectbox(
    "Choose a classification model",
    list(trained_models.keys()),
    index=0
)


# --------------------------------------------------
# SAMPLE MESSAGES
# --------------------------------------------------

samples = {
    "Select a sample": "",

    "Spam - Fake Prize":
        "Congratulations! You have won a free prize! "
        "Claim your reward now!",

    "Spam - Urgent Account Alert":
        "URGENT! Your account will be suspended. "
        "Click here immediately to verify your account.",

    "Spam - Lottery":
        "You have won a lottery worth 500000 dollars. "
        "Reply now with your details to claim your prize.",

    "Spam - Fake Job":
        "Work from home and earn 50000 rupees daily. "
        "Pay a registration fee now to secure your job.",

    "Ham - College":
        "Please submit your database assignment by Friday. "
        "Contact your faculty member if you have questions.",

    "Ham - Meeting":
        "Hi team, our project meeting is tomorrow at 10 AM. "
        "Please bring your progress updates.",

    "Ham - Personal":
        "Hey, are we still meeting this Saturday? "
        "Let me know what time works for you."
}

sample_choice = st.selectbox(
    "Or choose a sample message",
    list(samples.keys())
)

default_message = samples[sample_choice]


# --------------------------------------------------
# MESSAGE INPUT
# --------------------------------------------------

message = st.text_area(
    "Enter your message",
    value=default_message,
    placeholder="Paste a message here...",
    height=180
)


# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------

if st.button(
    "🔍 Classify Message",
    use_container_width=True
):

    if not message.strip():

        st.warning("Please enter a message.")

    else:

        model = trained_models[selected_model]

        prediction = int(model.predict([message])[0])

        probabilities = model.predict_proba([message])[0]

        spam_probability = float(probabilities[1])

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:

                st.error("🚨 SPAM")

            else:

                st.success("✅ HAM — Legitimate")

        with col2:

            st.metric(
                "Estimated Spam Probability",
                f"{spam_probability:.1%}"
            )

        st.progress(spam_probability)

        if spam_probability >= 0.5:

            st.write(
                "The model estimates a higher probability "
                "of spam than legitimate content."
            )

        else:

            st.write(
                "The model estimates a higher probability "
                "of legitimate content than spam."
            )

        st.caption(
            "This is a machine learning prediction, not a "
            "guarantee. The model was trained on SMS messages, "
            "so email predictions may be less reliable."
        )


# --------------------------------------------------
# CONFUSION MATRIX AND ERROR ANALYSIS
# --------------------------------------------------

st.divider()

st.header("📈 Test-Set Error Analysis")

analysis_model = st.selectbox(
    "Select model for error analysis",
    list(trained_models.keys()),
    key="analysis_model"
)

selected_predictions = predictions[analysis_model]

tn, fp, fn, tp = confusion_matrix(
    y_test,
    selected_predictions,
    labels=[0, 1]
).ravel()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Correct Ham", int(tn))
c2.metric("Ham Incorrectly Flagged", int(fp))
c3.metric("Spam Missed as Ham", int(fn))
c4.metric("Spam Correctly Detected", int(tp))

st.write(
    "A false negative means a spam message was incorrectly "
    "classified as HAM. A false positive means a legitimate "
    "message was incorrectly classified as SPAM."
)

misclassified = pd.DataFrame({
    "Message": X_test.reset_index(drop=True),
    "Actual": y_test.reset_index(drop=True).map({
        0: "HAM",
        1: "SPAM"
    }),
    "Predicted": pd.Series(selected_predictions).map({
        0: "HAM",
        1: "SPAM"
    })
})

misclassified = misclassified[
    misclassified["Actual"] != misclassified["Predicted"]
]

with st.expander("View misclassified test messages"):

    if misclassified.empty:

        st.success("No misclassified messages in this test set.")

    else:

        st.dataframe(
            misclassified,
            use_container_width=True,
            hide_index=True
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Academic Machine Learning Project | "
    "TF-IDF + Classification Algorithms"
)

