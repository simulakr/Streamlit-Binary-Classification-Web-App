import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt

st.title("Binary Classification Web App")
st.sidebar.title("Binary Classification Web App")
st.markdown("Are your mushrooms edible or poisonous? 🍄")
st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

@st.cache_data
def load_data():
    data = pd.read_csv("mushrooms.csv") 
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data 

@st.cache_data
def split(df):
    y = df["type"]
    x = df.drop(columns=["type"])
    return train_test_split(x, y, test_size=0.3, random_state=42)

def plot_metrics(metrics_list, model, x_test, y_test):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax)
        st.pyplot(fig)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        y_scores = model.predict_proba(x_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC Curve")
        ax.plot([0, 1], [0, 1], "k--")
        st.pyplot(fig)

    if "Precision Recall Curve" in metrics_list:
        st.subheader("Precision Recall Curve")
        y_scores = model.predict_proba(x_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, label="Precision-Recall Curve")
        st.pyplot(fig)

df = load_data()
x_train, x_test, y_train, y_test = split(df)

st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel")
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key="gamma")

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred), 2))
        plot_metrics(metrics, model, x_test, y_test)

if classifier == "Logistic Regression":
    st.sidebar.subheader("Model Hyperparameters")
    C = st.sidebar.number_input("C (Regularization Parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum Number of Iterations", 100, 500, key="max_iter")

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred), 2))
        plot_metrics(metrics, model, x_test, y_test)

if classifier == "Random Forest":
    st.sidebar.subheader("Model Hyperparameters")
    n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=100, key="n_estimators")
    max_depth = st.sidebar.number_input("Maximum depth of tree", 1, 20, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building", ("True", "False"), key="bootstrap") == "True"

    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision Recall Curve"))

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy:", round(accuracy, 2))
        st.write("Precision:", round(precision_score(y_test, y_pred), 2))
        st.write("Recall:", round(recall_score(y_test, y_pred), 2))
        plot_metrics(metrics, model, x_test, y_test)

if st.sidebar.checkbox("Show raw data", False):
    st.subheader("Mushroom Data Set (Classification)")
    st.write(df)
