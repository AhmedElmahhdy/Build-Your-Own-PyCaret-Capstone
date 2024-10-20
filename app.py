# app.py
import streamlit as st # type: ignore
import pandas as pd
from utils import data_handler, eda, model_training, automl

st.title("Automated Machine Learning App")

# Step 1: Upload Data
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
print("=================>>>>>",uploaded_file)
if uploaded_file is not None:
    df = data_handler.load_data(uploaded_file)
    st.write(df.head())

    # Step 2: EDA
    if st.button("Perform EDA"):
        st.write(eda.summary_statistics(df))
        eda.plot_distributions(df, df.columns)

    # Step 3: Select Target Variable
    target_variable = st.selectbox("Select Target Variable", df.columns)

    # Step 4: Model Selection
    model_type = st.selectbox("Model Type", ['classifier', 'regressor'])
    if st.button("Train Model"):
        score, model = model_training.train_model(df, target_variable, model_type)
        st.write(f"Model Score: {score}")

    # Step 5: AutoML
    if st.button("Run AutoML"):
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        best_model, best_score = automl.automl_classifier(X, y)
        st.write(f"Best Model: {best_model}, Score: {best_score}")
