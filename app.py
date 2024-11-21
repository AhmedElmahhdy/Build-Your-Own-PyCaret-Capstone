import streamlit as st # type: ignore
import pandas as pd
from xgboost import XGBClassifier
from utils import data_handler, eda, model_training, automl, detect_model_type

st.title("Automated Machine Learning App")

# Upload Data
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
print("=================>>>>>",uploaded_file)
if uploaded_file is not None:
    df = data_handler.load_data(uploaded_file)
    st.write(df.head())



    # EDA
    if st.button("Perform EDA"):
        df_summary , missing_values = eda.summary_statistics(df)
        st.write("Summary Statistics:\n", df_summary)
        st.write("Handle Categorical Data")
        st.write(eda.convert_categorical_to_numeric(df))
        st.write("Missing Values:\n", missing_values)
        st.pyplot(eda.plot_distributions(df, df.columns))
       
    # Select Target Variable
    target_variable = st.selectbox("Select Target Variable", df.columns)
    model_type = detect_model_type.detect_task_type(df[target_variable])
    st.write(" Model Type is :",model_type)


    # Train Model
    if st.button("Train Model"):
        df = eda.convert_categorical_to_numeric(df)
        st.write("Training Model...")
        score, model = model_training.train_model(df, target_variable, model_type)
        st.write(f"Model Score: {score}  ||  Model: {model}")

    # AutoML
    if st.button("Run AutoML"):
        df = eda.convert_categorical_to_numeric(df)
        X = df.drop(columns=[target_variable])
        y = df[target_variable].astype(int)
        best_model, best_score , model_scores = automl.automl_classifier(X, y)
        st.write(f"Best Model: {best_model}, Score: {best_score}")
        st.write("==============================")
        for score in model_scores:
            st.write(f"{score}: {model_scores[score]}")
    
