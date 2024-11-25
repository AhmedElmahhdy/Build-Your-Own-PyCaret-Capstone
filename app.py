import streamlit as st 
import pandas as pd
from utils import data_handler, eda, model_training, automl, detect_model_type

st.title("Automated Machine Learning App")

# Upload Data
uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
if uploaded_file is not None:
    df = data_handler.load_data(uploaded_file)
    st.write(df.head())



    # EDA
    st.write("Perform EDA")
    df_summary = eda.summary_statistics(df)
    st.write("Summary Statistics:\n", df_summary)

    # Handle Missing Continuous Values
    continuous_cols , missing_continuous_values = eda.continuous_variables(df)
    st.write("Continuous Features:\n", continuous_cols) 
    st.write("Total Missing Values:\n", missing_continuous_values)
    if missing_continuous_values > 0:
        handle_way = st.selectbox("Handle Missing Values in continuous variables", ['Mean', 'Median', 'Mode'])
        df = eda.handle_missing_continuous_values(df, handle_way)
        st.write("Data after handling missing values:\n",df)
    
    # Handle Missing Categorical Data
    categorical_cols , missing_categorical_values = eda.categorical_variables(df)
    st.write("Categorical Features:\n", categorical_cols) 
    st.write("Missing Values in Categorical Features:\n", missing_categorical_values)
    if missing_categorical_values > 0:
        handle_way = st.selectbox("Handle Missing Values in categorical variables", ['Mode','Additional Class'])
        df = eda.handle_categorical_data(df, handle_way)
        st.write("Data after handling missing values:\n",df)
    
        
        

        
# Drop Columns
    st.write("Drop Columns")
    column_to_drop = st.selectbox("Select columns to drop",df.columns)


    if st.button("Drop Selected Columns"):
        if column_to_drop in df.columns:
            df = eda.drop_columns(df, column_to_drop)
            st.write(f"Updated Dataset (after dropping '{column_to_drop}'):")
            st.write(df)
        else:
            st.write(f"Column '{column_to_drop}' not found in the dataset.")
        # try:
        #     print("============== we are here in button ==============")
        #     if column_to_drop in df.columns:
        #         print("============== we are here in if ==============")
        #         df = eda.drop_columns(df, column_to_drop)
        #         print("============== we are here after drop ==============")
        #         st.write(f"Updated Dataset (after dropping '{column_to_drop}'):")
        #         st.write(df)
        #     else:
        #         st.write(f"Column '{column_to_drop}' not found in the dataset.")
        # except Exception as e:
        #     st.write(e)
       
    # Select Target Variable
    target_variable = st.selectbox("Select Target Variable", df.columns)
    model_type = detect_model_type.detect_task_type(df[target_variable])
    st.write(" Model Type is :",model_type)


    # Train Model
    if st.button("Train Model"):
        df = eda.handle_categorical_data(df, 'additional class')
        st.write("Training Model...")
        score, model = model_training.train_model(df, target_variable, model_type)
        st.write(f"Model Score: {score}  ||  Model: {model}")

    # AutoML
    if st.button("Run AutoML"):
        df = eda.handle_categorical_data(df, 'additional class')
        X = df.drop(columns=[target_variable])
        y = df[target_variable].astype(int)
        best_model, best_score , model_scores = automl.automl_classifier(X, y)
        st.write(f"Best Model: {best_model}, Score: {best_score}")
        st.write("==============================")
        for score in model_scores:
            st.write(f"{score}: {model_scores[score]}")
    




























# # Import Streamlit and other required modules
# import streamlit as st
# import pandas as pd
# from utils import data_handler, eda, model_training, automl, detect_model_type

# st.title("Automated Machine Learning App")

# # Initialize df in session state
# if "df" not in st.session_state:
#     st.session_state.df = None

# # Upload Data
# uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
# if uploaded_file is not None:
#     st.session_state.df = data_handler.load_data(uploaded_file)
#     st.write(st.session_state.df.head())

# if st.session_state.df is not None:
#     df = st.session_state.df.copy()

#     # EDA
#     if st.button("Perform EDA"):
#         df_summary = eda.summary_statistics(df)
#         st.write("Summary Statistics:\n", df_summary)

#         # Handle Missing Continuous Values
#         continuous_cols, missing_continuous_values = eda.continuous_variables(df)
#         st.write("Continuous Features:\n", continuous_cols)
#         st.write("Missing Values in Continuous Features:\n", missing_continuous_values)
#         if missing_continuous_values > 0:
#             handle_way = st.selectbox("Handle Missing Values in continuous variables", ['Mean', 'Median', 'Mode'])
#             st.session_state.df = eda.handle_missing_continuous_values(st.session_state.df, handle_way)
#             st.write("Data after handling missing values:\n", st.session_state.df)

#         # Handle Missing Categorical Data
#         categorical_cols, missing_categorical_values = eda.categorical_variables(df)
#         st.write("Categorical Features:\n", categorical_cols)
#         st.write("Missing Values in Categorical Features:\n", missing_categorical_values)
#         if missing_categorical_values > 0:
#             handle_way = st.selectbox("Handle Missing Values in categorical variables", ['Mode', 'Additional Class'])
#             st.session_state.df = eda.handle_categorical_data(st.session_state.df, handle_way)
#             st.write("Data after handling missing values:\n", st.session_state.df)

#         # Drop Columns
#         st.write("Drop Columns")
#         column_to_drop = st.selectbox("Select a column to drop:", st.session_state.df.columns, placeholder="Select columns to drop")
#         if st.button("Drop Selected Columns"):
#             if column_to_drop in st.session_state.df.columns:
#                 st.session_state.df = eda.drop_columns(st.session_state.df, column_to_drop)
#                 st.write(f"Updated Dataset (after dropping '{column_to_drop}'):")
#                 st.write(st.session_state.df)
#             else:
#                 st.write(f"Column '{column_to_drop}' not found in the dataset.")

#     # Select Target Variable
#     target_variable = st.selectbox("Select Target Variable", st.session_state.df.columns)
#     model_type = detect_model_type.detect_task_type(st.session_state.df[target_variable])
#     st.write("Model Type is:", model_type)

#     # Train Model
#     if st.button("Train Model"):
#         st.session_state.df = eda.handle_categorical_data(st.session_state.df, 'additional class')
#         st.write("Training Model...")
#         score, model = model_training.train_model(st.session_state.df, target_variable, model_type)
#         st.write(f"Model Score: {score} || Model: {model}")

#     # AutoML
#     if st.button("Run AutoML"):
#         st.session_state.df = eda.handle_categorical_data(st.session_state.df, 'additional class')
#         X = st.session_state.df.drop(columns=[target_variable])
#         y = st.session_state.df[target_variable].astype(int)
#         best_model, best_score, model_scores = automl.automl_classifier(X, y)
#         st.write(f"Best Model: {best_model}, Score: {best_score}")
#         st.write("==============================")
#         for score in model_scores:
#             st.write(f"{score}: {model_scores[score]}")

