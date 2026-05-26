# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page title
st.set_page_config(page_title="Regression Project", layout="wide")

# App title
st.title("🎓 Student Performance Regression Project")

# Upload dataset
file = st.file_uploader("Upload student-por.csv Dataset", type=["csv"])

# Run after upload
if file is not None:

    # Read dataset
    df = pd.read_csv(file)

    # Show dataset
    st.subheader("Dataset")
    st.dataframe(df.head())

    # Show dataset shape
    st.write("Dataset Shape :", df.shape)

    # Check missing values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Separate numerical columns
    num_cols = df.select_dtypes(include=np.number).columns

    # Separate categorical columns
    cat_cols = df.select_dtypes(include="object").columns

    # Fill missing numerical values
    num_imputer = SimpleImputer(strategy="mean")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])

    # Fill missing categorical values
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

    # Encode categorical columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Show cleaned dataset
    st.subheader("Cleaned Dataset")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("Select Target Column", df.columns, index=len(df.columns)-1)

    # Create features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create regression model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)

    # Calculate MAE
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    # Calculate R2 Score
    r2 = r2_score(y_test, y_pred)

    # Show model metrics
    st.subheader("Model Performance")

    # Create columns
    c1, c2, c3, c4 = st.columns(4)

    # Show MAE
    c1.metric("MAE", round(mae, 2))

    # Show MSE
    c2.metric("MSE", round(mse, 2))

    # Show RMSE
    c3.metric("RMSE", round(rmse, 2))

    # Show R2 Score
    c4.metric("R2 Score", round(r2, 2))

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted")

    # Create figure
    fig, ax = plt.subplots()

    # Scatter plot
    ax.scatter(y_test, y_pred)

    # Set labels
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")

    # Set title
    ax.set_title("Regression Plot")

    # Show plot
    st.pyplot(fig)

    # Prediction section
    st.subheader("Predict Student Score")

    # Store inputs
    user_input = {}

    # Create input fields
    for col in X.columns:
        user_input[col] = st.number_input(
            f"Enter {col}",
            value=float(df[col].mean())
        )

    # Predict button
    if st.button("Predict"):

        # Convert into dataframe
        input_df = pd.DataFrame([user_input])

        # Scale input
        input_scaled = scaler.transform(input_df)

        # Predict value
        prediction = model.predict(input_scaled)

        # Show prediction
        st.success(f"Predicted {target} : {round(prediction[0], 2)}")

# Message before upload
else:

    # Show info
    st.info("Please upload the dataset")