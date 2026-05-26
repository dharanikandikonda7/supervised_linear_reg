# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Page settings
st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎓",
    layout="wide"
)

# Title
st.title("🎓 Student Score Prediction")
st.markdown("### Linear Regression Machine Learning Project")

# Read dataset
df = pd.read_csv("student-por.csv")

# Remove duplicates
df = df.drop_duplicates()

# Store encoders
encoders = {}

# Numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Categorical columns
cat_cols = df.select_dtypes(include="object").columns

# Fill missing numerical values
num_imputer = SimpleImputer(strategy="mean")
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Fill missing categorical values
cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# Encode categorical columns
for col in cat_cols:

    # Create encoder
    le = LabelEncoder()

    # Encode values
    df[col] = le.fit_transform(df[col])

    # Store encoder
    encoders[col] = le

# Features
X = df.drop("G3", axis=1)

# Target
y = df["G3"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

# Fit and transform train data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

# Predict output
y_pred = model.predict(X_test_scaled)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Show metrics
st.subheader("📊 Model Performance")

# Create columns
c1, c2, c3, c4 = st.columns(4)

# Show MAE
c1.metric("MAE", round(mae, 2))

# Show MSE
c2.metric("MSE", round(mse, 2))

# Show RMSE
c3.metric("RMSE", round(rmse, 2))

# Show R2 score
c4.metric("R² Score", round(r2, 2))

# Graph section
st.subheader("📈 Actual vs Predicted Values")

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Scatter plot
ax.scatter(y_test, y_pred)

# Labels
ax.set_xlabel("Actual Scores")
ax.set_ylabel("Predicted Scores")

# Title
ax.set_title("Linear Regression Prediction")

# Show graph
st.pyplot(fig)

# Prediction section
st.subheader("🎯 Predict Student Score")

# Store user input
user_input = {}

# Create columns layout
col1, col2 = st.columns(2)

# Numerical inputs
with col1:

    # Loop numerical columns
    for col in num_cols:

        # Skip target column
        if col != "G3":

            # Create slider
            user_input[col] = st.slider(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )

# Categorical inputs
with col2:

    # Loop categorical columns
    for col in cat_cols:

        # Get original labels
        options = list(encoders[col].classes_)

        # Create selectbox
        selected = st.selectbox(f"{col}", options)

        # Encode selected value
        user_input[col] = encoders[col].transform([selected])[0]

# Predict button
if st.button("Predict Score"):

    # Convert to dataframe
    input_df = pd.DataFrame([user_input])

    # Arrange columns correctly
    input_df = input_df[X.columns]

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Show result
    st.success(f"🎯 Predicted Student Score : {round(prediction[0], 2)}")