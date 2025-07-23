
import streamlit as st
import pandas as pd
import joblib

# Load the saved model
# Make sure 'best_model.pkl' is in the same directory
try:
    model = joblib.load("best_model.pkl")
except FileNotFoundError:
    st.error("Model file 'best_model.pkl' not found. Please upload it first.")
    st.stop()


st.set_page_config(page_title="Salary Insight Predictor", layout="wide")

# --- Custom CSS for a modern look ---
st.markdown("""
<style>
    /* Main app background */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Main content block with card-like effect */
    .block-container {
        padding: 2rem 3rem 3rem 3rem;
        background-color: #ffffff;
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.17);
        border: 1px solid rgba(255, 255, 255, 0.18);
        max-width: 900px; /* Limit max width for better readability */
        margin: auto; /* Center the card */
    }

    /* Title styling */
    h1 {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c3e50;
        text-align: center;
    }

    /* Subheader/description styling */
    .st-emotion-cache-16idsys p {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
    }
    
    /* Input label styling */
    .st-emotion-cache-1qg05ue {
        font-size: 1rem;
        font-weight: 500;
        color: #34495e;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border: none;
        padding: 0.75rem 0;
        border-radius: 10px;
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        font-weight: 600;
        font-size: 1.25rem;
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(37, 117, 252, 0.4);
        background: linear-gradient(45deg, #2575fc, #6a11cb);
    }
    
    /* Result display styling */
    .result-card {
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-weight: bold;
    }
    .result-success {
        background: linear-gradient(45deg, #28a745, #218838);
    }
    .result-info {
        background: linear-gradient(45deg, #17a2b8, #138496);
    }

</style>
""", unsafe_allow_html=True)

# --- App Layout ---

# Title and description
st.title("💼 Employee Salary  Predictor")
st.markdown("<p>Predict whether a person earns more or less than $50K based on their demographic info.</p>", unsafe_allow_html=True)
st.markdown("---")

# Input features organized into columns
st.subheader("📋 Enter the Details Below")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("age", 17, 75, 30)
    workclass = st.selectbox("workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Others'])
    marital_status = st.selectbox("marital-status", ['Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed', 'Married-spouse-absent'])
    occupation = st.selectbox("occupation", ['Prof-specialty', 'Exec-managerial', 'Craft-repair', 'Sales', 'Tech-support', 'Other-service', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Others'])
    capital_gain = st.number_input("capital-gain", min_value=0, max_value=100000, value=0, step=100)

with col2:
    education_num = st.slider("educational-num", 1, 16, 10)
    relationship = st.selectbox("relationship", ['Husband', 'Wife', 'Not-in-family', 'Own-child', 'Unmarried', 'Other-relative'])
    race = st.selectbox("race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'])
    gender = st.selectbox("gender", ['Male', 'Female'])
    capital_loss = st.number_input("capital-loss", min_value=0, max_value=5000, value=0, step=50)

hours_per_week = st.slider("hours-per-week", 1, 100, 40)
native_country = st.selectbox("native-country", ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'India', 'Others'])


# --- Prediction Logic ---

if st.button("🚀 Predict Salary"):
    # Manual encoding maps from your original script
    encoder_maps = {
        'workclass': {'Private': 4, 'Self-emp-not-inc': 5, 'Self-emp-inc': 3, 'Federal-gov': 1, 'Local-gov': 2, 'State-gov': 6, 'Others': 0},
        'marital-status': {'Never-married': 1, 'Married-civ-spouse': 0, 'Divorced': 2, 'Separated': 2, 'Widowed': 2, 'Married-spouse-absent': 2},
        'occupation': {'Tech-support': 12, 'Craft-repair': 2, 'Other-service': 8, 'Sales': 10, 'Exec-managerial': 4, 'Prof-specialty': 9, 'Handlers-cleaners': 5, 'Machine-op-inspct': 6, 'Adm-clerical': 0, 'Farming-fishing': 3, 'Transport-moving': 13, 'Priv-house-serv': 7, 'Protective-serv': 11, 'Armed-Forces': 1, 'Others': 14},
        'relationship': {'Wife': 5, 'Own-child': 1, 'Husband': 2, 'Not-in-family': 3, 'Other-relative': 0, 'Unmarried': 4},
        'race': {'White': 4, 'Black': 0, 'Asian-Pac-Islander': 1, 'Amer-Indian-Eskimo': 2, 'Other': 3},
        'gender': {'Female': 0, 'Male': 1},
        'native-country': {'United-States': 38, 'Mexico': 23, 'Philippines': 29, 'Germany': 10, 'Canada': 4, 'India': 16, 'Others': 0}
    }

    # Create the input dataframe for the model
    input_data = {
        'age': age,
        'workclass': encoder_maps['workclass'][workclass],
        'fnlwgt': 180000,  # A typical value, as it's not a user input
        'educational-num': education_num,
        'marital-status': encoder_maps['marital-status'][marital_status],
        'occupation': encoder_maps['occupation'][occupation],
        'relationship': encoder_maps['relationship'][relationship],
        'race': encoder_maps['race'][race],
        'gender': encoder_maps['gender'][gender],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': encoder_maps['native-country'].get(native_country, 0)
    }
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    
    # Display result in a styled card
    if prediction == 1:
        st.markdown('<div class="result-card result-success"><h2>Predicted Salary: >$50K</h2><p>This individual is likely to have a higher income bracket.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-card result-info"><h2>Predicted Salary: <=$50K</h2><p>This individual is likely to have a standard income bracket.</p></div>', unsafe_allow_html=True)
