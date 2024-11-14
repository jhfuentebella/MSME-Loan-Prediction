import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys

# Custom CSS
st.markdown("""
<style>
    /* Main title styling */
    .css-10trblm {
        color: #800020;
        font-weight: 600;
    }
    
    /* Subheader styling */
    .css-1xhc3i {
        color: #800020;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #FFF8E7;
        border: 1px solid #DAA520;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #800020;
        color: white;
        border: 2px solid #800020;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #DAA520;
        border-color: #DAA520;
        color: white;
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background-color: #FFF8E7;
        border: 1px solid #DAA520;
        border-radius: 4px;
        padding: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFF8E7;
    }
    
    /* Card styling for expandable sections */
    .stExpander {
        border: 1px solid #DAA520;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    /* Success message styling */
    .element-container.css-1e5imcs.e1tzin5v3 .stSuccess {
        background-color: #DAA520;
        color: white;
    }
    
    /* Header sections */
    h1, h2, h3, h4 {
        color: #800020;
    }
</style>
""", unsafe_allow_html=True)

# Add error handling for XGBoost import
try:
    import xgboost as xgb
    st.sidebar.success("✨ XGBoost is successfully installed!")
except ImportError:
    st.error("XGBoost is not installed. Please install it using: pip install xgboost")
    st.stop()

# Load the saved model and weights with error handling
@st.cache_resource
def load_model():
    with open('recall_primary_model_hypertuned_resampled.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('feature_column_hypertuned_resampled.pkl', 'rb') as file:
        columns = pickle.load(file)
    return model, columns

# Load the model and weights
model, feature_columns = load_model()

# Create the Streamlit app
st.title('🏦 MSME Loan Approval Prediction')
st.markdown("This application is a prototype for our model that considers non-financial factors in predicting loan approvals for MSMEs")
st.markdown("""
<div style='background-color: #FFF8E7; padding: 1rem; border-radius: 4px; border: 1px solid #DAA520;'>
    📝 Please fill out the form below with the customer's information for loan approval prediction.
</div>
""", unsafe_allow_html=True)

def get_user_input():
    input_dict = {}
    
    # Financial Factors
    st.header('Financial Information')
    with st.expander("Financial Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            input_dict['CURRENT_CREDIT_CARD_BALANCE'] = st.number_input(
                'Current Unsettled Credit Card Balance (PHP)',
                min_value=0,
                help="Total outstanding balance on credit cards"
            )
            input_dict['TOTAL_CREDIT_CARD_LIMIT'] = st.number_input(
                'Total Credit Card Limit (PHP)',
                min_value=0,
                help="Total credit limit across all credit cards"
            )
            
            # Calculate credit utilization ratio
            if input_dict['TOTAL_CREDIT_CARD_LIMIT']==0:
                input_dict['TOTAL_CREDIT_CARD_LIMIT']=1
            input_dict['CREDIT_UTILIZATION_RATIO'] = input_dict['CURRENT_CREDIT_CARD_BALANCE'] / input_dict['TOTAL_CREDIT_CARD_LIMIT']
            
            input_dict['MONTHLY_INCOME'] = st.number_input(
                'Monthly Income (PHP)', 
                min_value=0,
                help="Monthly income in local currency"
            )
            input_dict['BANK_TENURE'] = st.number_input(
                'Bank Tenure (years)', 
                min_value=0,
                help="Number of years as a bank customer"
            )
            input_dict['INCOME_SOURCE'] = st.selectbox(
            'Income Source',
            ['BUSINESS', 'SALARY', 'ALLOWANCE', 'PENSION', 'OTHER_SOURCES_NOT_SPECIFIED', 
             'REMITTANCE', 'COMMISSION', 'INTEREST_SAVINGS_PLACEMENTS_INVESTMENTS', 
             'ECONOMICALLY_INACTIVE', 'DONATION']
        )
        
        with col2:
            # Account Indicators
            st.subheader("Account Types")
            input_dict['SAVINGS_ACCOUNT_INDICATOR'] = st.checkbox('Has Savings Account')
            input_dict['CHECKING_ACCOUNT_INDICATOR'] = st.checkbox('Has Checking Account')
            input_dict['TIME_DEPOSIT_ACCOUNT_INDICATOR'] = st.checkbox('Has Time Deposit Account')
        
        # Loan Indicators
        st.subheader("Do you have any of the following Loans?")
        col3, col4 = st.columns(2)
        with col3:
            input_dict['PERSONAL_LOAN_INDICATOR'] = st.checkbox('Personal Loan')
            input_dict['BB_LOAN_INDICATOR'] = st.checkbox('BB Loan')
            input_dict['AUTO_LOAN_INDICATOR'] = st.checkbox('Auto Loan')
        with col4:
            input_dict['HOUSING_LOAN_INDICATOR'] = st.checkbox('Housing Loan')
            input_dict['INVESTMENT_INDICATOR'] = st.checkbox('Investments')

    # Demographic Factors
    st.header('Personal Information')
    with st.expander("Demographics", expanded=True):
        col5, col6 = st.columns(2)
        
        with col5:
            input_dict['AGE'] = st.number_input(
                'Age', 
                min_value=18, 
                max_value=100,
                help="Age in years"
            )
            input_dict['GENDER_MALE'] = st.selectbox('Gender', ['Female', 'Male']) == 'Male'
            input_dict['CUSTOMER_LOCATION'] = st.selectbox(
            'Customer Location',
            ['NATIONAL CAPITAL REGION', 'REGION VII (CENTRAL VISAYAS)', 'REGION III (CENTRAL LUZON)', 
             'REGION VI (WESTERN VISAYAS)', 'REGION VIII (EASTERN VISAYAS)', 'REGION IV-A (CALABARZON)',
             'REGION XII (SOCCSKSARGEN)', 'REGION XIII (CARAGA)', 'REGION IX (ZAMBOANGA PENINSULA)', 
             'REGION X (NORTHERN MINDANAO)', 'REGION I (ILOCOS REGION)', 'REGION XI (DAVAO REGION)',
             'REGION V (BICOL REGION)', 'REGION II (CAGAYAN VALLEY)', 
             'CORDILLERA ADMINISTRATIVE REGION (CAR)', 'MIMAROPA REGION']
        )
        
        with col6:
            # Education
            education_status = st.radio(
                "Education Level",
                ["Low", "Mid", "High"]
            )
            input_dict['EDUCATION_LOW'] = education_status == "Low"
            input_dict['EDUCATION_MID'] = education_status == "Mid"
            input_dict['EDUCATION_HIGH'] = education_status == "High"

            input_dict['FILCHI_INDICATOR_Y'] = st.checkbox('Are you Filipino-Chinese?')
            
        # Marital Status
        marital_status = st.selectbox(
            "Marital Status",
            ["Single", "Married", "Separated", "Widow(er)", "Other", "Divorced"]
        )
        input_dict['MARITAL_STATUS_SINGLE'] = marital_status == "Single"
        input_dict['MARITAL_STATUS_MARRIED'] = marital_status == "Married"
        input_dict['MARITAL_STATUS_SEPARATED'] = marital_status == "Separated"
        input_dict['MARITAL_STATUS_WIDOW(ER)'] = marital_status == "Widow(er)"
        input_dict['MARITAL_STATUS_OTHER'] = marital_status == "Other"
        input_dict['MARITAL_STATUS_OTHER'] = marital_status == "Divorced"

        

    # Socioeconomic Factors
    def calculate_socioeconomic_class(monthly_income):
        sec_ranges = {
            'A': (263707.14, 9999999999999999),
            'B1': (141455.43, 189999.7), 
            'B2': (40000.35, 100118.64),
            'C1': (20600.72, 56006.01),
            'C2': (26315.38, 38459.96),
            'D': (14643.98, 19999.42),
            'E': (0, 9999.3)
        }
        
        for sec, (min_income, max_income) in sec_ranges.items():
            if min_income <= monthly_income <= max_income:
                return sec
        
        return None

   
    sec = calculate_socioeconomic_class(input_dict['MONTHLY_INCOME'])
    if sec:
        for s in ['A', 'B1', 'B2', 'C1', 'C2', 'D', 'E']:
            input_dict[f'SEC_{s}'] = s == sec
    else:
        for s in ['A', 'B1', 'B2', 'C1', 'C2', 'D', 'E']:
             input_dict[f'SEC_{s}'] = False 

    
    # Lifestyle Factors
    st.header('Lifestyle Information')
    with st.expander("Lifestyle Details", expanded=True):
        col7, col8 = st.columns(2)
        
        with col7:
            input_dict['HOME_OWNER_INDICATOR_Y'] = st.checkbox('Home Owner')
            input_dict['CAR_OWNER_INDICATOR_Y'] = st.checkbox('Car Owner')
            input_dict['DIGITAL_INDICATOR_TRADITIONAL'] = st.checkbox('Traditional Banking User')
        
        with col8:
            input_dict['LIFE_INSURANCE_INDICATOR'] = st.checkbox('Has Life Insurance')
            input_dict['NONLIFE_INSURANCE_INDICATOR'] = st.checkbox('Has Non-life Insurance')

    # Social Responsibility
    st.header('Social Responsibility')
    with st.expander("Social Engagement", expanded=True):
        col9, col10 = st.columns(2)
        
        with col9:
            input_dict['HUMANITARIAN_AFF_INDICATOR_Y'] = st.checkbox('Transacted with Humanitarian Groups')
            input_dict['ENVIRONMENTAL_AFF_INDICATOR_Y'] = st.checkbox('Transacted with Environmental Groups')
        
        with col10:
            input_dict['OF_INDICATOR_Y'] = st.checkbox('Overseas Filipino')
            input_dict['RETIREES_INDICATOR_Y'] = st.checkbox('Retiree')

    # Convert to DataFrame with correct column order
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    return input_df

# Get user input
user_input = get_user_input()

# Add prediction button
if st.button('Predict Payment Behavior'):
    # Make prediction
    prediction = model.predict(user_input)
    probability = model.predict_proba(user_input)

    # # Show prediction
    # st.subheader('Prediction')
    # prediction_text = 'Likely to Default' if prediction[0] == 1 else 'Likely to Pay On Time'
    # st.write(prediction_text)

    # Show prediction
    st.subheader('Prediction Results')
        
    # Create columns for prediction display
    col_pred1, col_pred2 = st.columns([2, 3])
        
    with col_pred1:
        if prediction[0] == 0:
            st.success('Loan Approved! ✅')
        else:
            st.error('Loan Not Approved ❌')
            
        # Show probability
        st.metric(
            label="Approval Probability",
            value=f"{probability[0][1]:.2%}"
            )

    # # Show probability
    # st.subheader('Prediction Probability')
    # st.write(f'Probability of Paying On Time: {probability[0][0]:.2%}')
    # st.write(f'Probability of Defaulting: {probability[0][1]:.2%}')

# Sidebar for feature importance overview
st.sidebar.header("Feature Categories")
st.sidebar.write("""
- Financial Factors
- Demographic Factors
- Socioeconomic Factors
- Lifestyle Factors
- Social Responsibility
""")

# # Add prediction button
# if st.button('Predict Loan Approval', type='primary'):
#     try:
#         # Make prediction
#         prediction = model.predict(weighted_input)
#         probability = model.predict_proba(weighted_input)
        
#         # Show prediction
#         st.subheader('Prediction Results')
        
#         # Create columns for prediction display
#         col_pred1, col_pred2 = st.columns([2, 3])
        
#         with col_pred1:
#             if prediction[0] == 1:
#                 st.success('Loan Approved! ✅')
#             else:
#                 st.error('Loan Not Approved ❌')
            
#             # Show probability
#             st.metric(
#                 label="Approval Probability",
#                 value=f"{probability[0][1]:.2%}"
#             )
        
#         # Show feature importance for this prediction
#         if hasattr(model, 'feature_importances_'):
#             with col_pred2:
#                 st.subheader('Top Feature Contributions')
#                 importances = pd.DataFrame({
#                     'Feature': feature_columns,
#                     'Importance': model.feature_importances_
#                 }).sort_values('Importance', ascending=False).head(5)
                
#                 st.bar_chart(importances.set_index('Feature')['Importance'])
    
#     except Exception as e:
#         st.error(f"Error making prediction: {str(e)}")

# Add explanation section
st.markdown("""
---
### How it works
This loan approval prediction model evaluates applications based on multiple factors:

#### 🏦 Financial Factors
- Current credit card balance
- Total credit card limit
- Income level
- Banking history
- Existing loans and accounts

#### 👥 Demographics
- Age and gender
- Education level
- Marital status

#### 📊 Socioeconomic Status
- SEC classification (B1-E)

#### 🏠 Lifestyle Indicators
- Property ownership
- Insurance coverage
- Banking preferences

#### 🤝 Social Responsibility
- Community involvement
- Environmental engagement
- Organization memberships

The model uses XGBoost algorithm and has been trained on historical approval data.
""")

# Display system information
st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.write(f"Python version: {sys.version.split()[0]}")
st.sidebar.write(f"Pandas version: {pd.__version__}")
try:
    import xgboost as xgb
    st.sidebar.write(f"XGBoost version: {xgb.__version__}")
except ImportError:
    st.sidebar.error("XGBoost not installed")