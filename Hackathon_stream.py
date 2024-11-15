import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
from sklearn.preprocessing import StandardScaler

# Custom CSS from original code
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
    
    /* Other styles from original code */
    .css-1xarl3l {
        background-color: #FFF8E7;
        border: 1px solid #DAA520;
        border-radius: 4px;
        padding: 1rem;
    }
    
    .css-1d391kg {
        background-color: #FFF8E7;
    }
    
    .stExpander {
        border: 1px solid #DAA520;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    
    h1, h2, h3, h4 {
        color: #800020;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('recall_primary_model_hypertuned_resampled.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()


df2 = pd.read_csv('unscaled_trainval_df.csv')

# Assuming your DataFrame is called `df` and contains these columns:
X = df2[['CREDIT_UTILIZATION_RATIO', 'AGE', 'BANK_TENURE', 'MONTHLY_INCOME']]

# Initialize a scaler for each feature
scaler_credit_utilization = StandardScaler()
scaler_age = StandardScaler()
scaler_bank_tenure = StandardScaler()
scaler_monthly_income = StandardScaler()

# Fit each scaler on the respective feature column
scaler_credit_utilization.fit_transform(X[['CREDIT_UTILIZATION_RATIO']])
scaler_age.fit_transform(X[['AGE']])
scaler_bank_tenure.fit_transform(X[['BANK_TENURE']])
scaler_monthly_income.fit_transform(X[['MONTHLY_INCOME']])

# Create the Streamlit app
st.title('🏦 MSME Loan Approval Prediction')
st.markdown("This application predicts loan approvals based on various customer factors")
st.markdown("""
<div style='background-color: #FFF8E7; padding: 1rem; border-radius: 4px; border: 1px solid #DAA520;'>
    📝 Please fill out the form below with the customer's information for loan approval prediction.
</div>
""", unsafe_allow_html=True)

def get_user_input():
    # Initialize session state for storing values
    if 'input_dict' not in st.session_state:
        st.session_state.input_dict = {}
    
    # Initialize input_dict from session state
    input_dict = st.session_state.input_dict
    
    # Financial Information
    st.header('Financial Information')
    with st.expander("Financial Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            credit_balance = st.number_input(
                'Current Credit Card Balance (PHP)',
                value=st.session_state.input_dict.get('credit_balance', 0),
                help="Total outstanding balance on credit cards",
                key='credit_balance'
            )
            credit_limit = st.number_input(
                'Total Credit Card Limit (PHP)',
                value=st.session_state.input_dict.get('credit_limit', 1),
                help="Total credit limit across all credit cards",
                key='credit_limit'
            )
            
            
            # Calculate credit utilization ratio
            st.session_state.input_dict['CREDIT_UTILIZATION_RATIO_unscale'] = credit_balance / credit_limit 
            st.session_state.input_dict['CREDIT_UTILIZATION_RATIO'] = scaler_credit_utilization.transform([[st.session_state.input_dict['CREDIT_UTILIZATION_RATIO_unscale']]]).tolist()[0][0]
            
            

            monthly_income = st.number_input(
                'Monthly Income (PHP)', 
                value=st.session_state.input_dict.get('MONTHLY_INCOME_unscale', 0),
                help="Monthly income in local currency",
                key='monthly_income'
            )
            st.session_state.input_dict['MONTHLY_INCOME_unscale'] = monthly_income
            st.session_state.input_dict['MONTHLY_INCOME'] = scaler_monthly_income.transform([[st.session_state.input_dict['MONTHLY_INCOME_unscale']]]).tolist()[0][0]

            
            bank_tenure = st.number_input(
                'Bank Tenure (years)', 
                value=st.session_state.input_dict.get('BANK_TENURE_unscale', 0),
                help="Number of years as a bank customer",
                key='bank_tenure'
            )
            st.session_state.input_dict['BANK_TENURE_unscale'] = bank_tenure
            st.session_state.input_dict['BANK_TENURE'] = scaler_bank_tenure.transform([[st.session_state.input_dict['BANK_TENURE_unscale']]]).tolist()[0][0]
        
        with col2:
            # Account Indicators
            st.subheader("Account Types")
            input_dict['SAVINGS_ACCOUNT_INDICATOR'] = st.checkbox(
                'Has Savings Account',
                value=st.session_state.input_dict.get('SAVINGS_ACCOUNT_INDICATOR', False),
                key='savings_account'
            )
            input_dict['CHECKING_ACCOUNT_INDICATOR'] = st.checkbox(
                'Has Checking Account',
                value=st.session_state.input_dict.get('CHECKING_ACCOUNT_INDICATOR', False),
                key='checking_account'
            )
            input_dict['TIME_DEPOSIT_ACCOUNT_INDICATOR'] = st.checkbox(
                'Has Time Deposit Account',
                value=st.session_state.input_dict.get('TIME_DEPOSIT_ACCOUNT_INDICATOR', False),
                key='time_deposit_account'
            )
            
            # Loan Indicators
            st.subheader("Existing Loans")
            loan_indicators = ['AUTO_LOAN_INDICATOR', 'HOUSING_LOAN_INDICATOR', 
                             'PERSONAL_LOAN_INDICATOR', 'BB_LOAN_INDICATOR']
            for indicator in loan_indicators:
                st.session_state.input_dict[indicator] = st.checkbox(
                    indicator.replace('_INDICATOR', '').replace('_', ' ').title(),
                    value=st.session_state.input_dict.get(indicator, False),
                    key=indicator
                )

    # Personal Information
    st.header('Personal Information')
    with st.expander("Demographics", expanded=True):
        col3, col4 = st.columns(2)
        
        with col3:
            age = st.number_input(
                'Age', 
                value=st.session_state.input_dict.get('AGE_unscale', 18),
                help="Age in years",
                key='age'
            )
            st.session_state.input_dict['AGE_unscale'] = age
            st.session_state.input_dict['AGE'] = scaler_age.transform([[st.session_state.input_dict['AGE_unscale']]]).tolist()[0][0]
            
            gender = st.selectbox('Gender', ['Female', 'Male'], 
                                index=1 if st.session_state.input_dict.get('GENDER_MALE', False) else 0,
                                key='gender')
            st.session_state.input_dict['GENDER_MALE'] = gender == "Male"
            
            # Education
            education = st.radio("Education Level", ["Low", "Mid", "High"],
                               index=["Low", "Mid", "High"].index(st.session_state.input_dict.get('education', "Low")),
                               key='education')
            st.session_state.input_dict['EDUCATION_LOW'] = education == "Low"
            st.session_state.input_dict['EDUCATION_MID'] = education == "Mid"

        with col4:
            # Marital Status
            marital_status = st.selectbox(
                "Marital Status",
                ["Single", "Married", "Separated", "Widow(er)", "Other"],
                index=["Single", "Married", "Separated", "Widow(er)", "Other"].index(
                    st.session_state.input_dict.get('marital_status', "Single")
                ),
                key='marital_status'
            )
            st.session_state.input_dict.update({
                'MARITAL_STATUS_MARRIED': marital_status == "Married",
                'MARITAL_STATUS_OTHER': marital_status == "Other",
                'MARITAL_STATUS_SEPARATED': marital_status == "Separated",
                'MARITAL_STATUS_SINGLE': marital_status == "Single",
                'MARITAL_STATUS_WIDOW(ER)': marital_status == "Widow(er)"
            })
            
            input_dict['FILCHI_INDICATOR_Y'] = st.checkbox('Filipino-Chinese',
                value=st.session_state.input_dict.get('FILCHI_INDICATOR_Y', False),
                key='filchi_indicator')

    # Lifestyle Information
    st.header('Lifestyle Information')
    with st.expander("Lifestyle Details", expanded=True):
        col5, col6 = st.columns(2)
        
        with col5:
            # Property ownership and financial products
            lifestyle_indicators = [
                'HOME_OWNER_INDICATOR_Y',
                'CAR_OWNER_INDICATOR_Y',
                'LIFE_INSURANCE_INDICATOR',
                'NONLIFE_INSURANCE_INDICATOR',
                'INVESTMENT_INDICATOR'
            ]
            
            for indicator in lifestyle_indicators:
                display_name = indicator.replace('_INDICATOR_Y', '').replace('_INDICATOR', '').replace('_', ' ').title()
                st.session_state.input_dict[indicator] = st.checkbox(
                    display_name,
                    value=st.session_state.input_dict.get(indicator, False),
                    key=f'lifestyle_{indicator}'
                )
        
        with col6:
            # SEC Classification logic based on monthly income
            monthly_income = st.session_state.input_dict.get('MONTHLY_INCOME', 0)
            
            # Calculate SEC based on income ranges
            if monthly_income >= 250000:
                sec = 'A'
            elif monthly_income >= 100000:
                sec = 'B1'
            elif monthly_income >= 50000:
                sec = 'B2'
            elif monthly_income >= 25000:
                sec = 'C1'
            elif monthly_income >= 15000:
                sec = 'C2'
            elif monthly_income >= 10000:
                sec = 'D'
            else:
                sec = 'E'
            
            # Update SEC indicators in session state
            sec_options = ['B1', 'B2', 'C1', 'C2', 'D', 'E']
            for option in sec_options:
                st.session_state.input_dict[f'SEC_{option}'] = (sec == option)
            
            # Display current SEC classification
            st.write(f"SEC Classification based on income: {sec}")

    # Social Responsibility
    st.header('Social Responsibility')
    with st.expander("Social Engagement", expanded=True):
        col9, col10 = st.columns(2)
        
        with col9:
            input_dict['HUMANITARIAN_AFF_INDICATOR_Y'] = st.checkbox(
                'Transacted with Humanitarian Groups',
                value=st.session_state.input_dict.get('HUMANITARIAN_AFF_INDICATOR_Y', False),
                key='humanitarian_indicator'
            )
            input_dict['ENVIRONMENTAL_AFF_INDICATOR_Y'] = st.checkbox(
                'Transacted with Environmental Groups',
                value=st.session_state.input_dict.get('ENVIRONMENTAL_AFF_INDICATOR_Y', False),
                key='environmental_indicator'
            )
        
        with col10:
            input_dict['OF_INDICATOR_Y'] = st.checkbox(
                'Overseas Filipino',
                value=st.session_state.input_dict.get('OF_INDICATOR_Y', False),
                key='of_indicator'
            )
            input_dict['RETIREES_INDICATOR_Y'] = st.checkbox(
                'Retiree',
                value=st.session_state.input_dict.get('RETIREES_INDICATOR_Y', False),
                key='retirees_indicator'
            )
    
    # Update session state with all values
    st.session_state.input_dict.update(input_dict)
    
    return input_dict

    return st.session_state.input_dict

# Get user input and create DataFrame
user_input = get_user_input()

# Prepare features in correct order
expected_features = ['CREDIT_UTILIZATION_RATIO', 'AGE', 'BANK_TENURE', 'MONTHLY_INCOME',
                    'GENDER_MALE', 'MARITAL_STATUS_MARRIED', 'MARITAL_STATUS_OTHER',
                    'MARITAL_STATUS_SEPARATED', 'MARITAL_STATUS_SINGLE',
                    'MARITAL_STATUS_WIDOW(ER)', 'EDUCATION_LOW', 'EDUCATION_MID',
                    'DIGITAL_INDICATOR_TRADITIONAL', 'SEC_B1', 'SEC_B2', 'SEC_C1', 'SEC_C2',
                    'SEC_D', 'SEC_E', 'HOME_OWNER_INDICATOR_Y', 'CAR_OWNER_INDICATOR_Y',
                    'ENVIRONMENTAL_AFF_INDICATOR_Y', 'HUMANITARIAN_AFF_INDICATOR_Y',
                    'OF_INDICATOR_Y', 'RETIREES_INDICATOR_Y', 'FILCHI_INDICATOR_Y',
                    'SAVINGS_ACCOUNT_INDICATOR', 'CHECKING_ACCOUNT_INDICATOR',
                    'TIME_DEPOSIT_ACCOUNT_INDICATOR', 'AUTO_LOAN_INDICATOR',
                    'HOUSING_LOAN_INDICATOR', 'PERSONAL_LOAN_INDICATOR', 'BB_LOAN_INDICATOR',
                    'LIFE_INSURANCE_INDICATOR', 'NONLIFE_INSURANCE_INDICATOR',
                    'INVESTMENT_INDICATOR']

input_df = pd.DataFrame([user_input])
input_df = input_df.reindex(columns=expected_features, fill_value=0)



# Add prediction button
if st.button('Predict Loan Approval'):
    # Make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    
    # Show prediction
    st.subheader('Prediction Results')
    
    # Create columns for prediction display
    col_pred1, col_pred2 = st.columns([2, 3])
    
    with col_pred1:
        if prediction[0] == 0:  # 0 means approved
            st.success('Loan Approved! ✅')
            # Show approval probability
            st.metric(
                label="Approval Probability",
                value=f"{probability[0][0]:.2%}"  # Use probability[0][0] for approval
            )
        else:  # 1 means rejected
            st.error('Loan Not Approved ❌')
            # Show rejection probability
            st.metric(
                label="Rejection Probability",
                value=f"{probability[0][1]:.2%}"  # Use probability[0][1] for rejection
            )

# Sidebar content
st.sidebar.header("Feature Categories")
st.sidebar.write("""
- Financial Factors
- Demographic Factors
- Socioeconomic Factors
- Lifestyle Factors
- Social Responsibility
""")

# System information in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("System Information")
st.sidebar.write(f"Python version: {sys.version.split()[0]}")
st.sidebar.write(f"Pandas version: {pd.__version__}")

# Add explanation section at the bottom
st.markdown("""
---
### How it works
This loan approval prediction model evaluates applications based on multiple factors:

#### 🏦 Financial Factors
- Credit utilization ratio
- Monthly income
- Banking history
- Existing accounts and loans

#### 👥 Demographics
- Age and gender
- Education level
- Marital status

#### 📊 Socioeconomic Status
- SEC classification

#### 🏠 Lifestyle Indicators
- Property ownership
- Insurance coverage
- Banking preferences

#### 🤝 Social Responsibility
- Community involvement
- Environmental engagement
- Organization memberships
""")
