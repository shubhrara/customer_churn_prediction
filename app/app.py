import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Customer Churn Prediction Portal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism styling
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151922 100%);
        color: #e0e6ed;
    }
    
    /* Headers styling */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Glassmorphic card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 168, 204, 0.4);
    }
    
    /* Custom buttons */
    .stButton>button {
        background: linear-gradient(90deg, #0072ff 0%, #00c6ff 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(0, 114, 255, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(0, 114, 255, 0.5) !important;
    }
    
    /* Recommendation card */
    .rec-card {
        background: rgba(255, 159, 67, 0.05);
        border-left: 5px solid #ff9f43;
        border-radius: 4px 12px 12px 4px;
        padding: 16px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to read image as base64
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# Load pipeline model
@st.cache_resource
def load_pipeline():
    pipeline_path = os.path.join("models", "churn_pipeline.pkl")
    if os.path.exists(pipeline_path):
        with open(pipeline_path, "rb") as f:
            return pickle.load(f)
    return None

pipeline = load_pipeline()

# App Header
st.title("🛡️ Customer Churn Prediction Portal")
st.markdown("##### *Predict customer attrition risk in real-time and discover data-driven retention strategies.*")
st.markdown("---")

if pipeline is None:
    st.error("⚠️ The model pipeline could not be loaded. Please run model training `python src/train.py` first to generate `models/churn_pipeline.pkl`.")
    st.stop()

# Initialize session state for predictions to persist results between widget interactions
if "has_run" not in st.session_state:
    st.session_state.has_run = False
    st.session_state.pred = None
    st.session_state.prob = None
    st.session_state.playbook_recs = []

# Create tabs
tab_prediction, tab_analytics = st.tabs(["🔮 Prediction Engine", "📈 Business Analytics"])

with tab_prediction:
    st.markdown("### Enter Customer Details")
    st.write("Complete the form below to compute the probability of the customer churning.")
    
    # Organize input form in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card"><h4>👥 Demographics</h4>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card"><h4>💳 Contract & Billing</h4>', unsafe_allow_html=True)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-card"><h4>📞 Core Services</h4>', unsafe_allow_html=True)
        phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card"><h4>🔒 Premium Add-ons</h4>', unsafe_allow_html=True)
        online_security = st.selectbox("Online Security?", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup?", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection?", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support?", ["Yes", "No", "No internet service"])
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-card"><h4>📺 Streaming Features</h4>', unsafe_allow_html=True)
        streaming_tv = st.selectbox("Streaming TV?", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies?", ["Yes", "No", "No internet service"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-card"><h4>📊 Financials</h4>', unsafe_allow_html=True)
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=18.0, max_value=120.0, value=65.0, step=1.0)
        # Automatically estimate total charges based on tenure and monthly charges, but let user modify
        est_total = min(float(tenure * monthly_charges), 9000.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, max_value=9000.0, value=est_total, step=10.0)
        st.markdown('</div>', unsafe_allow_html=True)

    # Process prediction
    st.markdown("---")
    
    # Predict button
    if st.button("RUN PREDICTION"):
        # Build input dataframe
        input_data = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }])
        
        # Perform prediction using pipeline
        st.session_state.pred = pipeline.predict(input_data)[0]
        st.session_state.prob = pipeline.predict_proba(input_data)[0][1]
        st.session_state.has_run = True
        
        # Build playbook recommendations
        recommendations = []
        if contract == "Month-to-month":
            recommendations.append(
                "📌 **Contract Conversion**: Customer is on a month-to-month plan. Offer a **10% discount** on their monthly rate if they sign a 1-year contract, which reduces churn probability significantly."
            )
        if payment_method == "Electronic check":
            recommendations.append(
                "📌 **Autopay Incentive**: Electronic Check users are vulnerable to passive attrition. Offer a **5-unit billing credit** for one month to transition the customer to Credit Card or Bank Transfer autopay."
            )
        if internet_service == "Fiber optic":
            recommendations.append(
                "📌 **Service Satisfaction Check**: Fiber Optic customers have high churn. Arrange a proactively scheduled technical health check to resolve speed/reliability complaints."
            )
        if online_security == "No" or tech_support == "No":
            recommendations.append(
                "📌 **Security/Tech Add-on Bundle**: Customer is missing online security or tech support. Offer a free 3-month trial of the **Safety & Protection bundle** to build product sticky-ness."
            )
        if tenure <= 12:
            recommendations.append(
                "📌 **First-Year Onboarding**: This client has less than 1 year of tenure. Direct the customer success team to initiate a check-in call to ensure smooth usage."
            )
        st.session_state.playbook_recs = recommendations

    # Display prediction results
    if st.session_state.has_run:
        pred = st.session_state.pred
        prob = st.session_state.prob
        recommendations = st.session_state.playbook_recs
        
        # Calculate Confidence (probability of the selected class)
        confidence = max(prob, 1 - prob) * 100
        pred_label = "Churn" if pred == 1 else "No Churn"
        
        # Classification threshold rules
        if prob > 0.60:
            status = "High Risk"
            action = "Immediate Retention Campaign"
            color_hex = "#ff4b4b"  # Red
            badge_style = "background-color: rgba(255, 75, 75, 0.15); color: #ff4b4b; border: 1px solid #ff4b4b;"
        elif prob >= 0.30:
            status = "Medium Risk"
            action = "Monitor Customer"
            color_hex = "#ff9f43"  # Orange
            badge_style = "background-color: rgba(255, 159, 67, 0.15); color: #ff9f43; border: 1px solid #ff9f43;"
        else:
            status = "Low Risk"
            action = "Healthy Customer"
            color_hex = "#2ed573"  # Green
            badge_style = "background-color: rgba(46, 213, 115, 0.15); color: #2ed573; border: 1px solid #2ed573;"
            
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin-top: 0; margin-bottom: 20px;">📊 Prediction Results</h3>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.07);">
                        <td style="padding: 12px 0; font-weight: 600; color: #a0aec0;">Prediction</td>
                        <td style="padding: 12px 0; text-align: right; font-weight: 800; font-size: 1.15em;">{pred_label}</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.07);">
                        <td style="padding: 12px 0; font-weight: 600; color: #a0aec0;">Model Confidence</td>
                        <td style="padding: 12px 0; text-align: right; font-weight: 800; font-size: 1.15em; color: #00c6ff;">{confidence:.1f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.07);">
                        <td style="padding: 12px 0; font-weight: 600; color: #a0aec0;">Customer Status</td>
                        <td style="padding: 12px 0; text-align: right;">
                            <span style="padding: 6px 14px; border-radius: 20px; font-weight: 800; font-size: 0.95em; {badge_style}">
                                {status.upper()}
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding: 12px 0; font-weight: 600; color: #a0aec0;">Recommended Action</td>
                        <td style="padding: 12px 0; text-align: right; font-weight: 800; color: {color_hex};">{action}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            recommendations_html = ""
            for rec in recommendations:
                recommendations_html += f'<div class="rec-card">{rec}</div>'
            
            if not recommendations:
                recommendations_html = "<p style='color: #888; font-style: italic;'>Customer shows highly stable attributes. Maintain standard engagement.</p>"
                
            st.markdown(f"""
            <div class="metric-card" style="height: 100%;">
                <h3 style="margin-top: 0; margin-bottom: 20px;">💡 Actionable Customer Retention Playbook</h3>
                <div>
                    {recommendations_html}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Awaiting prediction placeholder card
        st.markdown("""
        <div class="metric-card" style="text-align: center; padding: 40px 20px;">
            <h3 style="color: #a0aec0; margin-bottom: 10px;">🔮 Prediction Results Awaiting Input</h3>
            <p style="color: #718096; max-width: 600px; margin: 0 auto;">
                Fill out the customer demographics, services, contract, and financials in the columns above, then click <b>RUN PREDICTION</b> to compute the model confidence, risk classification, and retention actions.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab_analytics:
    st.markdown("### Model Evaluation & Insights Dashboard")
    st.write("Analyze performance metrics and statistical data to understand how the ML algorithm predicts churn.")
    
    # Row 1: Model Information Cards
    info1, info2, info3 = st.columns(3)
    
    with info1:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Best Model</h4>
            <h3 style="margin: 10px 0; color: #ffffff; font-size: 1.6em;">Logistic Regression</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Selected after GridSearchCV tuning</p>
        </div>
        """, unsafe_allow_html=True)
        
    with info2:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Dataset</h4>
            <h3 style="margin: 10px 0; color: #ffffff; font-size: 1.45em;">IBM Telco Customer Churn</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Customer retention dataset</p>
        </div>
        """, unsafe_allow_html=True)
        
    with info3:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Prediction Type</h4>
            <h3 style="margin: 10px 0; color: #ffffff; font-size: 1.6em;">Binary Classification</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Churn vs No Churn</p>
        </div>
        """, unsafe_allow_html=True)

    # Row 2: Performance Metric Cards
    stat1, stat2, stat3 = st.columns(3)
    
    with stat1:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">Test Accuracy</h4>
            <h3 style="margin: 10px 0; color: #00c6ff; font-size: 2.2em; font-weight: 800;">80.5%</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Percentage of correct predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
    with stat2:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">ROC-AUC Score</h4>
            <h3 style="margin: 10px 0; color: #0072ff; font-size: 2.2em; font-weight: 800;">0.841</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Model discrimination ability</p>
        </div>
        """, unsafe_allow_html=True)
        
    with stat3:
        st.markdown("""
        <div class="metric-card" style="text-align: center; height: 160px; display: flex; flex-direction: column; justify-content: center;">
            <h4 style="margin: 0; color: #a0aec0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px;">F1-Score</h4>
            <h3 style="margin: 10px 0; color: #2ed573; font-size: 2.2em; font-weight: 800;">60.3%</h3>
            <p style="margin: 0; color: #718096; font-size: 0.85em;">Harmonic mean of precision & recall</p>
        </div>
        """, unsafe_allow_html=True)
        
    # Visualizations Column Layout
    vcol1, vcol2 = st.columns(2)
    
    with vcol1:
        # Feature Importance Card
        feat_path = os.path.join("app", "static", "feature_importance.png")
        feat_b64 = get_image_base64(feat_path)
        if feat_b64:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top: 0; margin-bottom: 15px;">📊 Feature Importance</h4>
                <img src="data:image/png;base64,{feat_b64}" style="width: 100%; border-radius: 8px;"/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Feature importance chart not found. Run evaluations first.")
            
        # Confusion Matrix Card
        cm_path = os.path.join("app", "static", "confusion_matrix.png")
        cm_b64 = get_image_base64(cm_path)
        if cm_b64:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top: 0; margin-bottom: 15px;">📊 Confusion Matrix</h4>
                <img src="data:image/png;base64,{cm_b64}" style="width: 100%; border-radius: 8px;"/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Confusion matrix chart not found. Run evaluations first.")
            
    with vcol2:
        # ROC Curve Comparison Card
        roc_path = os.path.join("app", "static", "roc_curve_comparison.png")
        roc_b64 = get_image_base64(roc_path)
        if roc_b64:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="margin-top: 0; margin-bottom: 15px;">📈 ROC Curve Comparison</h4>
                <img src="data:image/png;base64,{roc_b64}" style="width: 100%; border-radius: 8px;"/>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("ROC comparison chart not found. Run evaluations first.")
            
        # Summary Card
        st.markdown("""
        <div class="metric-card">
            <h4 style="margin-top: 0; margin-bottom: 15px;">📑 Summary of Key Insights</h4>
            <ul style="padding-left: 20px; line-height: 1.6; color: #cbd5e0; margin: 0;">
                <li style="margin-bottom: 10px;"><b>Logistic Regression</b> was chosen as the production classifier due to its superior F1-score balance, robust convergence, and direct interpretability of coefficients.</li>
                <li style="margin-bottom: 10px;"><b>Primary Attrition Drivers</b>: Short tenure, Month-to-month contracts, Fiber optic connection problems, and Electronic Check payments.</li>
                <li style="margin-bottom: 0;"><b>Optimal Retention Policy</b>: Transitioning month-to-month clients to contracts and check payments to autopay. Proactively addressing technical feedback of fiber optic customers.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
