import streamlit as st
import numpy as np
import joblib
import tensorflow as tf
import os
import pandas as pd
from datetime import datetime

# Hide TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# -----------------------------
# Load Model & Files
# -----------------------------
@st.cache_resource
def load_models():
    # Rebuild model architecture manually
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(11,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Load weights
    model.load_weights("model.h5")

    # Load scaler & features
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("features.pkl")

    return model, scaler, feature_names

model, scaler, feature_names = load_models()

# -----------------------------
# Session State for History
# -----------------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = "light"

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Churn Prediction System",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Custom CSS based on theme
# -----------------------------
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a2e;
        }
        .main-header {
            background: linear-gradient(135deg, #16213e, #0f3460);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .card {
            background-color: #16213e;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #0f3460;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="main-header">
    <h1 style="color: white; text-align: center; margin: 0;">📊 Telecom Customer Churn Prediction</h1>
    <p style="color: #E0E7FF; text-align: center; margin-top: 0.5rem;">AI-Powered Customer Retention Analytics</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar Controls
# -----------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    
    # Theme Toggle
    theme_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.theme == "dark")
    if theme_toggle:
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    
    # Reset button
    if st.button("🔄 Reset All Values", use_container_width=True):
        st.rerun()
    
    # Export History
    if len(st.session_state.prediction_history) > 0:
        if st.button("📥 Export History", use_container_width=True):
            df = pd.DataFrame(st.session_state.prediction_history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📈 Session Stats")
    total_pred = len(st.session_state.prediction_history)
    st.metric("Total Predictions", total_pred)
    
    if total_pred > 0:
        high_risk = sum(1 for p in st.session_state.prediction_history if p['probability'] > 0.5)
        st.metric("High Risk Cases", high_risk, delta=f"{(high_risk/total_pred)*100:.0f}%")
    
    st.markdown("---")
    st.markdown("### 📌 About")
    st.info("""
    **Model:** Neural Network  
    **Accuracy:** 87.5%  
    **Features:** 11 parameters  
    **Year:** 2026
    """)

# -----------------------------
# Main Content Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📝 Prediction", "📊 History", "📖 Guide"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Customer Information")
        st.markdown("---")
        
        inputs = []
        
        # Create 3 columns for better layout
        cols = st.columns(3)
        
        for i, feature in enumerate(feature_names):
            col_idx = i % 3
            with cols[col_idx]:
                val = st.number_input(
                    f"**{feature}**", 
                    value=0.0, 
                    key=f"input_{feature}",
                    step=0.01,
                    format="%.2f",
                    help=f"Enter value for {feature}"
                )
                inputs.append(val)
        
        # Batch prediction option
        st.markdown("---")
        batch_mode = st.checkbox("Enable Batch Prediction Mode")
        
        if batch_mode:
            st.info("Enter multiple values separated by commas (e.g., 100, 200, 300)")
            batch_inputs = []
            for feature in feature_names:
                batch_val = st.text_input(f"{feature} (comma separated)", key=f"batch_{feature}")
                batch_inputs.append(batch_val)
    
    with col2:
        st.markdown("### 🎯 Prediction Settings")
        st.markdown("---")
        
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.05, help="Adjust churn sensitivity")
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        ✅ **High values (>0.7)** indicate risk  
        ✅ **Low values (<0.3)** indicate safety  
        ✅ **Adjust threshold** for sensitivity  
        ✅ **Check history** for patterns
        """)
        
    st.markdown("### 📊 Sample Data")
    if st.button("Load Sample Customer"):
        sample_values = [650, 45, 3, 50000, 2, 1, 1, 0, 1, 0, 0]
    # Direct number input values set karne ke liye session state use karo
        for i, feature in enumerate(feature_names):
            st.session_state[f"input_{feature}"] = sample_values[i]
        st.success("✅ Sample data loaded!")
        st.rerun()

with tab2:
    st.markdown("### 📊 Prediction History")
    st.markdown("---")
    
    if len(st.session_state.prediction_history) == 0:
        st.info("No predictions yet. Go to Prediction tab to get started!")
    else:
        # Show history as dataframe
        history_df = pd.DataFrame(st.session_state.prediction_history)
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        history_df['probability'] = history_df['probability'].apply(lambda x: f"{x:.2%}")
        history_df['result'] = history_df.apply(lambda x: "⚠️ High Risk" if x['risk'] else "✅ Low Risk", axis=1)
        
        st.dataframe(
            history_df[['timestamp', 'probability', 'result']],
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("### 📈 Summary Statistics")
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        probs = [p['probability'] for p in st.session_state.prediction_history]
        with col_stat1:
            st.metric("Avg Probability", f"{np.mean(probs):.2%}")
        with col_stat2:
            st.metric("Max Risk", f"{np.max(probs):.2%}")
        with col_stat3:
            st.metric("Min Risk", f"{np.min(probs):.2%}")
        with col_stat4:
            high_risk_count = sum(1 for p in st.session_state.prediction_history if p['probability'] > 0.5)
            st.metric("High Risk %", f"{(high_risk_count/len(probs))*100:.1f}%")
        
        # Clear history button
        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.prediction_history = []
            st.rerun()

with tab3:
    st.markdown("### 📖 User Guide")
    st.markdown("---")
    
    col_guide1, col_guide2 = st.columns(2)
    
    with col_guide1:
        st.markdown("#### 🎯 How to Use")
        st.markdown("""
        1. **Enter Customer Details**
           - Fill all 11 feature values
           - Use number inputs with step 0.01
        
        2. **Adjust Settings**
           - Change risk threshold if needed
           - Switch to dark mode for comfort
        
        3. **Get Prediction**
           - Click Predict button
           - View probability and recommendation
        """)
        
        st.markdown("#### 📊 Understanding Results")
        st.markdown("""
        - **< 0.3**: Low risk (Green)
        - **0.3 - 0.7**: Medium risk (Yellow)  
        - **> 0.7**: High risk (Red)
        """)
    
    with col_guide2:
        st.markdown("#### 🚀 Features")
        st.markdown("""
        - ✅ Real-time predictions
        - ✅ Prediction history tracking
        - ✅ Export data to CSV
        - ✅ Dark/Light theme
        - ✅ Batch prediction mode
        - ✅ Adjustable threshold
        - ✅ Sample data loader
        """)
        
        st.markdown("#### 💡 Pro Tips")
        st.markdown("""
        - Use history to track patterns
        - Export data for analysis
        - Lower threshold for early warnings
        - Save frequent customers as samples
        """)

# -----------------------------
# Prediction Button & Logic
# -----------------------------
if tab1:
    with tab1:
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_btn = st.button("🔮 Predict Churn", use_container_width=True, type="primary")
        
        if predict_btn:
            if batch_mode:
                # Batch prediction logic
                st.markdown("### 📊 Batch Prediction Results")
                
                batch_data = []
                valid_batch = True
                
                for idx, feature in enumerate(feature_names):
                    batch_vals = batch_inputs[idx].strip()
                    if batch_vals:
                        try:
                            vals = [float(x.strip()) for x in batch_vals.split(',')]
                            batch_data.append(vals)
                        except:
                            st.error(f"Invalid input for {feature}")
                            valid_batch = False
                            break
                    else:
                        valid_batch = False
                        st.warning(f"Please enter values for {feature}")
                
                if valid_batch and batch_data:
                    num_samples = len(batch_data[0])
                    results = []
                    
                    for i in range(num_samples):
                        sample_inputs = [batch_data[feat_idx][i] for feat_idx in range(len(feature_names))]
                        data = np.array(sample_inputs).reshape(1, -1)
                        
                        num_features = len(scaler.mean_)
                        numeric_data = data[:, :num_features]
                        scaled_numeric = scaler.transform(numeric_data)
                        encoded_data = data[:, num_features:]
                        final_input = np.concatenate([scaled_numeric, encoded_data], axis=1)
                        
                        pred = model.predict(final_input, verbose=0)
                        prob = float(pred[0][0])
                        results.append(prob)
                    
                    # Display batch results
                    for i, prob in enumerate(results):
                        st.markdown(f"**Customer {i+1}:** Probability = {prob:.2%}")
                        st.progress(prob)
                        if prob > threshold:
                            st.warning("⚠️ High Risk")
                        else:
                            st.success("✅ Low Risk")
                        st.markdown("---")
            
            else:
                # Single prediction
                data = np.array(inputs).reshape(1, -1)
                
                num_features = len(scaler.mean_)
                numeric_data = data[:, :num_features]
                scaled_numeric = scaler.transform(numeric_data)
                encoded_data = data[:, num_features:]
                final_input = np.concatenate([scaled_numeric, encoded_data], axis=1)
                
                prediction = model.predict(final_input, verbose=0)
                prob = float(prediction[0][0])
                
                # Save to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'probability': prob,
                    'risk': prob > threshold,
                    'inputs': inputs.copy()
                })
                
                # Display Result
                st.markdown("---")
                st.markdown("### 📈 Prediction Result")
                
                # Metrics row
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric("Churn Probability", f"{prob:.2%}")
                with col_met2:
                    risk_level = "High" if prob > threshold else "Low"
                    st.metric("Risk Level", risk_level)
                with col_met3:
                    st.metric("Threshold Used", f"{threshold:.0%}")
                
                # Progress bar
                st.markdown("**Risk Meter**")
                st.progress(prob)
                
                # Result based on threshold
                if prob > threshold:
                    st.error(f"### ❌ High Risk: Customer Likely to Churn")
                    st.warning("**📋 Action Plan:**\n- Offer retention discount\n- Schedule customer call\n- Send personalized offer\n- Priority support access")
                else:
                    st.success(f"### ✅ Low Risk: Customer Likely to Stay")
                    st.info("**📋 Action Plan:**\n- Continue good service\n- Send satisfaction survey\n- Consider loyalty program\n- Regular follow-ups")
                
                # Detailed expander
                with st.expander("📊 View Detailed Analysis"):
                    st.write(f"**Raw Score:** {prob:.4f}")
                    st.write(f"**Risk Classification:** {'High Risk' if prob > threshold else 'Low Risk'}")
                    st.write(f"**Confidence:** {abs(0.5 - prob) * 2:.1%}")
                    st.write(f"**Recommendation Strength:** {'Strong' if abs(prob - 0.5) > 0.3 else 'Moderate'}")
                
                # Feedback
                st.markdown("---")
                feedback = st.radio("Was this prediction accurate?", ["👍 Yes", "👎 No"], horizontal=True)
                if feedback:
                    st.success("Thanks for your feedback! This helps improve our model.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="background-color: #1F2937; padding: 1rem; border-radius: 10px; margin-top: 2rem; text-align: center;">
    <p style="color: #9CA3AF; margin: 0;">
        © 2026 Churn Prediction System | Neural Network Model v2.0 | Real-time Analytics
    </p>
</div>
""", unsafe_allow_html=True)