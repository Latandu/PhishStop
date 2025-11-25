import streamlit as st
import numpy as np
import torch
import pickle
import json
import tempfile
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from app.models.hybrid_mlp_model import HybridMLPClassifier
from app.data_processing.feature_extraction import FeatureExtraction

# Set page config
st.set_page_config(
    page_title="PhishStop Email Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading
@st.cache_resource
def load_models():
    """Load all three models"""
    # 1. Load TF-IDF model
    with open('phishstop/models/saved_models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('phishstop/models/saved_models/tfidf_classifier.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
    
    # 2. Load Hybrid XGBoost model
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('phishstop/models/saved_models/xgboost_hybrid.json')
    
    # 3. Load Hybrid MLP model and configuration
    with open('phishstop/models/saved_artifacts/experiment_config.json', 'r') as f:
        config = json.load(f)
    
    hybrid_config = config['experiment_config']
    hybrid_model = HybridMLPClassifier(
        embedding_dim=hybrid_config['embedding_dim'],
        num_features=len(hybrid_config['numeric_features']),
        feature_hidden_dim=hybrid_config['mlp_feature_hidden_dim'],
        dropout=hybrid_config['mlp_dropout']
    )
    checkpoint = torch.load('phishstop/models/saved_models/mlp_hybrid.pth', map_location='cpu')
    hybrid_model.load_state_dict(checkpoint['model_state_dict'])
    hybrid_model.eval()
    
    # Load sentence transformer for embeddings
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Use feature names from experiment config (12 features used during training)
    feature_names = hybrid_config['numeric_features']
      # Initialize FeatureExtraction class
    feature_extractor = FeatureExtraction()
    
    return {
        'tfidf_vectorizer': tfidf_vectorizer,
        'tfidf_model': tfidf_model,
        'xgb_model': xgb_model,
        'hybrid_model': hybrid_model,
        'sentence_model': sentence_model,
        'feature_names': feature_names,
        'feature_extractor': feature_extractor
    }

def predict_all_models(subject, body, models):
    """Get predictions from all three models using FeatureExtraction.process_text"""
    try:
        # Use FeatureExtraction to process subject + body
        # Language detection happens inside HTMLTransformer
        df = models['feature_extractor'].process_text(subject, body)
        
        if df is None:
            raise Exception("Only English emails are supported. The models were trained on English text only.")
        
        # Generate embeddings using sentence transformer
        embeddings = models['sentence_model'].encode(
            df['body_subject'].to_list(),
            convert_to_numpy=True,
            batch_size=1        )
        
        # Extract engineered features (12 features from config)
        features_array = df.select(models['feature_names']).fill_null(0).to_numpy()
        
        # Get the combined text for TF-IDF
        combined_text = df['body_subject'][0]
        
        # 1. TF-IDF Prediction
        from scipy.sparse import hstack
        tfidf_vec = models['tfidf_vectorizer'].transform([combined_text])
        X_tfidf_combined = hstack([tfidf_vec, features_array])
        tfidf_prob = models['tfidf_model'].predict_proba(X_tfidf_combined)[0][1]
        tfidf_pred = "PHISHING" if tfidf_prob > 0.5 else "LEGITIMATE"
        
        # 2. Hybrid XGBoost Prediction
        # Use full embeddings (no SVD)
        xgb_input = np.concatenate([embeddings, features_array], axis=1)
        xgb_prob = models['xgb_model'].predict_proba(xgb_input)[0][1]
        xgb_pred = "PHISHING" if xgb_prob > 0.5 else "LEGITIMATE"
        
        # 3. Hybrid MLP Prediction
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embeddings)
            feat_tensor = torch.FloatTensor(features_array)
            hybrid_prob = torch.sigmoid(models['hybrid_model'](emb_tensor, feat_tensor)).item()
        hybrid_pred = "PHISHING" if hybrid_prob > 0.5 else "LEGITIMATE"
        
        return {
            'tfidf': {'prediction': tfidf_pred, 'confidence': float(tfidf_prob * 100)},
            'xgboost': {'prediction': xgb_pred, 'confidence': float(xgb_prob * 100)},
            'hybrid': {'prediction': hybrid_pred, 'confidence': float(hybrid_prob * 100)}
        }, df
    except Exception as e:
        raise Exception(f"Error processing text: {str(e)}")


def predict_from_eml(eml_path, models):
    """Get predictions from .eml file using FeatureExtraction pipeline"""
    try:
        # Use FeatureExtraction to process the .eml file
        # Language detection happens inside HTMLTransformer
        df = models['feature_extractor'].process_eml_to_dataframe(eml_path)
        
        if df is None:
            raise Exception("Language check failed: Only English emails are supported. The models were trained on English text only.")
        
        # Generate embeddings using sentence transformer
        embeddings = models['sentence_model'].encode(
            df['body_subject'].to_list(),
            convert_to_numpy=True,            batch_size=1
        )
        
        # Extract engineered features (12 features from config)
        features_array = df.select(models['feature_names']).fill_null(0).to_numpy()
        
        # Get the combined text for TF-IDF
        combined_text = df['body_subject'][0]
        
        # 1. TF-IDF Prediction
        from scipy.sparse import hstack
        tfidf_vec = models['tfidf_vectorizer'].transform([combined_text])
        X_tfidf_combined = hstack([tfidf_vec, features_array])
        tfidf_prob = models['tfidf_model'].predict_proba(X_tfidf_combined)[0][1]
        tfidf_pred = "PHISHING" if tfidf_prob > 0.5 else "LEGITIMATE"
        
        # 2. Hybrid XGBoost Prediction
        # Use full embeddings (no SVD)
        xgb_input = np.concatenate([embeddings, features_array], axis=1)
        xgb_prob = models['xgb_model'].predict_proba(xgb_input)[0][1]
        xgb_pred = "PHISHING" if xgb_prob > 0.5 else "LEGITIMATE"
        
        # 3. Hybrid MLP Prediction
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(embeddings)
            feat_tensor = torch.FloatTensor(features_array)
            hybrid_prob = torch.sigmoid(models['hybrid_model'](emb_tensor, feat_tensor)).item()
        hybrid_pred = "PHISHING" if hybrid_prob > 0.5 else "LEGITIMATE"
        
        return {
            'tfidf': {'prediction': tfidf_pred, 'confidence': float(tfidf_prob * 100)},
            'xgboost': {'prediction': xgb_pred, 'confidence': float(xgb_prob * 100)},
            'hybrid': {'prediction': hybrid_pred, 'confidence': float(hybrid_prob * 100)}
        }, df
    except Exception as e:
        raise Exception(f"Error processing .eml file: {str(e)}")


def display_results(results, df=None):
    """Display prediction results in a formatted way"""
    st.header("Analysis Results")
    st.markdown("---")
    
    # Display results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("TF-IDF Model")
        st.caption("Text-Only")
        is_phishing = results['tfidf']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['tfidf']['prediction'])
        else:
            st.success(results['tfidf']['prediction'])
        st.progress(results['tfidf']['confidence'] / 100)
        st.caption(f"Confidence: {results['tfidf']['confidence']:.1f}%")
    
    with col2:
        st.subheader("Hybrid XGBoost")
        st.caption("Embeddings + Features")
        is_phishing = results['xgboost']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['xgboost']['prediction'])
        else:
            st.success(results['xgboost']['prediction'])
        st.progress(results['xgboost']['confidence'] / 100)
        st.caption(f"Confidence: {results['xgboost']['confidence']:.1f}%")
    
    with col3:
        st.subheader("Hybrid MLP")
        st.caption("Embeddings + Features")
        is_phishing = results['hybrid']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['hybrid']['prediction'])
        else:
            st.success(results['hybrid']['prediction'])
        st.progress(results['hybrid']['confidence'] / 100)
        st.caption(f"Confidence: {results['hybrid']['confidence']:.1f}%")
    
    # Overall verdict
    avg_confidence = np.mean([results['tfidf']['confidence'], 
                             results['xgboost']['confidence'], 
                             results['hybrid']['confidence']])
    
    st.markdown("---")
    st.subheader("Overall Assessment")
    
    if avg_confidence > 50:
        st.error(f"**HIGH RISK DETECTED**")
        st.metric("Average Phishing Probability", f"{avg_confidence:.1f}%")
        st.warning("Do not click any links or provide personal information. Report this email as phishing.")
    else:
        st.success(f"**LOW RISK DETECTED**")
        st.metric("Average Phishing Probability", f"{avg_confidence:.1f}%")
        st.info("Email appears legitimate. However, always verify the sender's identity before taking action.")


# Main app
st.title("PhishStop Email Analyzer")
st.markdown("Machine Learning-Based Phishing Detection")
st.markdown("---")

# Load models
with st.spinner("Loading models..."):
    try:
        models = load_models()
        st.success("All models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

# Input section
st.header("Email Analysis")
st.info("ℹ️ **Note:** This tool only analyzes English emails. The models were trained on English text only.")

# Add tabs for different input methods
tab1, tab2 = st.tabs(["Manual Entry", "Upload EML File"])

with tab1:
    subject = st.text_input("Email Subject", placeholder="Re: Your account needs verification")
    body = st.text_area("Email Body", height=200, 
                        placeholder="Dear user,\n\nYour account will be suspended unless you verify...")
    
    # Analyze button
    if st.button("Analyze Email", type="primary", use_container_width=True, key="manual"):
        if not subject or not body:
            st.warning("Please enter both subject and body")
        else:
            with st.spinner("Analyzing..."):
                try:
                    results, df = predict_all_models(subject, body, models)
                    display_results(results, df)
                except Exception as e:
                    st.error(f"Error analyzing email: {str(e)}")

with tab2:
    uploaded_file = st.file_uploader("Upload an .eml file", type=['eml'], help="Upload a raw email file (.eml format)")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.eml') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        try:
            # Display email preview
            with st.expander("Email Preview", expanded=False):
                from phishstop.data_processing.feature_extraction import FeatureExtraction
                fe = FeatureExtraction()
                features = fe.process_eml(tmp_path)
                
                st.markdown(f"**From:** {features.get('sender_email', 'N/A')}")
                st.markdown(f"**To:** {features.get('receiver_email', 'N/A')}")
                st.markdown(f"**Subject:** {features.get('subject', 'N/A')}")
                st.markdown(f"**Date:** {features.get('Date', 'N/A')}")
                st.markdown("**Body:**")
                st.text_area("Body Preview", features.get('body_text', '')[:500] + "...", height=150, disabled=True, key="preview", label_visibility="collapsed")
            
            if st.button("Analyze Uploaded Email", type="primary", use_container_width=True, key="eml"):
                with st.spinner("Processing .eml file and analyzing..."):
                    try:
                        results, df = predict_from_eml(tmp_path, models)
                        display_results(results, df)
                    except Exception as e:
                        st.error(f"Error analyzing email: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

# Sidebar with info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses three different machine learning models to detect phishing emails:
    
    **1. TF-IDF + Logistic Regression**
    - Text-based analysis
    - Fast and interpretable
    
    **2. Hybrid XGBoost**
    - SVD embeddings + numerical features
    - Gradient boosting on combined features
    
    **3. Hybrid MLP**
    - Combines embeddings + features
    - Deep learning approach
    
    ---
    
    ### ⚠️ Language Requirement
    
    **English Only:** This tool is designed to analyze English emails only. 
    The models were trained exclusively on English text and may produce 
    inaccurate results for other languages.
    
    ---
    
    ### Input Methods
    
    **Manual Entry**
    - Enter subject and body text directly
    - Quick analysis for testing
    
    **Upload .EML File**
    - Full email parsing with headers
    - Extracts 17+ engineered features
    - Authentication checks (SPF, DKIM, DMARC)
    - Link analysis & malicious domain detection
    - More accurate predictions
    """)
    
    st.markdown("---")
    
    st.header("Disclaimer")
    st.markdown("""
    This is an automated analysis tool for educational and research purposes. 
    Always exercise caution with suspicious emails and verify sender authenticity.
    """)
