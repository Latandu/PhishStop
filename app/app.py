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
from models.hybrid_mlp_model import HybridMLPClassifier
from data_processing.feature_extraction import FeatureExtraction
from scipy.sparse import hstack
import polars as pl

st.set_page_config(
    page_title="PhishStop Email Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load all three models"""
    with open('output/saved_models/tfidf_vectorizer.pkl', 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    with open('output/saved_models/tfidf_classifier.pkl', 'rb') as f:
        tfidf_model = pickle.load(f)
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('output/saved_models/xgboost_hybrid.json')

    with open('output/saved_artifacts/experiment_config.json', 'r') as f:
        config = json.load(f)
    
    hybrid_config = config['experiment_config']
    hybrid_model = HybridMLPClassifier(
        embedding_dim=hybrid_config['embedding_dim'],
        num_features=len(hybrid_config['numeric_features']),
        feature_hidden_dim=hybrid_config['mlp_feature_hidden_dim'],
        dropout=hybrid_config['mlp_dropout']
    )

    checkpoint = torch.load('output/saved_models/mlp_hybrid.pth', map_location='cpu')
    hybrid_model.load_state_dict(checkpoint['model_state_dict'])
    hybrid_model.eval()
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    feature_names = hybrid_config['numeric_features']
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

def check_for_phsihing(prob):
    return "PHISHING" if prob > 0.5 else "LEGITIMATE"

def prediction(models, df):
    embeddings = models['sentence_model'].encode(
            df['body_subject'].to_list(),
            convert_to_numpy=True,
            batch_size=1
            )
        
    features_array = df.select(models['feature_names']).fill_null(0).to_numpy()

    tfidf_vec = models['tfidf_vectorizer'].transform([df['body_subject'][0]])
    X_tfidf_combined = hstack([tfidf_vec, features_array])
    tfidf_prob = models['tfidf_model'].predict_proba(X_tfidf_combined)[0][1]
    tfidf_pred = check_for_phsihing(tfidf_prob)


    xgb_input = np.concatenate([embeddings, features_array], axis=1)
    xgb_prob = models['xgb_model'].predict_proba(xgb_input)[0][1]
    xgb_pred = check_for_phsihing(xgb_prob)
    
    with torch.no_grad():
        emb_tensor = torch.FloatTensor(embeddings)
        feat_tensor = torch.FloatTensor(features_array)
        hybrid_prob = torch.sigmoid(models['hybrid_model'](emb_tensor, feat_tensor)).item()
    hybrid_pred = check_for_phsihing(hybrid_prob)
        
    return {
            'tfidf': {'prediction': tfidf_pred, 'confidence': float(tfidf_prob * 100)},
            'xgboost': {'prediction': xgb_pred, 'confidence': float(xgb_prob * 100)},
            'hybrid': {'prediction': hybrid_pred, 'confidence': float(hybrid_prob * 100)}
        }, df

def predict_from_eml(eml_path, models):
    try:

        df = models['feature_extractor'].process_eml_to_dataframe(eml_path)        
        if df is None:
            raise Exception("Language check failed: Only English emails are supported. The models were trained on English text only.")
        return prediction(models, df)
    
    except Exception as e:
        raise Exception(f"Error processing .eml file: {str(e)}")


def display_results(results, df=None):
    """Display prediction results in a formatted way"""
    st.header("Analysis Results")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("TF-IDF Model")
        st.caption("Embeddings + Features")
        is_phishing = results['tfidf']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['tfidf']['prediction'])
        else:
            st.success(results['tfidf']['prediction'])
        confidence = results['tfidf']['confidence'] if is_phishing else (100 - results['tfidf']['confidence'])
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.1f}%")
    
    with col2:
        st.subheader("Hybrid XGBoost")
        st.caption("Embeddings + Features")
        is_phishing = results['xgboost']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['xgboost']['prediction'])
        else:
            st.success(results['xgboost']['prediction'])

        confidence = results['xgboost']['confidence'] if is_phishing else (100 - results['xgboost']['confidence'])
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.1f}%")
    
    with col3:
        st.subheader("Hybrid MLP")
        st.caption("Embeddings + Features")
        is_phishing = results['hybrid']['prediction'] == "PHISHING"
        if is_phishing:
            st.error(results['hybrid']['prediction'])
        else:
            st.success(results['hybrid']['prediction'])
        confidence = results['hybrid']['confidence'] if is_phishing else (100 - results['hybrid']['confidence'])
        st.progress(confidence / 100)
        st.caption(f"Confidence: {confidence:.1f}%")
    
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



def display_model_result(title, subtitle, prediction, confidence):
    is_phishing = prediction == "PHISHING"
    adjusted_conf = confidence if is_phishing else (100 - confidence)

    st.subheader(title)
    st.caption(subtitle)

    if is_phishing:
        st.error(prediction)
    else:
        st.success(prediction)

    st.progress(adjusted_conf / 100)
    st.caption(f"Confidence: {adjusted_conf:.1f}%")


def preview_eml_file(path, fe):
    try:
        features = fe.process_eml(path)
    except ValueError:
        st.error(f"Only English emails are supported. The email appears to be in a different language or couldn't be correctly parsed.")
        return False
    except Exception as e:
        st.error(f"**Unexpected Error**\n\nCould not process the email: {str(e)}")
        return False
    if not features:
        st.warning("**Empty Email**\n\nThe email file appears to be empty or could not be parsed properly.")
        return False
    
    body_text = features.get('body_text', '').strip()
    subject = features.get('subject', '').strip()
    
    if not body_text and not subject:
        st.warning("**Empty Email Content**\n\nThe email has no body text or subject. Cannot perform analysis on empty content.")
        return False

    with st.expander("Email Preview", expanded=False):
        st.markdown(f"**From:** {features.get('sender_email', 'N/A')}")
        st.markdown(f"**To:** {features.get('receiver_email', 'N/A')}")
        st.markdown(f"**Subject:** {features.get('subject', 'N/A')}")
        st.markdown(f"**Date:** {features.get('Date', 'N/A')}")
        st.markdown("**Body:**")
        st.text_area(
            "Body Preview",
            features.get('body_text', '')[:500] + "...",
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key="preview"
        )
    return True


st.title("PhishStop Email Analyzer")
st.markdown("Machine Learning-Based Phishing Detection")
st.markdown("---")

with st.spinner("Loading models..."):
    try:
        models = load_models()
        st.success("All models loaded successfully")
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

st.header("Email Analysis")
st.info("**Note:** This tool analyzes English emails only. Models were trained on English text.")

uploaded_file = st.file_uploader("Upload an .eml file", type=['eml'])
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.eml') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        fe = FeatureExtraction()
        is_english = preview_eml_file(tmp_path, fe)

        if is_english:
            if st.button("Analyze Uploaded Email", type="primary", use_container_width=True, key="eml"):
                with st.spinner("Processing .eml file and analyzing..."):
                    try:
                        results, df = predict_from_eml(tmp_path, models)
                        display_results(results, df)
                    except Exception as e:
                        st.error(f"Error analyzing email: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


with st.sidebar:
    st.header("About")
    st.markdown("""
    This tool uses three machine learning models to detect phishing emails:

    **1. TF-IDF + Logistic Regression**  
    - Text-based  
    - Fast and interpretable  

    **2. Hybrid XGBoost**  
    - SVD embeddings + numerical features  

    **3. Hybrid MLP**  
    - Deep learning + feature fusion  

    ---

    ### Language Requirement  
    **English Only:** Models were trained exclusively on English text.

    ---

    ### Input Methods  
    **Upload .EML File** – Full feature extraction & header analysis
    """)

    st.markdown("---")

    st.header("Disclaimer")
    st.markdown("""
    This tool is for educational and research purposes only.  
    Always exercise caution and verify suspicious emails.
    """)