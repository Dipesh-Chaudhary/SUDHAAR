"""
Model loading utilities with GPU/CPU support
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForMaskedLM,
)
import streamlit as st
from config import *

@st.cache_resource
def load_models():
    """
    Load all required models with caching
    Returns device and all models
    """
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with st.spinner(f"Loading models on {device}..."):
        # Load tokenizers
        ged_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        gector_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, add_prefix_space=True)
        
        # Load models
        ged_model = AutoModelForSequenceClassification.from_pretrained(
            GED_MODEL_HUB_ID
        ).to(device)
        
        binary_model = AutoModelForTokenClassification.from_pretrained(
            BINARY_MODEL_HUB_ID
        ).to(device)
        
        error_model = AutoModelForTokenClassification.from_pretrained(
            ERROR_TYPE_MODEL_HUB_ID
        ).to(device)
        
        mlm_model = AutoModelForMaskedLM.from_pretrained(
            MLM_MODEL_HUB_ID
        ).to(device)
        
        # Setting to evaluation mode
        ged_model.eval()
        binary_model.eval()
        error_model.eval()
        mlm_model.eval()
        
    return {
        'device': device,
        'ged_tokenizer': ged_tokenizer,
        'gector_tokenizer': gector_tokenizer,
        'ged_model': ged_model,
        'binary_model': binary_model,
        'error_model': error_model,
        'mlm_model': mlm_model
    }