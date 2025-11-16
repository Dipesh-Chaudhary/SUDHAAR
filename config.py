"""
Configuration file for Nepali GEC models
"""

# Base model from IRIIS-RESEARCH for all tokenizers
BASE_MODEL = "IRIIS-RESEARCH/RoBERTa_Nepali_125M"

# Model Hub IDs (fine-tuned models on HuggingFace)
GED_MODEL_HUB_ID = "DipeshChaudhary/roberta-nepali-sequence-ged"
BINARY_MODEL_HUB_ID = "DipeshChaudhary/nepali-gec-binary-detector"
ERROR_TYPE_MODEL_HUB_ID = "DipeshChaudhary/nepali-gec-error-type-classifier"
MLM_MODEL_HUB_ID = "DipeshChaudhary/nepali-mlm-guesser-finetuned-model-1"

# Error type labels (7 types)
ERROR_ID2LABEL = {
    0: '$DELETE',
    1: '$REPLACE',
    2: '$APPEND',
    3: '$SWAP_NEXT',
    4: '$SWAP_PREV',
    5: '$MERGE_NEXT',
    6: '$MERGE_PREV'
}

ERROR_LABEL2ID = {v: k for k, v in ERROR_ID2LABEL.items()}

# Optimal thresholds from your research
BINARY_THRESHOLD = 0.42

ERROR_TYPE_THRESHOLDS = {
    '$DELETE': 0.05,
    '$REPLACE': 0.51,
    '$APPEND': 0.45,
    '$SWAP_NEXT': 0.50,
    '$SWAP_PREV': 0.51,
    '$MERGE_NEXT': 0.05,
    '$MERGE_PREV': 0.05,
}

# Tags categorized by reliability (based on the evaluation)
RELIABLE_TAGS = {'$REPLACE', '$APPEND', '$SWAP_NEXT', '$SWAP_PREV'}
UNRELIABLE_TAGS = {'$DELETE', '$MERGE_NEXT', '$MERGE_PREV'}

# Pipeline parameters
MAX_CORRECTION_ITERATIONS = 10
MLM_TOP_K_SUGGESTIONS = 5
MAX_SEQUENCE_LENGTH = 128