"""
Streamlit UI for Nepali Grammar Error Correction
"""
import streamlit as st
import torch
from src.model_loader import load_models
from src.inference import NepaliGECEngine

# Page config
st.set_page_config(
    page_title="Nepali GEC - Semantic-Aware Grammar Correction",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stat-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .correct-text {
        color: #28a745;
        font-weight: bold;
    }
    .incorrect-text {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">📝 Nepali Grammar Error Correction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Semantic-Aware GEC using RoBERTa-based Models</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This system uses **4 fine-tuned RoBERTa models** trained on IRIIS-RESEARCH's base model:
    
    1. **GED Model**: Sentence-level correctness detection
    2. **Binary Token Classifier**: Pinpoints erroneous tokens (threshold: 0.42)
    3. **Error Type Classifier**: Identifies 7 error types
    4. **MLM Model**: Generates correction suggestions
    
    **Key Features:**
    - Semantic-aware corrections
    - Token-level error detection
    - Multi-strategy MLM suggestions
    - Confidence-based ranking
    """)
    
    st.divider()
    
    # Device info
    device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Running on: **{device}**")
    
    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    
    st.divider()
    
    st.header("📊 Model Statistics")
    st.markdown("""
    **Training Data:**
    - Base: Sumit Aryal's dataset
    - Model: IRIIS-RESEARCH RoBERTa (125M params)
    
    **Performance:**
    - Binary Detection: F1 optimized at 0.42
    - Error Types: 7 classes (4 reliable, 3 unreliable)
    - Reliable Tags: REPLACE, APPEND, SWAP_NEXT, SWAP_PREV
    """)

# Load models
@st.cache_resource
def initialize_engine():
    models = load_models()
    return NepaliGECEngine(models)

with st.spinner("Loading models... (This may take a minute on first run)"):
    engine = initialize_engine()

st.success("✅ Models loaded successfully!")

# Main interface
st.header("🔍 Grammar Correction")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    input_text = st.text_area(
        "Enter Nepali sentence:",
        height=150,
        placeholder="नेपाली वाक्य यहाँ लेख्नुहोस्...",
        help="Enter a Nepali sentence to check for grammatical errors"
    )
    
    check_button = st.button("🔍 Check Grammar", type="primary", use_container_width=True)

with col2:
    st.subheader("Output")
    output_placeholder = st.empty()

# Processing
if check_button and input_text.strip():
    with st.spinner("Analyzing sentence..."):
        result = engine.correct_sentence(input_text.strip())
    
    # Display results
    with output_placeholder.container():
        if result['is_correct']:
            st.success("✅ Sentence is grammatically correct!")
            st.markdown(f"**Confidence:** {result['confidence']:.2%}")
        else:
            st.warning("⚠️ Grammatical errors detected")
            
            # Corrected sentence
            st.markdown("### Corrected Sentence")
            st.info(result['corrected'])
            
            st.markdown(f"**Confidence:** {result['confidence']:.2%}")
            st.markdown(f"**Message:** {result['message']}")
            
            # Alternative suggestions
            if result['suggestions']:
                st.markdown("### Alternative Suggestions")
                for idx, sugg in enumerate(result['suggestions'], 1):
                    with st.expander(f"Suggestion {idx} (Confidence: {sugg['confidence']:.2%})"):
                        st.write(sugg['sentence'])
                        st.caption(f"Changed: '{sugg['original_word']}' → '{sugg['corrected_word']}' at position {sugg['position']}")

# Examples
st.divider()
st.header("📚 Examples")

examples = [
    "नाम मेरो दिपेश हो ।",
    "म स्कुल जान्छु ।",
    "यो किताब मेरो हो ।",
    "डोजर पनि माटोले छोपं गएको छ ।",
    "कार्यालयमा जडान गरिएका आधा दर्जनभन्दा बढी क्यामेराले चौबिसै घन्टा काम गर्न सक्ी सौर्य ऊर्जामा जडान गरिएको हो।",
    "तालिमको लागि बोलाउँदा पनि नजाने मुडमा ढुक्क भएर बसिरहेको थिएँ ।",
    "दुर्घटनाबाजा बजाएर आउँदैन मानिसकै ससानोगल्तीले हुने हो ।",
    "भीमदत्त नगरपालिकापछि प्रदेश सरकारको सबैभन्दा धेरै बजेट कृष्णपुर नगरपालिकामा परेको ।",
    "समृद्धिका लागि आर्थिक पातो नै मुख्य हो ।",
    "मेरो नाम दिपेश हो"
]

st.write("Try these example sentences:")
cols = st.columns(len(examples))
for idx, (col, example) in enumerate(zip(cols, examples)):
    with col:
        if st.button(f"Example {idx+1}", use_container_width=True):
            st.session_state.example_text = example

if 'example_text' in st.session_state:
    st.info(f"Selected: {st.session_state.example_text}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Semantic-Aware Nepali GEC System</strong></p>
    <p>Built with 🔥 using RoBERTa, Transformers & Streamlit</p>
    <p><em>Note: This is a research prototype. Corrections may not always be perfect.</em></p>
</div>
""", unsafe_allow_html=True)