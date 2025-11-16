# 📝 Nepali Grammar Error Correction (GEC) System

A state-of-the-art Nepali Grammar Error Correction system using fine-tuned RoBERTa models with semantic awareness and token-level error detection. [TRY HERE!](https://sudhaar-nepali-grammar-correction-rhqrzobqtavukjxj8xbedu.streamlit.app/)


[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-orange)](https://huggingface.co/DipeshChaudhary)

## 🎯 Overview

This project implements a comprehensive Nepali GEC system that goes beyond simple rule-based corrections by leveraging:
- **Semantic understanding** through contextualized embeddings
- **Token-level error detection** with optimized thresholds
- **Multi-model orchestration** for robust corrections
- **Confidence-based ranking** of suggestions

### Why This Approach?

Unlike previous Nepali GEC attempts (like Sumit Aryal's work using NepBERTa with basic MLM), this system:

1. **Uses IRIIS-RESEARCH's RoBERTa** (125M params) - proven to outperform NepBERTa on Nepali NLU tasks
2. **Implements 4 specialized models** instead of generic fine-tuning:
   - Sentence-level GED (Grammar Error Detection)
   - Binary token classifier (error pinpointing)
   - 7-class error type classifier (semantic categories)
   - Masked Language Model (contextual suggestions)
3. **Optimizes thresholds** through empirical research (e.g., 0.42 for binary detection)
4. **Categorizes tags by reliability** (4 reliable, 3 unreliable) based on F1 performance

## 🏗️ Architecture

### **4-Model Orchestration Pipeline**

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INPUT SENTENCE                                │
│                    "नाम मेरो दिपेश हो ।"                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           1. GED MODEL                                 │
│                    Sequence-Level Classification                       │
│                                                                         │
│  Input: Full sentence → RoBERTa → Binary: Correct/Incorrect            │
│                                                                         │
│  Output: {correct: false, confidence: 0.92}                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────┴───────────────┐
                    │                               │
               CORRECT?                         INCORRECT?
                    │                               │
                    ▼                               ▼
            ┌─────────────┐            ┌─────────────────────────────┐
            │   OUTPUT    │            │  2. BINARY TOKEN CLASSIFIER │
            │ CORRECTED   │            │    Token-Level Detection     │
            │ SENTENCE    │            │                              │
            └─────────────┘            │  Input: Token sequence       │
                                        │  → RoBERTa → Binary labels  │
                                        │                              │
                                        │  Threshold: 0.42            │
                                        │                              │
                                        │  Output: [ERROR, ERROR, KEEP]│
                                        └─────────────────────────────┘
                                                            │
                                                            ▼
                                        ┌─────────────────────────────┐
                                        │  3. ERROR TYPE CLASSIFIER    │
                                        │   Semantic Categorization    │
                                        │                              │
                                        │  Input: Error tokens         │
                                        │  → RoBERTa → 7-class logits  │
                                        │                              │
                                        │  Classes: DELETE, REPLACE,   │
                                        │          APPEND, SWAP_*,     │
                                        │          MERGE_*             │
                                        │                              │
                                        │  Reliability: 4 reliable,    │
                                        │              3 unreliable    │
                                        │                              │
                                        │  Output: [SWAP_NEXT@0.51]    │
                                        |          [SWAP_PREV@0.47]    |
                                        └─────────────────────────────┘
                                                            │
                                                            ▼
                                        ┌─────────────────────────────┐
                                        │     4. MLM MODEL            │
                                        │  Contextual Suggestions      │
                                        │                              │
                                        │  Input: Masked sentence      │
                                        │  → RoBERTa MLM → Top-k       │
                                        │                              │
                                        │  Strategies: Single/Double/  │
                                        │          Triple masking      │
                                        │                              │
                                        │  Output: [] (since swaps     │
                                        | doesn't want MLM suggestions)|
                                        └─────────────────────────────┘
                                                            │
                                                            ▼
                                        ┌─────────────────────────────┐
                                        │    VALIDATION LOOP           │
                                        │                              │
                                        │  Apply correction → GED check│
                                        │                              │
                                        │  If still incorrect:         │
                                        │  → Iterate (max 10 rounds)   │
                                        │                              │
                                        │  If correct: → Final output  │
                                        └─────────────────────────────┘
                                                            │
                                                            ▼
                                        ┌─────────────────────────────┐
                                        │     RANKED SUGGESTIONS       │
                                        │                              │
                                        │  By confidence score:        │
                                        │  1. "मेरो नाम दिपेश हो ।"   │
                                        │     (confidence: 0.87)       │
                                        │                              │
                                        │  2. Alternative corrections  │
                                        │     (if applicable)          │
                                        └─────────────────────────────┘
```

### The 4 Fine-Tuned Models

| Model | Purpose | Base Architecture | Key Innovation |
|-------|---------|-------------------|----------------|
| **GED** | Sentence-level correctness | Sequence Classification | Binary: correct/incorrect |
| **Binary Token** | Error localization | Token Classification | Optimized @ 0.42 threshold |
| **Error Type** | Semantic categorization | Token Classification | 7 error types (4 reliable) |
| **MLM** | Suggestion generation | Masked LM | Opcode-based masking |

## 📊 Error Type Classification

### 7 Error Types with Optimized Thresholds

| Error Type | Threshold | Reliability | Description |
|------------|-----------|-------------|-------------|
| `$DELETE` | 0.05 | ❌ Unreliable | Token should be removed |
| `$REPLACE` | 0.51 | ✅ Reliable | Token needs replacement |
| `$APPEND` | 0.45 | ✅ Reliable | Add token after |
| `$SWAP_NEXT` | 0.50 | ✅ Reliable | Swap with next token |
| `$SWAP_PREV` | 0.51 | ✅ Reliable | Swap with previous token |
| `$MERGE_NEXT` | 0.05 | ❌ Unreliable | Merge with next token |
| `$MERGE_PREV` | 0.05 | ❌ Unreliable | Merge with previous |

**Why some are unreliable?**
During evaluation, `$DELETE`, `$MERGE_NEXT`, and `$MERGE_PREV` showed poor F1 scores even at optimal thresholds. The system de-prioritizes these in the correction pipeline.

## 🚀 Key Features

### 1. **Semantic-Aware Corrections**
- Uses contextualized RoBERTa embeddings
- Understands sentence meaning, not just grammar rules
- Preserves semantic integrity

### 2. **Multi-Strategy MLM**
- **Opcode-based masking**: Intelligently masks based on error type
- **Confidence filtering**: Only suggests high-confidence replacements
- **Iterative refinement**: Up to 10 correction iterations

### 3. **Threshold Optimization**
- Binary detection: 0.42 (empirically determined)
- Per-error-type thresholds (0.05 to 0.51)
- Balanced precision-recall tradeoff

### 4. **Robust Validation**
- Dual validation (GED + Binary)
- Suggestion re-validation before output
- Confidence scoring for ranking

## 📈 Performance & Statistics

### Model Training

- **Base Model**: IRIIS-RESEARCH/RoBERTa_Nepali_125M
- **Training Data**: Sumit Aryal's dataset + augmentations
- **Preprocessing**: Token-level annotation with 10 initial tags → refined to 7
- **Training Setup**: See `notebooks/`

### Performance Metrics

#### GED Model (Sequence-Level)
- **Accuracy**: 0.9234
- **F1 Score**: 0.9156
- **Precision**: 0.9087
- **Recall**: 0.9226
- **Training Samples**: 10,082,804
- **Validation Samples**: 771,511

#### Binary Token Classifier
- **Test Accuracy**: 0.9874
- **Test F1 Score**: 0.8761
- **Test Precision**: 0.9344
- **Test Recall**: 0.8247
- **Test Sentence Accuracy**: 0.8831
- **Optimized Threshold**: 0.42

#### Error Type Classifier
- **Accuracy**: 0.950165
- **Macro F1 Score**: 0.634618
- **Training Epochs**: 3
- **Error Types**: 7 classes (DELETE, REPLACE, APPEND, SWAP_NEXT, SWAP_PREV, MERGE_NEXT, MERGE_PREV)

#### MLM Model
- **Fine-tuning Epochs**: 3
- **Batch Size**: 512
- **Learning Rate**: 5e-5
- **Max Sequence Length**: 128
- **Multi-strategy Masking**: Single, double, and triple mask approaches

### Threshold Optimization Results

You can see `notebooks/` experiments:

- **Binary Classifier**: Optimal F1 at threshold 0.42
- **Reliable Tags** (REPLACE, APPEND, SWAP_*): F1 > 0.75
- **Unreliable Tags** (DELETE, MERGE_*): F1 < 0.45

### Why IRIIS-RESEARCH over NepBERTa?

As shown in the NepGLUE paper benchmark:
- IRIIS-RESEARCH RoBERTa: **95.60** NepGLUE score
- NepBERTa: **93.55** NepGLUE score
- **+2 points improvement** on Nepali language understanding

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for faster inference

### Setup

```bash
# Clone repository

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Models

The fine-tuned models are hosted on HuggingFace Hub:

```python
# Models will auto-download on first run
# Or manually download:
from huggingface_hub import snapshot_download

snapshot_download(repo_id="DipeshChaudhary/roberta-nepali-sequence-ged")
snapshot_download(repo_id="DipeshChaudhary/nepali-gec-binary-detector")
snapshot_download(repo_id="DipeshChaudhary/nepali-gec-error-type-classifier")
snapshot_download(repo_id="DipeshChaudhary/nepali-mlm-guesser-finetuned-model-1")
```

## 🎮 Usage

### Streamlit Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Python API

```python
from src.model_loader import load_models
from src.inference import NepaliGECEngine

# Initialize
models = load_models()
engine = NepaliGECEngine(models)

# Correct a sentence
sentence = "नाम मेरो दिपेश हो ।"
result = engine.correct_sentence(sentence)

print(result['corrected'])
print(f"Confidence: {result['confidence']:.2%}")
print(f"Suggestions: {len(result['suggestions'])}")
```

### Batch Processing

```python
sentences = [
    "नाम मेरो दिपेश हो ।",
    "म स्कुल जान्छु ।",
    "यो किताब मेरो हो ।"
]

for sent in sentences:
    result = engine.correct_sentence(sent)
    print(f"Original: {sent}")
    print(f"Corrected: {result['corrected']}\n")
```

## 📁 Project Structure

```
nepali-semantic-gec/
├── app.py                          # Streamlit web interface
├── config.py                       # Configuration & hyperparameters
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── src/
│   ├── __init__.py
│   ├── model_loader.py            # Model loading with GPU/CPU support
│   ├── inference.py               # Core GEC orchestration engine
│   └── data_processing.py         # Preprocessing utilities
├── notebooks/
│   ├── 01-SEQUENCE-LEVEL/roberta-nepali-sequence-ged.ipynb  # GED model training
│   ├── 02-TOKEN - LEVEL/          # Token-level models
│   │   ├── 02-token-level-GED(binary)/  # Binary classifier
│   │   └── 03-error-type--gector-tag/   # Error type classifier
│   ├── 03-MLM-suggster/           # MLM fine-tuning
│   └── FINAL-orchestration-for-inference/  # Complete pipeline
├── models/
│   └── .gitkeep                   # Models downloaded at runtime
└── docs/
    └── model_architecture.md      # Detailed architecture docs
```

## 📚 Research Decisions

### Why IRIIS-RESEARCH RoBERTa vs NepBERTa (NepGLUE benchmark)
- **Superior Performance**: IRIIS-RESEARCH RoBERTa 125M outperforms NepBERTa by 2 points on Nep-gLUE benchmark (95.60 vs 93.55) [[1]](#references)
- **Larger Training Corpus**: Trained on 27.5 GB of Nepali text (2.4x larger than previous datasets) [[1]](#references)
- **Optimized Architecture**: Better tokenization and vocabulary coverage for Nepali language [[1]](#references)
- **State-of-the-Art Results**: Achieves highest scores across NER, POS, text classification, and categorical pair similarity tasks [[1]](#references)

### Why 4 Specialized Models vs Generic Fine-tuning
- **Task-Specific Optimization**: Each model specializes in different aspects of GEC (detection, classification, correction)
- **Threshold Optimization**: Enables precise control over error detection sensitivity (0.42 for binary, per-error-type thresholds)
- **Reliability Categorization**: 4 reliable error types get priority corrections, 3 unreliable types use conservative approach
- **Multi-Strategy MLM**: Single/double/triple masking strategies for contextual suggestions
- **Iterative Refinement**: Validation loop ensures corrections don't introduce new errors

### Why Threshold Optimization and Reliability Categorization
- **Precision vs Recall Trade-off**: Optimized thresholds balance false positives and false negatives
- **Error Type Reliability**: DELETE, REPLACE, APPEND, SWAP_* are reliable; MERGE_* are unreliable
- **Quality Control**: Prevents over-correction and maintains semantic accuracy
- **Performance Gains**: Significant improvements in F1 scores across error types

### Why Opcode-Based Masking for MLM
- **Contextual Awareness**: Masks tokens based on grammatical error patterns
- **Semantic Preservation**: Maintains sentence meaning while generating corrections
- **Diverse Suggestions**: Multiple masking strategies provide varied correction options
- **Validation Integration**: Works seamlessly with GED validation loop

## 📖 Dataset Details

### Nepali GEC Dataset (Sumit Aryal et al.)
Our work builds upon the foundational Nepali Grammatical Error Correction dataset created by Sumit Aryal et al. [[2]](#references). This dataset represents the first comprehensive parallel corpus for Nepali GEC, containing 8.1 million source-target pairs.

#### Dataset Creation Methodology
The dataset was created to address the lack of publicly available parallel corpus for Nepali GEC. Authors decided to use the already available `A Large Scale Nepali Text Corpus` [[5]](#references) as the base of raw nepali sentences.The authors identified five main types of Nepali grammatical errors and systematically generated them through data augmentation techniques.

##### Error Types Identified:
1. **Verb Inflection** (39.39%): Modification of verb forms that disrupt subject-verb agreement, honorific levels, tense, and number
2. **Homophones** (6.2%): Words that sound alike but have different meanings and spellings
3. **Punctuation** (12.84%): Incorrect use of symbols like commas, full stops, question marks, and exclamation marks
4. **Sentence Structure** (12.31%): Incorrect arrangement of words and phrases that changes sentence meaning
5. **Sentence Fragments** (28.89%): Incomplete sentences divided into:
   - **Pronoun Missing** (3.89%): Subject pronouns omitted causing ambiguity
   - **Main Verb Missing** (12.68%): Principal verbs that convey primary action/state omitted
   - **Auxiliary Verb Missing** (12.69%): Helping verbs that add tense/mood/voice omitted

##### Data Sourcing and Preprocessing:
- **Source**: "A LARGE SCALE NEPALI TEXT CORPUS" (Lamsal, 2020) [[5]](#references)
- **Filtering**: Sentences with 3-20 words, accounting for punctuation
- **Cleaning**: Removed non-Devanagari script characters, converted English numerals to Nepali numerals
- **Deduplication**: Extracted unique sentences to remove redundancy
- **Exclusion**: Sentences with only single parenthesis or quotes discarded

##### Data Augmentation Process:
The authors employed systematic noise injection techniques to generate grammatical errors:

1. **Verb Inflection Generation**:
   - Extracted verbs using POS Tagger [[4]](#references)
   - Applied hybrid lemmatizer to get root forms [[3]](#references)
   - Collected verb suffixes and grouped by similarity
   - Replaced verb suffixes with similar alternatives from predefined dictionaries

2. **Homophones Generation**:
   - Scraped homophone pairs from online sources [[2]](#references)
   - Manually added missing pairs
   - Created homophone dictionary for systematic replacement

3. **Punctuation Errors**:
   - Random probability-based removal/replacement of punctuation marks
   - Full stops, question marks, exclamation marks: removed or replaced with alternatives
   - Other symbols: removed with random probability

4. **Sentence Structure Errors**:
   - Random swapping of word positions within sentences

5. **Sentence Fragment Generation**:
   - **Pronoun Removal**: Used POS tags to identify and remove pronouns
   - **Verb Removal**: Distinguished main vs auxiliary verbs using linguistic rules
     - Main verbs: Final verbs completing sentence structure
     - Auxiliary verbs: Preceding helping verbs
   - Removed identified verbs to create missing verb errors

##### Dataset Statistics:
- **Total Error Instances**: 8,130,496 across 7 error types
- **Training Split**: 95% (7,723,971 pairs) - 2,568,682 correct, 7,514,122 incorrect sentences
- **Validation Split**: 5% (406,525 pairs) - 365,606 correct, 405,905 incorrect sentences
- **Error Distribution**:
  | Error Type | Instances | Percentage |
  |------------|-----------|------------|
  | Verb Inflection | 3,202,676 | 39.39% |
  | Punctuation | 1,044,203 | 12.84% |
  | Auxiliary Verb Missing | 1,031,388 | 12.69% |
  | Main Verb Missing | 1,031,274 | 12.68% |
  | Sentence Structure | 1,001,038 | 12.31% |
  | Homophones | 503,524 | 6.20% |
  | Pronouns | 316,393 | 3.89% |

##### Key Characteristics:
- **Synthetic Generation**: All errors generated automatically using linguistic tools, not manually introduced
- **Single Error per Sentence**: Each erroneous sentence contains exactly one grammatical mistake
- **Real-world Replication**: Error patterns designed to mimic common Nepali writing mistakes
- **Diversity**: Covers wide range of grammatical phenomena including subject-verb agreement, honorifics, and morphological variations

#### Our Improvements:
- **Superior GED Performance**: Our RoBERTa-based GED model achieves 92.34% accuracy vs their MuRIL's 91.15% and NepBERTa's 81.73% [[2]](#references)
- **Extended Architecture**: While they used 2-model approach (GED + MLM), we developed 4 specialized models
- **Advanced Techniques**: Threshold optimization, reliability categorization, and iterative validation

## 🔗 References

[1] Thapa, P., Nyachhyon, J., Sharma, M., & Bal, B. K. (2025). Development of Pre-Trained Transformer-based Models for the Nepali Language. arXiv preprint arXiv:2411.15734. https://arxiv.org/html/2411.15734

[2] Aryal, S., Jaiswal, A. (2024). BERT-Based Nepali Grammatical Error Detection and Correction Leveraging a New Corpus. In IEEE Xplore. https://ieeexplore.ieee.org/document/10896043

[3] D. P. D. L., “Nepali Lemmatizer,” 2020. https://github.com/dpakpdl/NepaliLemmatizer

[4] E. 911, “Nepali POS Tagger,” 2018. https://github.com/e911/Nepali-POS-Tagger

[5] R. Lamsal, “A Large Scale Nepali Text Corpus",IEEE Dataport, 2020.[Online]. Available: https://dx.doi.org/10.21227/jxrd-d245

## 📊 Evaluation Metrics

From `notebooks/`:

### Correction Accuracy
- **Sentence-level**: 87.3% (on test set)
- **Token-level**: 91.2% error detection
- **Suggestion quality**: 78.5% (validated corrections)

### Inference Speed (CPU)
- Average: 1.2s per sentence
- With GPU: 0.3s per sentence

### Model Sizes
- GED: 125M params
- Binary: 125M params
- Error Type: 125M params
- MLM: 125M params
- **Total**: ~500M params (~2GB disk)


## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Unreliable tags**: Better training for DELETE/MERGE operations
2. **Multi-sentence**: Extend to paragraph-level corrections
3. **Error explanations**: Add linguistic explanations for errors
4. **More error types**: Expand beyond 7 categories
5. **Speed optimization**: Model distillation for faster inference

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- **Sumit Aryal**: Base dataset and initial GEC exploration
- **IRIIS-RESEARCH**: Pre-trained RoBERTa model for Nepali
- **NepGLUE**: Benchmark for model selection
- **HuggingFace**: Transformers library and model hosting

## 📚 References

1. Timilsina et al. (2022) - NepBERTa: Nepali Language Model
2. IRIIS-RESEARCH - RoBERTa Nepali 125M
3. Aryal et al. - Nepali GEC Dataset
4. NepGLUE Benchmark paper

---

## 📊 Model Hub Links

All fine-tuned models are available on HuggingFace Hub:

- **GED Model**: [DipeshChaudhary/roberta-nepali-sequence-ged](https://huggingface.co/DipeshChaudhary/roberta-nepali-sequence-ged)
- **Binary Token Classifier**: [DipeshChaudhary/nepali-gec-binary-detector](https://huggingface.co/DipeshChaudhary/nepali-gec-binary-detector)
- **Error Type Classifier**: [DipeshChaudhary/nepali-gec-error-type-classifier](https://huggingface.co/DipeshChaudhary/nepali-gec-error-type-classifier)
- **MLM Model**: [DipeshChaudhary/nepali-mlm-guesser-finetuned-model-1](https://huggingface.co/DipeshChaudhary/nepali-mlm-guesser-finetuned-model-1)

