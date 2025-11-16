"""
Data preprocessing utilities for Nepali GEC
Includes text normalization, tokenization helpers, and edit operation utilities
"""
import re
from typing import List, Tuple, Dict
from difflib import SequenceMatcher


class NepaliTextProcessor:
    """
    Preprocessing utilities for Nepali text
    """
    
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize Devanagari Unicode characters
        Handles common unicode normalization issues in Nepali
        """
        import unicodedata
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        return text
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """Remove multiple spaces, tabs, newlines"""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def separate_punctuation(text: str) -> str:
        """
        Separate Nepali punctuation from words for better tokenization
        This matches the preprocessing in your training data
        """
        # Nepali full stop (।), comma, question mark, exclamation
        text = re.sub(r'([।?,!])', r' \1', text)
        return text
    
    @staticmethod
    def merge_punctuation(text: str) -> str:
        """
        Merge separated punctuation back to words
        Inverse operation of separate_punctuation
        """
        text = re.sub(r'\s+([।?,!])', r'\1', text)
        return text
    
    @staticmethod
    def is_nepali_text(text: str) -> bool:
        """
        Check if text contains Nepali (Devanagari) characters
        """
        # Devanagari Unicode range: U+0900 to U+097F
        nepali_pattern = re.compile(r'[\u0900-\u097F]')
        return bool(nepali_pattern.search(text))
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Complete cleaning pipeline
        """
        text = NepaliTextProcessor.normalize_unicode(text)
        text = NepaliTextProcessor.remove_extra_whitespace(text)
        return text
    
    @staticmethod
    def preprocess_for_model(text: str) -> str:
        """
        Preprocessing pipeline before feeding to model
        Matches your training preprocessing
        """
        text = NepaliTextProcessor.clean_text(text)
        text = NepaliTextProcessor.separate_punctuation(text)
        return text
    
    @staticmethod
    def postprocess_from_model(text: str) -> str:
        """
        Postprocessing pipeline after model output
        """
        text = NepaliTextProcessor.merge_punctuation(text)
        text = NepaliTextProcessor.remove_extra_whitespace(text)
        return text


class EditOperations:
    """
    Utilities for computing edit operations between sentences
    Used for generating training data and analyzing corrections
    """
    
    @staticmethod
    def get_opcodes(source: str, target: str) -> List[Tuple[str, int, int, int, int]]:
        """
        Get edit operations (opcodes) between source and target
        Uses SequenceMatcher from difflib
        
        Returns list of (tag, i1, i2, j1, j2):
        - 'replace': source[i1:i2] should be replaced by target[j1:j2]
        - 'delete':  source[i1:i2] should be deleted
        - 'insert':  target[j1:j2] should be inserted at source[i1:i1]
        - 'equal':   source[i1:i2] == target[j1:j2]
        """
        source_tokens = source.split()
        target_tokens = target.split()
        
        matcher = SequenceMatcher(None, source_tokens, target_tokens)
        return matcher.get_opcodes()
    
    @staticmethod
    def opcodes_to_edits(opcodes: List[Tuple], source_tokens: List[str]) -> List[Dict]:
        """
        Convert opcodes to structured edit information
        
        Returns list of dicts with:
        - position: word position in source
        - operation: type of edit
        - source_text: original text
        - target_text: corrected text
        """
        edits = []
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                continue
            
            edit = {
                'operation': tag,
                'source_start': i1,
                'source_end': i2,
                'target_start': j1,
                'target_end': j2,
                'source_text': ' '.join(source_tokens[i1:i2]) if i1 < i2 else '',
            }
            
            edits.append(edit)
        
        return edits
    
    @staticmethod
    def compute_edit_distance(source: str, target: str) -> int:
        """
        Compute Levenshtein distance at token level
        """
        source_tokens = source.split()
        target_tokens = target.split()
        
        m, n = len(source_tokens), len(target_tokens)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if source_tokens[i-1] == target_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]
    
    @staticmethod
    def highlight_differences(source: str, target: str) -> Tuple[str, str]:
        """
        Create highlighted version of source and target showing differences
        Returns (highlighted_source, highlighted_target)
        """
        opcodes = EditOperations.get_opcodes(source, target)
        source_tokens = source.split()
        target_tokens = target.split()
        
        highlighted_source = []
        highlighted_target = []
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                highlighted_source.extend(source_tokens[i1:i2])
                highlighted_target.extend(target_tokens[j1:j2])
            elif tag == 'replace':
                highlighted_source.extend([f"[{w}]" for w in source_tokens[i1:i2]])
                highlighted_target.extend([f"[{w}]" for w in target_tokens[j1:j2]])
            elif tag == 'delete':
                highlighted_source.extend([f"[-{w}]" for w in source_tokens[i1:i2]])
            elif tag == 'insert':
                highlighted_target.extend([f"[+{w}]" for w in target_tokens[j1:j2]])
        
        return ' '.join(highlighted_source), ' '.join(highlighted_target)


class TokenLevelAnnotation:
    """
    Utilities for token-level annotation as used in your training data
    """
    
    # The 7 error tags from your classification model
    ERROR_TAGS = [
        '$DELETE',
        '$REPLACE', 
        '$APPEND',
        '$SWAP_NEXT',
        '$SWAP_PREV',
        '$MERGE_NEXT',
        '$MERGE_PREV'
    ]
    
    @staticmethod
    def generate_token_labels(source: str, target: str) -> List[Tuple[str, str]]:
        """
        Generate token-level labels from source and target sentences
        
        Returns list of (token, label) pairs
        This matches the annotation strategy in your training data
        """
        opcodes = EditOperations.get_opcodes(source, target)
        source_tokens = source.split()
        target_tokens = target.split()
        
        labels = ['$KEEP'] * len(source_tokens)
        
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                continue
            elif tag == 'delete':
                for i in range(i1, i2):
                    labels[i] = '$DELETE'
            elif tag == 'replace':
                for i in range(i1, i2):
                    labels[i] = '$REPLACE'
            elif tag == 'insert':
                if i1 > 0:
                    labels[i1-1] = '$APPEND'
        
        return list(zip(source_tokens, labels))
    
    @staticmethod
    def detect_swap_operations(source: str, target: str) -> List[int]:
        """
        Detect positions where tokens should be swapped
        """
        source_tokens = source.split()
        target_tokens = target.split()
        
        swap_positions = []
        
        for i in range(len(source_tokens) - 1):
            # Check if adjacent tokens are swapped
            if (i + 1 < len(target_tokens) and 
                source_tokens[i] == target_tokens[i+1] and
                source_tokens[i+1] == target_tokens[i]):
                swap_positions.append(i)
        
        return swap_positions
    
    @staticmethod
    def create_masked_sentences(sentence: str, mask_token: str = "[MASK]") -> List[str]:
        """
        Create masked versions of sentence for MLM prediction
        This is the strategy used in your inference notebook
        
        Creates:
        1. Mask each word individually
        2. Mask each position between words (for insertions)
        """
        tokens = sentence.split()
        masked_sentences = []
        
        # Strategy 1: Mask each word
        for i in range(len(tokens)):
            masked = tokens.copy()
            masked[i] = mask_token
            masked_sentences.append(' '.join(masked))
        
        # Strategy 2: Mask positions (for insertions)
        for i in range(len(tokens) + 1):
            masked = tokens.copy()
            masked.insert(i, mask_token)
            masked_sentences.append(' '.join(masked))
        
        return masked_sentences


class DatasetUtils:
    """
    Utilities for working with your training dataset format
    """
    
    @staticmethod
    def parse_token_classification_format(data: Dict) -> List[Dict]:
        """
        Parse token classification dataset format
        Expects format with 'tokens' and 'tags' columns
        """
        parsed_data = []
        
        for i in range(len(data['tokens'])):
            tokens = data['tokens'][i]
            tags = data['tags'][i]
            
            parsed_data.append({
                'tokens': tokens,
                'tags': tags,
                'sentence': ' '.join(tokens)
            })
        
        return parsed_data
    
    @staticmethod
    def convert_binary_labels(tags: List[str]) -> List[int]:
        """
        Convert multi-class tags to binary (error/no-error)
        Used for your binary token classifier
        """
        binary = []
        for tag in tags:
            if tag == '$KEEP' or tag == 'O':
                binary.append(0)  # No error
            else:
                binary.append(1)  # Error
        return binary
    
    @staticmethod
    def map_to_error_types(tags: List[str]) -> List[int]:
        """
        Map tags to 7-class error type IDs
        Matches your ERROR_ID2LABEL in config.py
        """
        tag_to_id = {
            '$DELETE': 0,
            '$REPLACE': 1,
            '$APPEND': 2,
            '$SWAP_NEXT': 3,
            '$SWAP_PREV': 4,
            '$MERGE_NEXT': 5,
            '$MERGE_PREV': 6
        }
        
        mapped = []
        for tag in tags:
            mapped.append(tag_to_id.get(tag, -1))  # -1 for $KEEP or unknown
        
        return mapped


# Utility functions for common operations
def preprocess_sentence(sentence: str) -> str:
    """Quick preprocessing for inference"""
    processor = NepaliTextProcessor()
    return processor.preprocess_for_model(sentence)


def postprocess_sentence(sentence: str) -> str:
    """Quick postprocessing after inference"""
    processor = NepaliTextProcessor()
    return processor.postprocess_from_model(sentence)


def compute_alignment(source: str, target: str) -> Dict:
    """
    Compute complete alignment information between source and target
    """
    ops = EditOperations()
    
    opcodes = ops.get_opcodes(source, target)
    edit_distance = ops.compute_edit_distance(source, target)
    highlighted = ops.highlight_differences(source, target)
    
    return {
        'opcodes': opcodes,
        'edit_distance': edit_distance,
        'highlighted_source': highlighted[0],
        'highlighted_target': highlighted[1]
    }


if __name__ == "__main__":
    # Test preprocessing
    test_sentence = "नाम   मेरो  दिपेश हो  ।"
    
    print("Original:", repr(test_sentence))
    print("Preprocessed:", preprocess_sentence(test_sentence))
    print("Postprocessed:", postprocess_sentence(test_sentence))
    
    # Test edit operations
    source = "नाम मेरो दिपेश हो ।"
    target = "मेरो नाम दिपेश हो ।"
    
    alignment = compute_alignment(source, target)
    print("\nAlignment Info:")
    print("Edit Distance:", alignment['edit_distance'])
    print("Source:", alignment['highlighted_source'])
    print("Target:", alignment['highlighted_target'])
    
    # Test masking
    masked = TokenLevelAnnotation.create_masked_sentences("यो किताब मेरो हो")
    print(f"\nGenerated {len(masked)} masked sentences")
    print("First 3:", masked[:3])