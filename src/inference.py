"""
Complete inference orchestration
"""
import torch
import re
from typing import List, Tuple, Dict, Any
from config import *

class NepaliGECEngine:
    def __init__(self, models_dict):
        self.device = models_dict['device']
        self.ged_tokenizer = models_dict['ged_tokenizer']
        self.gector_tokenizer = models_dict['gector_tokenizer']
        self.ged_model = models_dict['ged_model']
        self.binary_model = models_dict['binary_model']
        self.error_model = models_dict['error_model']
        self.mlm_model = models_dict['mlm_model']
    
    def preprocess_sentence(self, sentence: str) -> str:
        """Separate punctuation for better token alignment"""
        sentence = re.sub(r'([।?,!])', r' \1', sentence)
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        return sentence
    
    def postprocess_sentence(self, tokens: List[str]) -> str:
        """Re-merge separated punctuation"""
        sentence = " ".join(tokens)
        sentence = re.sub(r'\s+([।?,!])', r'\1', sentence).strip()
        return sentence
    
    def check_sentence_correctness(self, sentence: str) -> Tuple[bool, float]:
        """Check if sentence is correct using GED model"""
        inputs = self.ged_tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.ged_model(**inputs).logits
        
        probs = torch.softmax(logits, dim=-1)[0]
        is_correct = torch.argmax(probs).item() == 0
        confidence = probs[0 if is_correct else 1].item()
        
        return is_correct, confidence
    
    def validate_with_binary_model(self, sentence: str) -> Tuple[bool, List[int]]:
        """Validate using token-level binary classifier"""
        inputs = self.gector_tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            binary_logits = self.binary_model(**inputs).logits
        
        binary_probs = torch.softmax(binary_logits, dim=-1)[0]
        word_ids = inputs.word_ids(batch_index=0)
        
        word_start_token_map = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_start_token_map:
                word_start_token_map[word_idx] = token_idx
        
        error_positions = []
        for word_idx in sorted(word_start_token_map.keys()):
            token_idx = word_start_token_map[word_idx]
            binary_error_prob = binary_probs[token_idx, 1].item()
            
            if binary_error_prob > BINARY_THRESHOLD:
                error_positions.append(word_idx)
        
        return len(error_positions) == 0, error_positions
    
    def get_token_predictions(self, sentence: str) -> List[Dict[str, Any]]:
        """Get comprehensive token-level predictions"""
        inputs = self.gector_tokenizer(
            sentence, 
            return_tensors="pt", 
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)
        
        with torch.no_grad():
            binary_logits = self.binary_model(**inputs).logits
            error_logits = self.error_model(**inputs).logits
        
        binary_probs = torch.softmax(binary_logits, dim=-1)[0]
        error_probs = torch.softmax(error_logits, dim=-1)[0]
        
        word_ids = inputs.word_ids(batch_index=0)
        word_start_token_map = {}
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in word_start_token_map:
                word_start_token_map[word_idx] = token_idx
        
        tokens = sentence.split()
        predictions = []
        
        for word_idx in sorted(word_start_token_map.keys()):
            if word_idx >= len(tokens):
                continue
            
            token_idx = word_start_token_map[word_idx]
            token = tokens[word_idx]
            
            binary_error_prob = binary_probs[token_idx, 1].item()
            
            error_type_probs = {}
            for error_id, error_tag in ERROR_ID2LABEL.items():
                error_type_probs[error_tag] = error_probs[token_idx, error_id].item()
            
            candidate_tags = []
            for tag, prob in error_type_probs.items():
                threshold = ERROR_TYPE_THRESHOLDS[tag]
                if prob > threshold:
                    candidate_tags.append((tag, prob))
            
            candidate_tags.sort(key=lambda x: x[1], reverse=True)
            
            is_error = binary_error_prob > BINARY_THRESHOLD
            primary_tag = candidate_tags[0][0] if candidate_tags else None
            
            pred = {
                'word_idx': word_idx,
                'token': token,
                'binary_error_prob': binary_error_prob,
                'error_type_probs': error_type_probs,
                'candidate_tags': candidate_tags,
                'primary_tag': primary_tag,
                'is_error': is_error,
                'is_reliable': primary_tag in RELIABLE_TAGS if primary_tag else False
            }
            
            predictions.append(pred)
        
        return predictions
    
    def get_mlm_suggestions(self, sentence: str, mask_positions: List[int], 
                           k: int = MLM_TOP_K_SUGGESTIONS) -> List[List[str]]:
        """Get MLM suggestions with multi-mask strategy"""
        if not mask_positions:
            return []
        
        token_list = sentence.split()
        valid_positions = [pos for pos in mask_positions if 0 <= pos <= len(token_list)]
        
        if not valid_positions:
            return []
        
        all_suggestions_per_position = []
        
        for pos in valid_positions:
            position_suggestions = []
            
            # Single mask strategy
            single_mask_suggestions = self._get_single_mask_suggestions(token_list, pos, k)
            position_suggestions.extend(single_mask_suggestions)
            
            # Remove duplicates
            seen = set()
            unique_suggestions = []
            for sugg in position_suggestions:
                if sugg and sugg not in seen:
                    seen.add(sugg)
                    unique_suggestions.append(sugg)
                    if len(unique_suggestions) >= k:
                        break
            
            all_suggestions_per_position.append(unique_suggestions[:k])
        
        return all_suggestions_per_position
    
    def _get_single_mask_suggestions(self, token_list: List[str], 
                                    position: int, k: int) -> List[str]:
        """Get suggestions with a single [MASK] token"""
        temp_list = list(token_list)
        
        if position < len(temp_list):
            temp_list[position] = self.gector_tokenizer.mask_token
        elif position == len(temp_list):
            temp_list.append(self.gector_tokenizer.mask_token)
        else:
            return []
        
        masked_sentence = " ".join(temp_list)
        return self._predict_masked_tokens(masked_sentence, k)
    
    def _predict_masked_tokens(self, masked_sentence: str, k: int) -> List[str]:
        """Core MLM prediction function"""
        inputs = self.gector_tokenizer(
            masked_sentence, 
            return_tensors='pt', 
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)
        
        mask_token_indices = torch.where(
            inputs["input_ids"] == self.gector_tokenizer.mask_token_id
        )[1]
        
        if mask_token_indices.numel() == 0:
            return []
        
        with torch.no_grad():
            logits = self.mlm_model(**inputs).logits
        
        suggestions = []
        
        for mask_idx in mask_token_indices:
            mask_logits = logits[0, mask_idx, :]
            top_k_tokens = torch.topk(mask_logits, k * 3).indices.tolist()
            
            for token_id in top_k_tokens:
                decoded = self.gector_tokenizer.decode(token_id).strip()
                
                if self._is_valid_token(decoded):
                    suggestions.append(decoded)
                
                if len(suggestions) >= k:
                    break
            
            if suggestions:
                break
        
        return suggestions[:k]
    
    def _is_valid_token(self, token: str) -> bool:
        """Check if a token is valid for use as a correction"""
        if not token or len(token) < 1:
            return False
        
        invalid_tokens = {'(', ')', '[', ']', '{', '}', '<', '>', '"', "'", '`', ',', '.'}
        if token in invalid_tokens:
            return False
        
        if token.startswith('##'):
            return False
        
        return True
    
    def correct_sentence(self, sentence: str) -> Dict[str, Any]:
        """
        Complete GEC pipeline
        Returns: dict with 'corrected', 'is_correct', 'suggestions', 'iterations'
        """
        preprocessed = self.preprocess_sentence(sentence)
        
        # Check if already correct
        is_ged_correct, ged_conf = self.check_sentence_correctness(preprocessed)
        is_binary_correct, error_positions = self.validate_with_binary_model(preprocessed)
        
        if is_ged_correct and is_binary_correct:
            return {
                'original': sentence,
                'corrected': sentence,
                'is_correct': True,
                'confidence': ged_conf,
                'suggestions': [],
                'iterations': 0,
                'message': 'Sentence is already correct'
            }
        
        # Get error predictions
        predictions = self.get_token_predictions(preprocessed)
        error_predictions = [p for p in predictions if p['is_error']]
        
        if not error_predictions:
            return {
                'original': sentence,
                'corrected': sentence,
                'is_correct': False,
                'confidence': ged_conf,
                'suggestions': [],
                'iterations': 0,
                'message': 'No errors detected by binary model but GED flagged as incorrect'
            }
        
        # Generate corrections
        tokens = preprocessed.split()
        error_positions = [p['word_idx'] for p in error_predictions]
        suggestions_list = self.get_mlm_suggestions(preprocessed, error_positions)
        
        validated_suggestions = []
        
        for pos_idx, (pos, suggestions) in enumerate(zip(error_positions, suggestions_list)):
            for sugg in suggestions:
                # Create candidate sentence
                test_tokens = list(tokens)
                if pos < len(test_tokens):
                    test_tokens[pos] = sugg
                    candidate = " ".join(test_tokens)
                    
                    # Validate candidate
                    is_correct, conf = self.check_sentence_correctness(candidate)
                    if is_correct:
                        validated_suggestions.append({
                            'sentence': self.postprocess_sentence(test_tokens),
                            'position': pos,
                            'original_word': tokens[pos],
                            'corrected_word': sugg,
                            'confidence': conf
                        })
        
        # Sort by confidence
        validated_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        if validated_suggestions:
            best_correction = validated_suggestions[0]
            return {
                'original': sentence,
                'corrected': best_correction['sentence'],
                'is_correct': False,
                'confidence': best_correction['confidence'],
                'suggestions': validated_suggestions[:3],  # Top 3
                'iterations': 1,
                'message': f"Corrected {len(error_predictions)} error(s)"
            }
        else:
            return {
                'original': sentence,
                'corrected': sentence,
                'is_correct': False,
                'confidence': ged_conf,
                'suggestions': [],
                'iterations': 1,
                'message': 'Could not generate valid corrections'
            }