"""
Complete inference orchestration
(Replicated from inference-final-orchestrations.ipynb)
"""
import torch
import re
from typing import List, Tuple, Dict, Any
from copy import deepcopy
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

    # ===========================================================================
    # PREPROCESSING & POSTPROCESSING
    # ===========================================================================

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

    # ===========================================================================
    # GED & BINARY VALIDATION
    # ===========================================================================

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
        # ID 0 is correct, ID 1 is incorrect
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
        
        all_correct = len(error_positions) == 0
        return all_correct, error_positions

    def is_sentence_fully_correct(self, sentence: str) -> Tuple[bool, str]:
        """Validate sentence using BOTH GED and Binary models."""
        ged_correct, ged_conf = self.check_sentence_correctness(sentence)
        binary_correct, error_positions = self.validate_with_binary_model(sentence)
        
        if ged_correct and binary_correct:
            return True, f"✓ Both models agree (GED: {ged_conf:.3f})"
        elif not ged_correct and not binary_correct:
            return False, f"✗ Both models detect errors (GED: {ged_conf:.3f}, Binary: {len(error_positions)} errors)"
        elif not ged_correct:
            return False, f"✗ GED detects error (conf: {ged_conf:.3f})"
        else: # (binary_correct is False)
            return False, f"✗ Binary detects {len(error_positions)} error(s) at positions: {error_positions}"

    # ===========================================================================
    # MLM GUESSING (Multi-token strategy from notebook)
    # ===========================================================================

    def get_mlm_suggestions(self, sentence: str, mask_positions: List[int], 
                           k: int = MLM_TOP_K_SUGGESTIONS, verbose: bool = False) -> List[List[str]]:
        """Get MLM suggestions with multi-mask strategy."""
        if not mask_positions:
            return []
        
        token_list = sentence.split()
        valid_positions = [pos for pos in mask_positions if 0 <= pos <= len(token_list)]
        if not valid_positions:
            return []
        
        all_suggestions_per_position = []
        
        for pos in valid_positions:
            position_suggestions = []
            
            # Strategy 1: Single mask
            single_mask_suggestions = self._get_single_mask_suggestions(token_list, pos, k)
            position_suggestions.extend(single_mask_suggestions)
            
            # Strategy 2: Double mask
            double_mask_suggestions = self._get_multi_mask_suggestions(token_list, pos, num_masks=2, k=k)
            position_suggestions.extend(double_mask_suggestions)
            
            # Strategy 3: Triple mask
            triple_mask_suggestions = self._get_multi_mask_suggestions(token_list, pos, num_masks=3, k=k)
            position_suggestions.extend(triple_mask_suggestions)
            
            # Remove duplicates and keep top-k
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

    def _get_single_mask_suggestions(self, token_list: List[str], position: int, k: int) -> List[str]:
        """Get suggestions with a single [MASK] token."""
        temp_list = list(token_list)
        
        if position < len(temp_list):
            temp_list[position] = self.gector_tokenizer.mask_token
        elif position == len(temp_list):
            temp_list.append(self.gector_tokenizer.mask_token)
        else:
            return []
        
        masked_sentence = " ".join(temp_list)
        return self._predict_masked_tokens(masked_sentence, k)

    def _get_multi_mask_suggestions(self, token_list: List[str], position: int, num_masks: int, k: int) -> List[str]:
        """Get suggestions by placing multiple consecutive [MASK] tokens."""
        temp_list = list(token_list)
        
        if position < len(temp_list):
            temp_list[position:position+1] = [self.gector_tokenizer.mask_token] * num_masks
        elif position == len(temp_list):
            temp_list.extend([self.gector_tokenizer.mask_token] * num_masks)
        else:
            return []
        
        masked_sentence = " ".join(temp_list)
        return self._predict_masked_tokens(masked_sentence, k, expect_multi_token=True)

    def _predict_masked_tokens(self, masked_sentence: str, k: int, expect_multi_token: bool = False) -> List[str]:
        """Core MLM prediction function (from notebook)."""
        inputs = self.gector_tokenizer(
            masked_sentence, 
            return_tensors='pt', 
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        ).to(self.device)
        mask_token_indices = torch.where(inputs["input_ids"] == self.gector_tokenizer.mask_token_id)[1]
        
        if mask_token_indices.numel() == 0:
            return []
        
        with torch.no_grad():
            logits = self.mlm_model(**inputs).logits
        
        suggestions = []
        
        if expect_multi_token and mask_token_indices.numel() > 1:
            num_masks = mask_token_indices.numel()
            first_mask_logits = logits[0, mask_token_indices[0], :]
            top_first = torch.topk(first_mask_logits, min(k * 2, 50)).indices.tolist()
            
            for first_token_id in top_first:
                first_decoded = self.gector_tokenizer.decode(first_token_id).strip()
                
                if not self._is_valid_token(first_decoded):
                    continue
                
                remaining_parts = []
                for mask_idx in mask_token_indices[1:]:
                    mask_logits = logits[0, mask_idx, :]
                    top_token_id = torch.argmax(mask_logits).item()
                    decoded = self.gector_tokenizer.decode(top_token_id).strip()
                    
                    if decoded.startswith('##'):
                        decoded = decoded[2:]
                    
                    if decoded and len(decoded) > 0:
                        remaining_parts.append(decoded)
                
                if remaining_parts:
                    full_word = first_decoded + "".join(remaining_parts)
                else:
                    full_word = first_decoded
                
                if full_word and len(full_word) > 0:
                    suggestions.append(full_word)
                
                if len(suggestions) >= k:
                    break
        else:
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
        """Check if a token is valid for use as a correction."""
        if not token or len(token) < 1:
            return False
        
        invalid_tokens = {'(', ')', '[', ']', '{', '}', '<', '>', '"', "'", '`', ',', '.'}
        if token in invalid_tokens:
            return False
        
        if token.startswith('##'):
            return False
        
        return True

    def apply_mlm_to_positions(self, tokens: List[str], positions: List[int], suggestions: List[List[str]]) -> str:
        """Apply MLM suggestions to specific positions."""
        result = list(tokens)
        
        for pos, sugg_list in zip(positions, suggestions):
            if sugg_list and 0 <= pos < len(result):
                suggestion = sugg_list[0]
                if self._is_valid_token(suggestion):
                    result[pos] = suggestion
        
        return " ".join(result)

    # ===========================================================================
    # TOKEN CLASSIFICATION & ERROR DETECTION
    # ===========================================================================

    def get_token_predictions(self, sentence: str, verbose: bool = False) -> List[Dict[str, Any]]:
        """Get comprehensive token-level predictions."""
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

    # ===========================================================================
    # CORRECTION STRATEGIES (from notebook)
    # ===========================================================================

    def apply_tag_based_correction(self, tokens: List[str], corrections: List[Dict[str, Any]], 
                                    use_mlm_for_unreliable: bool = False) -> str:
        """Apply corrections based on predicted tags with proper phase handling."""
        result_tokens = list(tokens)
        valid_corrections = [c for c in corrections if c['primary_tag'] is not None]
        
        if not valid_corrections:
            return " ".join(result_tokens)
        
        swaps = []
        deletes_merges = []
        replaces_appends = []
        
        for corr in valid_corrections:
            tag = corr['primary_tag']
            if tag in ['$SWAP_NEXT', '$SWAP_PREV']:
                swaps.append(corr)
            elif tag in ['$DELETE', '$MERGE_NEXT', '$MERGE_PREV']:
                deletes_merges.append(corr)
            else:
                replaces_appends.append(corr)
        
        # PHASE 1: Swaps
        for corr in swaps:
            idx = corr['word_idx']
            tag = corr['primary_tag']
            
            if tag == '$SWAP_NEXT' and idx + 1 < len(result_tokens):
                result_tokens[idx], result_tokens[idx + 1] = result_tokens[idx + 1], result_tokens[idx]
            elif tag == '$SWAP_PREV' and idx > 0:
                result_tokens[idx], result_tokens[idx - 1] = result_tokens[idx - 1], result_tokens[idx]
        
        # PHASE 2: Deletions and Merges
        for corr in sorted(deletes_merges, key=lambda x: x['word_idx'], reverse=True):
            idx = corr['word_idx']
            tag = corr['primary_tag']
            
            if idx >= len(result_tokens):
                continue
            
            if use_mlm_for_unreliable:
                continue
            
            if tag == '$DELETE':
                result_tokens.pop(idx)
            elif tag == '$MERGE_NEXT' and idx + 1 < len(result_tokens):
                result_tokens[idx] = result_tokens[idx] + result_tokens.pop(idx + 1)
            elif tag == '$MERGE_PREV' and idx > 0:
                result_tokens[idx - 1] = result_tokens[idx - 1] + result_tokens.pop(idx)
        
        # PHASE 3: Replacements and Appends
        offset = 0
        for corr in sorted(replaces_appends, key=lambda x: x['word_idx']):
            original_idx = corr['word_idx']
            tag = corr['primary_tag']
            current_idx = original_idx + offset
            
            if current_idx >= len(result_tokens):
                continue
            
            if tag == '$REPLACE':
                suggestions = self.get_mlm_suggestions(" ".join(result_tokens), [current_idx], k=5, verbose=False)
                if suggestions and suggestions[0]:
                    for sugg in suggestions[0]:
                        if self._is_valid_token(sugg):
                            result_tokens[current_idx] = sugg
                            break
            
            elif tag == '$APPEND':
                suggestions = self.get_mlm_suggestions(" ".join(result_tokens), [current_idx + 1], k=5)
                if suggestions and suggestions[0]:
                    for sugg in suggestions[0]:
                        if self._is_valid_token(sugg):
                            result_tokens.insert(current_idx + 1, sugg)
                            offset += 1
                            break
        
        return " ".join(result_tokens)

    def generate_correction_candidates(self, tokens: List[str], predictions: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Generate multiple correction candidates with strategies."""
        candidates = []
        
        error_predictions = [p for p in predictions if p['is_error']]
        if not error_predictions:
            return [((" ".join(tokens), "no_errors"))]
        
        valid_predictions = [p for p in error_predictions if p['primary_tag'] is not None]
        
        # Strategy 1: Primary tags
        if valid_predictions:
            corrected = self.apply_tag_based_correction(tokens, valid_predictions, use_mlm_for_unreliable=False)
            candidates.append((corrected, "primary_tags"))
        
        # Strategy 2: Forced swap
        swap_corrections = [p for p in valid_predictions if 'SWAP' in (p['primary_tag'] or '')]
        non_swap_errors = [p for p in valid_predictions if 'SWAP' not in (p['primary_tag'] or '')]
        
        if len(swap_corrections) == 1 and non_swap_errors:
            swap_idx = swap_corrections[0]['word_idx']
            for other in non_swap_errors:
                other_idx = other['word_idx']
                if abs(swap_idx - other_idx) == 1:
                    temp_tokens = list(tokens)
                    if swap_idx < other_idx:
                        temp_tokens[swap_idx], temp_tokens[other_idx] = temp_tokens[other_idx], temp_tokens[swap_idx]
                    else:
                        temp_tokens[other_idx], temp_tokens[swap_idx] = temp_tokens[swap_idx], temp_tokens[other_idx]
                    
                    remaining = [p for p in valid_predictions if p['word_idx'] not in [swap_idx, other_idx]]
                    corrected = self.apply_tag_based_correction(temp_tokens, remaining, use_mlm_for_unreliable=False)
                    candidates.append((corrected, "forced_swap"))
                    break
        
        # Strategy 3: MLM for unreliable
        unreliable_predictions = [p for p in valid_predictions if not p['is_reliable']]
        if unreliable_predictions:
            unreliable_positions = [p['word_idx'] for p in unreliable_predictions]
            suggestions = self.get_mlm_suggestions(" ".join(tokens), unreliable_positions)
            
            if suggestions and len(suggestions) == len(unreliable_positions):
                corrected = self.apply_mlm_to_positions(tokens, unreliable_positions, suggestions)
                reliable_corrections = [p for p in valid_predictions if p['is_reliable']]
                if reliable_corrections:
                    corrected = self.apply_tag_based_correction(corrected.split(), reliable_corrections, use_mlm_for_unreliable=False)
                candidates.append((corrected, "mlm_unreliable"))
        
        # Strategy 4: Full MLM
        all_error_positions = [p['word_idx'] for p in error_predictions]
        if len(all_error_positions) > 1:
            suggestions = self.get_mlm_suggestions(" ".join(tokens), all_error_positions)
            if suggestions and len(suggestions) == len(all_error_positions):
                corrected = self.apply_mlm_to_positions(tokens, all_error_positions, suggestions)
                candidates.append((corrected, "full_mlm"))
        
        # Remove duplicates
        seen = set()
        unique_candidates = []
        for sent, strategy in candidates:
            if sent not in seen:
                seen.add(sent)
                unique_candidates.append((sent, strategy))
        
        return unique_candidates

    # ===========================================================================
    # MAIN PIPELINE (from notebook)
    # ===========================================================================

    def correct_sentence(self, sentence: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Complete GEC pipeline with aggressive MLM fallback.
        Returns: dict formatted for app.py
        """
        preprocessed = self.preprocess_sentence(sentence)
        
        current_sentence = preprocessed
        iteration = 0
        correction_history = []
        seen_sentences = {current_sentence}
        previous_error_count = float('inf')
        
        # For UI suggestions
        ui_suggestions = []
        
        while iteration < MAX_CORRECTION_ITERATIONS:
            iteration += 1
            
            is_correct, validation_msg = self.is_sentence_fully_correct(current_sentence)
            
            if is_correct:
                break
            
            predictions = self.get_token_predictions(current_sentence, verbose=verbose)
            error_predictions = [p for p in predictions if p['is_error']]
            
            # CRITICAL FIX: GED-only error scenario - EXHAUSTIVE SEARCH
            if not error_predictions:
                if verbose:
                    print("⚠️  Binary found no errors, but GED detected issues.")
                    print("🔄 Attempting AGGRESSIVE MLM on ALL token positions...")
                
                tokens = current_sentence.split()
                all_valid_candidates = []  # Store (sentence, confidence, strategy, position)
                
                # PHASE 1: Single-position MLM
                for pos in range(len(tokens)):
                    suggestions_single = self.get_mlm_suggestions(current_sentence, [pos], k=5, verbose=False)
                    
                    if suggestions_single and suggestions_single[0]:
                        for sugg in suggestions_single[0]:
                            test_tokens = list(tokens)
                            test_tokens[pos] = sugg
                            test_sentence = " ".join(test_tokens)
                            
                            if test_sentence in seen_sentences:
                                continue
                            
                            is_valid, reason = self.is_sentence_fully_correct(test_sentence)
                            
                            if is_valid:
                                ged_correct, ged_conf = self.check_sentence_correctness(test_sentence)
                                cand_dict = {
                                    'sentence': self.postprocess_sentence(test_tokens),
                                    'confidence': ged_conf,
                                    'strategy': f"1mask_pos{pos}",
                                    'position': pos,
                                    'original_word': tokens[pos],
                                    'corrected_word': sugg
                                }
                                all_valid_candidates.append(cand_dict)
                                ui_suggestions.append(cand_dict) # Add to UI suggestions
                
                # PHASES 2 & 3: (Skipped for brevity in UI engine, but included in notebook)
                # You can add Phase 2 (pairs) and Phase 3 (triplets) here
                # if you need the *full* exhaustive search.
                
                # SELECT BEST CANDIDATE BY HIGHEST CONFIDENCE
                if all_valid_candidates:
                    all_valid_candidates.sort(key=lambda x: x['confidence'], reverse=True)
                    best = all_valid_candidates[0]
                    
                    # Need to re-preprocess for the next loop iteration
                    current_sentence = self.preprocess_sentence(best['sentence']) 
                    seen_sentences.add(current_sentence)
                    correction_history.append((iteration, best['strategy'], current_sentence))
                else:
                    break # No valid corrections found
                
                continue  # Skip to next iteration
            
            # --- Standard Error Correction Path ---
            
            if len(error_predictions) > previous_error_count * 1.5:
                break # Error count increased significantly
            
            previous_error_count = len(error_predictions)
            
            tokens = current_sentence.split()
            candidates = self.generate_correction_candidates(tokens, predictions)
            
            best_candidate = None
            best_strategy = None
            
            for candidate, strategy in candidates:
                if candidate in seen_sentences:
                    continue
                
                is_valid, reason = self.is_sentence_fully_correct(candidate)
                
                # For UI: check confidence of valid candidates
                if is_valid:
                    ged_correct, ged_conf = self.check_sentence_correctness(candidate)
                    ui_suggestions.append({
                        'sentence': self.postprocess_sentence(candidate.split()),
                        'confidence': ged_conf,
                        'strategy': strategy,
                        'position': -1, 'original_word': 'N/A', 'corrected_word': 'N/A'
                    })

                if is_valid:
                    best_candidate = candidate
                    best_strategy = strategy
                    break
            
            if best_candidate:
                current_sentence = best_candidate
                seen_sentences.add(current_sentence)
                correction_history.append((iteration, best_strategy, current_sentence))
            else:
                if candidates:
                    # Use best candidate even if it failed validation, as a fallback
                    current_sentence = candidates[0][0]
                    seen_sentences.add(current_sentence)
                else:
                    break # No candidates
        
        # --- Final formatting for app.py ---
        
        final_output = self.postprocess_sentence(current_sentence.split())
        
        # Final validation
        is_correct_final, validation_msg_final = self.is_sentence_fully_correct(current_sentence)
        ged_correct, ged_conf = self.check_sentence_correctness(current_sentence)

        # Sort and filter suggestions for UI
        ui_suggestions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates
        seen_suggs = set()
        final_suggestions = []
        for s in ui_suggestions:
            if s['sentence'] not in seen_suggs:
                seen_suggs.add(s['sentence'])
                final_suggestions.append(s)
        
        # Check if original was actually correct
        is_original_correct, _ = self.is_sentence_fully_correct(preprocessed)
        
        return {
            'original': sentence,
            'corrected': final_output,
            'is_correct': is_original_correct, # Was the *original* correct?
            'confidence': ged_conf, # Confidence of the *final* sentence
            'suggestions': final_suggestions[:3], # Top 3 unique suggestions
            'iterations': iteration,
            'message': validation_msg_final
        }