#!/usr/bin/env python3
"""
Python example using transformers AutoTokenizer - the high-level interface most commonly used.
This uses the same underlying Rust tokenizers library but through the transformers abstraction.
"""

import sys
import torch
import numpy as np
from transformers import AutoTokenizer

def main():
    print("Loading bert-base-uncased tokenizer using AutoTokenizer from transformers...")
    
    try:
        # Load tokenizer using AutoTokenizer - much simpler!
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        print("✓ Successfully loaded tokenizer!")
        
        # Test text to tokenize (same as other versions)
        text = "The rain, in Spain, falls mainly on the plain."
        print(f"\nInput text: {text}")
        
        # Tokenize with different return types to show flexibility
        print("\n=== AUTOTOKENIZER OUTPUT ANALYSIS ===")
        
        # Method 1: Return PyTorch tensors (most common in ML)
        encoding_torch = tokenizer(text, return_tensors="pt")
        
        # Method 2: Return numpy arrays
        encoding_numpy = tokenizer(text, return_tensors="np")
        
        # Method 3: Return Python lists (like our previous examples)
        encoding_python = tokenizer(text, return_tensors=None)
        
        # Method 4: Get detailed token information
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Add special tokens manually for comparison
        tokens_with_special = tokenizer.tokenize(text, add_special_tokens=True)
        
        # Get full encoding with all possible return values
        full_encoding = tokenizer.encode_plus(
            text, 
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
            return_overflowing_tokens=True,
            return_length=True
        )
        
        # Debug: Print all available keys in the encoding
        print(f"\n🔍 FULL ENCODING KEYS:")
        print(f"  Available keys: {list(full_encoding.keys())}")
        print(f"  Full encoding object: {full_encoding}")
        
        print("📊 AUTOTOKENIZER OUTPUT SHAPES AND TYPES:")
        
        # PyTorch tensors
        print(f"\n🔥 PYTORCH TENSORS (return_tensors='pt'):")
        for key, tensor in encoding_torch.items():
            print(f"  {key}: torch.Tensor")
            print(f"    └─ Shape: {list(tensor.shape)}")
            print(f"    └─ dtype: {tensor.dtype}")
            print(f"    └─ device: {tensor.device}")
            print(f"    └─ Values: {tensor.squeeze().tolist()}")
        
        # NumPy arrays
        print(f"\n🔢 NUMPY ARRAYS (return_tensors='np'):")
        for key, array in encoding_numpy.items():
            print(f"  {key}: numpy.ndarray")
            print(f"    └─ Shape: {list(array.shape)}")
            print(f"    └─ dtype: {array.dtype}")
            print(f"    └─ Memory: {array.nbytes} bytes")
            print(f"    └─ Values: {array.squeeze().tolist()}")
        
        # Python lists
        print(f"\n🐍 PYTHON LISTS (return_tensors=None):")
        for key, value in encoding_python.items():
            print(f"  {key}: {type(value).__name__}")
            print(f"    └─ Length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            print(f"    └─ Values: {value}")
        
        # Show additional attributes if available
        if 'special_tokens_mask' in full_encoding:
            special_mask = full_encoding['special_tokens_mask']
            print(f"\n🎭 SPECIAL TOKENS MASK:")
            print(f"  special_tokens_mask: {type(special_mask).__name__}")
            try:
                print(f"    └─ Length: {len(special_mask)}")
                print(f"    └─ Values: {special_mask}")
                print(f"    └─ Meaning: 1=special token, 0=regular token")
                
                # Handle different data types for summing
                try:
                    if isinstance(special_mask, list):
                        special_count = sum(special_mask)
                    elif hasattr(special_mask, 'sum'):
                        special_count = special_mask.sum()
                    elif hasattr(special_mask, 'tolist'):
                        special_count = sum(special_mask.tolist())
                    else:
                        special_count = sum(special_mask)
                    
                    print(f"    └─ Special tokens count: {special_count}")
                    print(f"    └─ Regular tokens count: {len(special_mask) - special_count}")
                except Exception as sum_error:
                    print(f"    └─ Could not calculate counts: {sum_error}")
                    
            except Exception as e:
                print(f"    └─ Error processing special_tokens_mask: {e}")
        
        if 'overflowing_tokens' in full_encoding:
            overflow = full_encoding['overflowing_tokens']
            print(f"\n🌊 OVERFLOWING TOKENS:")
            print(f"  overflowing_tokens: {type(overflow).__name__}")
            try:
                overflow_len = len(overflow) if overflow and hasattr(overflow, '__len__') else 0
                print(f"    └─ Length: {overflow_len}")
                print(f"    └─ Content: {overflow if overflow else 'None (no overflow)'}")
                print(f"    └─ Meaning: Additional tokens when input exceeds max length")
            except Exception as e:
                print(f"    └─ Error processing overflowing_tokens: {e}")
        
        if 'overflow_to_sample_mapping' in full_encoding:
            overflow_mapping = full_encoding['overflow_to_sample_mapping']
            print(f"\n🗺️  OVERFLOW TO SAMPLE MAPPING:")
            print(f"  overflow_to_sample_mapping: {type(overflow_mapping).__name__}")
            try:
                mapping_len = len(overflow_mapping) if overflow_mapping and hasattr(overflow_mapping, '__len__') else 0
                print(f"    └─ Length: {mapping_len}")
                print(f"    └─ Content: {overflow_mapping if overflow_mapping else 'None (no overflow)'}")
                print(f"    └─ Meaning: Maps overflow tokens back to original sample index in batch")
            except Exception as e:
                print(f"    └─ Error processing overflow_to_sample_mapping: {e}")
        
        if 'length' in full_encoding:
            length = full_encoding['length']
            print(f"\n📏 SEQUENCE LENGTH:")
            print(f"  length: {type(length).__name__}")
            print(f"    └─ Value: {length}")
            print(f"    └─ Meaning: Total number of tokens including special tokens")
        
        # Token analysis
        print(f"\n🔍 TOKEN ANALYSIS:")
        print(f"  Raw tokens (no special): {tokens}")
        print(f"  Token count (no special): {len(tokens)}")
        print(f"  Tokens with special: {tokens_with_special}")
        print(f"  Token count (with special): {len(tokens_with_special)}")
        
        # Detailed encoding with offsets and special token info
        print(f"\n📍 DETAILED ENCODING WITH OFFSETS:")
        input_ids = full_encoding['input_ids']
        offsets = full_encoding.get('offset_mapping', [])
        special_mask = full_encoding.get('special_tokens_mask', [])
        
        # Safely handle different data types
        safe_offsets = offsets if offsets else [(0,0)] * len(input_ids)
        safe_special_mask = []
        if special_mask:
            if hasattr(special_mask, 'tolist'):
                safe_special_mask = special_mask.tolist()
            elif isinstance(special_mask, list):
                safe_special_mask = special_mask
            else:
                safe_special_mask = list(special_mask)
        else:
            safe_special_mask = [0] * len(input_ids)
            
        for i, (token_id, offset, is_special) in enumerate(zip(
            input_ids, 
            safe_offsets,
            safe_special_mask
        )):
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            start, end = offset
            
            # Handle special tokens and extract original text
            if is_special or (start == end == 0 and token in ['[CLS]', '[SEP]']):
                original_text = "[special]"
                token_type = "SPECIAL"
            elif start < len(text) and end <= len(text) and start <= end:
                original_text = text[start:end]
                token_type = "REGULAR"
            else:
                original_text = "[unknown]"
                token_type = "UNKNOWN"
            
            print(f"  [{i:2}] '{token}' -> ID:{token_id:5} | "
                  f"offset:({start:2},{end:2}) | type:{token_type:7} | orig:'{original_text}'")
        
        # Vocabulary information
        print(f"\n📚 VOCABULARY INFORMATION:")
        print(f"  Vocabulary size: {len(tokenizer)}")
        print(f"  Special tokens: {tokenizer.special_tokens_map}")
        print(f"  Model max length: {tokenizer.model_max_length}")
        print(f"  Padding side: {tokenizer.padding_side}")
        print(f"  Truncation side: {tokenizer.truncation_side}")
        
        # Show round-trip (encode -> decode)
        print(f"\n🔄 ROUND-TRIP TEST:")
        decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_with_special = tokenizer.decode(input_ids, skip_special_tokens=False)
        print(f"  Original: '{text}'")
        print(f"  Decoded (no special): '{decoded}'")
        print(f"  Decoded (with special): '{decoded_with_special}'")
        print(f"  Round-trip success: {text.lower() == decoded.lower()}")
        
        # Batch processing example
        print(f"\n📦 BATCH PROCESSING EXAMPLE:")
        texts = [
            "Hello world!",
            "This is a longer sentence with more tokens.",
            "Short."
        ]
        
        batch_encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        print(f"  Input texts: {len(texts)}")
        print(f"  Batch shape: {list(batch_encoding['input_ids'].shape)}")
        print(f"  Padding token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
        
        # Show the batch
        for i, text_input in enumerate(texts):
            ids = batch_encoding['input_ids'][i].tolist()
            mask = batch_encoding['attention_mask'][i].tolist()
            print(f"  [{i}] '{text_input}'")
            print(f"      IDs: {ids}")
            print(f"      Mask: {mask}")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install transformers torch numpy")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()