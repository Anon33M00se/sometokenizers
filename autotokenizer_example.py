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
        full_encoding = tokenizer.encode_plus(
            text, 
            return_offsets_mapping=True,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        
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
        
        # Token analysis
        print(f"\n🔍 TOKEN ANALYSIS:")
        print(f"  Raw tokens (no special): {tokens}")
        print(f"  Token count (no special): {len(tokens)}")
        print(f"  Tokens with special: {tokens_with_special}")
        print(f"  Token count (with special): {len(tokens_with_special)}")
        
        # Detailed encoding with offsets
        print(f"\n📍 DETAILED ENCODING WITH OFFSETS:")
        input_ids = full_encoding['input_ids']
        offsets = full_encoding.get('offset_mapping', [])
        
        for i, (token_id, offset) in enumerate(zip(input_ids, offsets if offsets else [(0,0)] * len(input_ids))):
            token = tokenizer.convert_ids_to_tokens([token_id])[0]
            start, end = offset
            
            # Handle special tokens and extract original text
            if start == end == 0 and token in ['[CLS]', '[SEP]']:
                original_text = "[special]"
            elif start < len(text) and end <= len(text) and start <= end:
                original_text = text[start:end]
            else:
                original_text = "[unknown]"
            
            print(f"  [{i:2}] '{token}' -> ID:{token_id:5} | "
                  f"offset:({start:2},{end:2}) | orig:'{original_text}'")
        
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