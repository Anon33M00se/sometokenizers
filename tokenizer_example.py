#!/usr/bin/env python3
"""
Python equivalent of the Rust tokenizer example using the official tokenizers library.
This uses the same underlying Rust tokenizers library but with Python bindings.
"""

import sys
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import numpy as np

def main():
    print("Loading bert-base-uncased tokenizer from Hugging Face...")
    
    try:
        # Download tokenizer.json from Hugging Face Hub
        tokenizer_path = hf_hub_download(
            repo_id="bert-base-uncased",
            filename="tokenizer.json"
        )
        print(f"Downloaded tokenizer to: {tokenizer_path}")
        
        # Load the tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print("✓ Successfully loaded tokenizer!")
        
        # Test text to tokenize (same as Rust version)
        text = "The rain, in Spain, falls mainly on the plain."
        print(f"\nInput text: {text}")
        
        # Tokenize the text WITH special tokens to match updated Rust version
        encoding = tokenizer.encode(text, add_special_tokens=True)
        
        # Debug: Print the encoding object to see all attributes
        print(f"\n🔍 ENCODING OBJECT ATTRIBUTES:")
        print(f"  encoding object: {encoding}")
        print(f"  available attributes: {[attr for attr in dir(encoding) if not attr.startswith('_')]}")
        
        print("\n=== TOKENIZER OUTPUT ANALYSIS (WITH SPECIAL TOKENS) ===")
        
        # Extract all the data
        tokens = encoding.tokens
        token_ids = encoding.ids
        attention_mask = encoding.attention_mask
        type_ids = encoding.type_ids
        offsets = encoding.offsets
        special_tokens_mask = encoding.special_tokens_mask
        overflowing = encoding.overflowing
        
        # Show detailed type and shape information
        print("📊 OUTPUT SHAPES AND TYPES:")
        print(f"  tokens: list[str] with length {len(tokens)}")
        print(f"    └─ Type: {type(tokens).__name__}")
        print(f"    └─ Shape: [{len(tokens)}]")
        print(f"    └─ Memory size: ~{sum(len(t) for t in tokens)} bytes")
        
        print(f"  token_ids: list[int] with length {len(token_ids)}")
        print(f"    └─ Type: {type(token_ids).__name__}")
        print(f"    └─ Shape: [{len(token_ids)}]")
        print(f"    └─ Memory size: {len(token_ids) * 4} bytes (as int32)")
        
        print(f"  attention_mask: list[int] with length {len(attention_mask)}")
        print(f"    └─ Type: {type(attention_mask).__name__}")
        print(f"    └─ Shape: [{len(attention_mask)}]")
        print(f"    └─ Values: {attention_mask}")
        
        print(f"  type_ids: list[int] with length {len(type_ids)}")
        print(f"    └─ Type: {type(type_ids).__name__}")
        print(f"    └─ Shape: [{len(type_ids)}]")
        print(f"    └─ Values: {type_ids}")
        
        print(f"  special_tokens_mask: list[int] with length {len(special_tokens_mask)}")
        print(f"    └─ Type: {type(special_tokens_mask).__name__}")
        print(f"    └─ Shape: [{len(special_tokens_mask)}]")
        print(f"    └─ Values: {special_tokens_mask}")
        print(f"    └─ Meaning: 1=special token, 0=regular token")
        
        print(f"  offsets: list[tuple[int, int]] with length {len(offsets)}")
        print(f"    └─ Type: {type(offsets).__name__}")
        print(f"    └─ Shape: [{len(offsets)}]")
        
        print(f"  overflowing: list with length {len(overflowing)}")
        print(f"    └─ Type: {type(overflowing).__name__}")
        print(f"    └─ Content: {overflowing if overflowing else 'None (no overflow)'}")
        print(f"    └─ Meaning: Additional tokens when input exceeds max length")
        
        print("\n📝 RAW OUTPUT VALUES:")
        print(f"  Tokens: {tokens}")
        print(f"  Token IDs: {token_ids}")
        print(f"  Special tokens mask: {special_tokens_mask}")
        print(f"  Total tokens: {len(tokens)}")
        
        # Show statistics
        min_id = min(token_ids) if token_ids else 0
        max_id = max(token_ids) if token_ids else 0
        avg_token_len = sum(len(t) for t in tokens) / len(tokens) if tokens else 0
        
        print("\n📈 STATISTICS:")
        print(f"  Token ID range: {min_id} - {max_id}")
        print(f"  Average token length: {avg_token_len:.2f} characters")
        print(f"  Vocabulary compression: {(len(tokens) / len(text)) * 100:.1f}% "
              f"(original: {len(text)} chars, tokens: {len(tokens)})")
        print(f"  Special tokens count: {sum(special_tokens_mask)}")
        print(f"  Regular tokens count: {len(special_tokens_mask) - sum(special_tokens_mask)}")
        
        # Show token-by-token breakdown with offsets and special token info
        print("\n🔍 TOKEN-BY-TOKEN BREAKDOWN:")
        for i, (token, token_id, (start, end), is_special) in enumerate(zip(tokens, token_ids, offsets, special_tokens_mask)):
            if is_special:
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
        
        # Show comparison with/without special tokens
        print("\n🔄 COMPARISON WITH/WITHOUT SPECIAL TOKENS:")
        encoding_no_special = tokenizer.encode(text, add_special_tokens=False)
        print(f"  With special tokens: {len(tokens)} tokens")
        print(f"  Without special tokens: {len(encoding_no_special.tokens)} tokens")
        print(f"  Special tokens added: {sum(special_tokens_mask)}")
        
        # Bonus: Show numpy array conversion (common in ML workflows)
        print("\n🔢 NUMPY ARRAY CONVERSIONS:")
        ids_array = np.array(token_ids, dtype=np.int32)
        mask_array = np.array(attention_mask, dtype=np.int32)
        special_mask_array = np.array(special_tokens_mask, dtype=np.int32)
        
        print(f"  token_ids as numpy: {ids_array.dtype} array with shape {ids_array.shape}")
        print(f"  attention_mask as numpy: {mask_array.dtype} array with shape {mask_array.shape}")
        print(f"  special_tokens_mask as numpy: {special_mask_array.dtype} array with shape {special_mask_array.shape}")
        print(f"  Memory usage: {ids_array.nbytes + mask_array.nbytes + special_mask_array.nbytes} bytes total")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install tokenizers huggingface-hub numpy")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()