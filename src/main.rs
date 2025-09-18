use hf_hub::api::tokio::Api;
use tokenizers::tokenizer::{Result, Tokenizer};

#[tokio::main]
async fn main() -> Result<()> {
    println!("Loading bert-base-uncased tokenizer from Hugging Face...");
    
    let api = Api::new().unwrap();
    let repo = api.model("bert-base-uncased".to_string());
    let tokenizer_filename = repo
        .get("tokenizer.json")
        .await
        .expect("Failed to download tokenizer.json");
    
    println!("Downloaded tokenizer to: {:?}", tokenizer_filename);
    let tokenizer = Tokenizer::from_file(&tokenizer_filename)?;
    println!("✓ Successfully loaded tokenizer!");
    
    let text = "The rain, in Spain, falls mainly on the plain.";
    println!("\nInput text: {}", text);
    
    // Tokenize the text WITH special tokens to match Python behavior
    let encoding = tokenizer.encode(text, true)?; // true = add_special_tokens
    
    println!("\n=== TOKENIZER OUTPUT ANALYSIS (WITH SPECIAL TOKENS) ===");
    // Show detailed type and shape information
    let tokens = encoding.get_tokens();
    let token_ids = encoding.get_ids();
    let attention_mask = encoding.get_attention_mask();
    let type_ids = encoding.get_type_ids();
    let offsets = encoding.get_offsets();

    println!("📊 OUTPUT SHAPES AND TYPES:");
    println!("  tokens: Vec<String> with length {}", tokens.len());
    println!("    └─ Type: Vec<String>");
    println!("    └─ Shape: [{}]", tokens.len());
    println!("    └─ Memory size: ~{} bytes", tokens.iter().map(|s| s.len()).sum::<usize>());
    
    println!("  token_ids: &[u32] with length {}", token_ids.len());
    println!("    └─ Type: &[u32]");
    println!("    └─ Shape: [{}]", token_ids.len());
    println!("    └─ Memory size: {} bytes", token_ids.len() * 4);
    
    println!("  attention_mask: &[u32] with length {}", attention_mask.len());
    println!("    └─ Type: &[u32]");
    println!("    └─ Shape: [{}]", attention_mask.len());
    println!("    └─ Values: {:?}", attention_mask);
    
    println!("  type_ids: &[u32] with length {}", type_ids.len());
    println!("    └─ Type: &[u32]");
    println!("    └─ Shape: [{}]", type_ids.len());
    println!("    └─ Values: {:?}", type_ids);
    
    println!("  offsets: &[(usize, usize)] with length {}", offsets.len());
    println!("    └─ Type: &[(usize, usize)]");
    println!("    └─ Shape: [{}]", offsets.len());
    
    println!("\n📝 RAW OUTPUT VALUES:");
    println!("  Tokens: {:?}", tokens);
    println!("  Token IDs: {:?}", token_ids);
    println!("  Total tokens: {}", encoding.len());
    
    // Show statistics
    let min_id = token_ids.iter().min().unwrap_or(&0);
    let max_id = token_ids.iter().max().unwrap_or(&0);
    let avg_token_len: f32 = tokens.iter().map(|t| t.len()).sum::<usize>() as f32 / tokens.len() as f32;
    
    println!("\n📈 STATISTICS:");
    println!("  Token ID range: {} - {}", min_id, max_id);
    println!("  Average token length: {:.2} characters", avg_token_len);
    println!("  Vocabulary compression: {:.1}% (original: {} chars, tokens: {})", 
             (tokens.len() as f32 / text.len() as f32) * 100.0, text.len(), tokens.len());
    
    // Show token-by-token breakdown with offsets
    println!("\n🔍 TOKEN-BY-TOKEN BREAKDOWN:");
    for (i, ((token, id), (start, end))) in tokens.iter().zip(token_ids.iter()).zip(offsets.iter()).enumerate() {
        let original_text = if token == "[CLS]" || token == "[SEP]" {
            "[special]"
        } else if *start < text.len() && *end <= text.len() && start <= end {
            &text[*start..*end]
        } else {
            "[unknown]"
        };
        println!("  [{:2}] '{}' -> ID:{:5} | offset:({:2},{:2}) | orig:'{}'", 
                 i, token, id, start, end, original_text);
    }
    
    // Show comparison with/without special tokens
    println!("\n🔄 COMPARISON WITH/WITHOUT SPECIAL TOKENS:");
    let encoding_no_special = tokenizer.encode(text, false)?;
    println!("  With special tokens: {} tokens", encoding.len());
    println!("  Without special tokens: {} tokens", encoding_no_special.len());
    println!("  Special tokens added: [CLS] at start, [SEP] at end");
    
    // Decode test to match Python versions
    let decoded = tokenizer.decode(token_ids, true)?; // skip_special_tokens = true
    println!("\n📤 DECODE TEST:");
    println!("  Original: '{}'", text);
    println!("  Decoded: '{}'", decoded);
    println!("  Round-trip match: {}", text.to_lowercase() == decoded.to_lowercase());
    
    Ok(())
}
