import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from datasets import load_dataset
from tqdm import tqdm

from kvpress import ExpectedAttentionPress, KnormPress, SnapKVPress, StreamingLLMPress, RandomPress, DMAScorePress, KVzipPress, IndexerScorePress, RandomPress_with_sink
from kvpress.presses.dma_score_press import load_model_with_dma_press
from kvpress.presses.indexer_score_press import load_model_with_indexer_press

def calculate_decode_ppl_with_compression(model, tokenizer, context_text, target_text, press=None, max_context_length=4096, device="cuda:0"):
    model.eval()
    
    context_ids = tokenizer.encode(context_text, return_tensors="pt", add_special_tokens=True)
    target_ids = tokenizer.encode(target_text, return_tensors="pt", add_special_tokens=False)
    
    if context_ids.shape[1] > max_context_length:
        context_ids = context_ids[:, :max_context_length]
    
    context_ids = context_ids.to(device)
    target_ids = target_ids.to(device)
    
    original_cache_size = context_ids.shape[1]
    compressed_cache_size = None
    
    with torch.no_grad():
        cache = DynamicCache()
        
        if press is not None:
            with press(model):
                outputs = model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
                compressed_cache_size = cache.get_seq_length()
        else:
            outputs = model.model(input_ids=context_ids, past_key_values=cache, use_cache=True)
            compressed_cache_size = cache.get_seq_length()
        
        nlls = []
        
        for i in range(target_ids.shape[1]):
            next_token = target_ids[:, i:i+1]
            outputs = model(input_ids=next_token, past_key_values=cache, use_cache=True)
            
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            if i + 1 < target_ids.shape[1]:
                actual_next_token = target_ids[:, i + 1]
                nll = -log_probs[0, actual_next_token]
                nlls.append(nll)
            
            cache = outputs.past_key_values
        
        if nlls:
            avg_nll = torch.stack(nlls).mean()
            ppl = torch.exp(avg_nll).item()
        else:
            ppl = float('inf')
    
    return {"ppl": ppl, "original_cache_size": original_cache_size, "compressed_cache_size": compressed_cache_size, "num_decode_tokens": len(nlls)}


def test_wikitext2_decode_ppl(model_path="/aifs4su/guhao/Models/llama-32-1B", compression_ratios=[1.0, 0.8, 0.6, 0.4, 0.2], press_type="expected_attention", context_length=2048, decode_length=512, num_samples=5, device="cuda:0", use_flash_attention=False):
    model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = "eager"
    if press_type == "dma_score":
        model, tokenizer = load_model_with_dma_press(model_path, model_kwargs=model_kwargs)
    elif press_type == "indexer_score":
        model, tokenizer = load_model_with_indexer_press(model_path, model_kwargs=model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join(dataset["text"])
    full_tokens = tokenizer.encode(full_text)
    samples = []
    stride = (len(full_tokens) - context_length - decode_length) // num_samples
    
    for i in range(num_samples):
        start_idx = i * stride
        context_end = start_idx + context_length
        target_end = context_end + decode_length
        
        if target_end > len(full_tokens):
            break
        
        context_tokens = full_tokens[start_idx:context_end]
        target_tokens = full_tokens[context_end:target_end]
        
        context_text = tokenizer.decode(context_tokens)
        target_text = tokenizer.decode(target_tokens)
        
        samples.append({
            "context": context_text,
            "target": target_text,
            "idx": i
        })
    
    def create_press(ratio):
        if press_type == "expected_attention":
            return ExpectedAttentionPress(compression_ratio=ratio)
        elif press_type == "knorm":
            return KnormPress(compression_ratio=ratio)
        elif press_type == "snapkv":
            return SnapKVPress(compression_ratio=ratio)
        elif press_type == "streaming_llm":
            return StreamingLLMPress(compression_ratio=ratio)
        elif press_type == "random":
            return RandomPress(compression_ratio=ratio)
        elif press_type == "random_with_sink":
            return RandomPress_with_sink(compression_ratio=ratio)
        elif press_type == "dma_score":
            return DMAScorePress(compression_ratio=ratio)
        elif press_type == "kvzip":
            return KVzipPress(compression_ratio=ratio)
        elif press_type == "indexer_score":
            return IndexerScorePress(compression_ratio=ratio)
        else:
            raise ValueError(f"Unknown press type: {press_type}")
    
    all_results = {}
    
    for ratio in compression_ratios:
        if ratio >= 1.0:
            press = None
        else:
            press = create_press(ratio)
        
        ppls = []
        total_original_size = 0
        total_compressed_size = 0
        
        for sample in tqdm(samples, desc=f"Compression Ratio={ratio:.2f}"):
            result = calculate_decode_ppl_with_compression(
                model=model,
                tokenizer=tokenizer,
                context_text=sample["context"],
                target_text=sample["target"],
                press=press,
                max_context_length=context_length,
                device=device
            )
            
            ppls.append(result["ppl"])
            total_original_size += result["original_cache_size"]
            total_compressed_size += result["compressed_cache_size"]
        
        avg_ppl = sum(ppls) / len(ppls)
        avg_original = total_original_size / len(samples)
        avg_compressed = total_compressed_size / len(samples)
        actual_ratio = avg_compressed / avg_original
        
        all_results[ratio] = {
            "avg_ppl": avg_ppl,
            "avg_original_size": avg_original,
            "avg_used_size": avg_compressed,
        }
        
        actual_kept_ratio = avg_compressed / avg_original if avg_original > 0 else 0
        
        delete_pct = 0 if ratio >= 1.0 else ratio * 100
        
        print(f"\nResults for Compression Ratio={ratio:.2f} (Delete {delete_pct:.0f}%):")
        print(f"  Original Cache Size: {avg_original:.1f}")
        print(f"  Used Cache Size: {avg_compressed:.1f} (Kept {actual_kept_ratio*100:.1f}%)")
        print(f"  PPL: {avg_ppl:.4f}")
        
        torch.cuda.empty_cache()
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS - DECODE PPL WITH COMPRESSED KV CACHE")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Press Method: {press_type}")
    print(f"Context Length: {context_length}")
    print(f"Decode Length: {decode_length}")
    print(f"Number of Samples: {len(samples)}")
    print(f"Attention Implementation: {model_kwargs['attn_implementation']}")
    print()
    print(f"{'Delete%':<10} {'Keep%':<10} {'Original':<12} {'Used':<12} {'PPL':<12}")
    print("-" * 80)
    
    for ratio in sorted(compression_ratios, reverse=True):
        result = all_results[ratio]
        avg_ppl = result["avg_ppl"]
        avg_original = result["avg_original_size"]
        avg_used = result["avg_used_size"]
        kept_ratio = avg_used / avg_original if avg_original > 0 else 0
        
        delete_pct = 0 if ratio >= 1.0 else ratio * 100
        
        print(f"{delete_pct:<10.0f} {kept_ratio*100:<10.1f} {avg_original:<12.1f} {avg_used:<12.1f} {avg_ppl:<12.4f}")
    
    return all_results


if __name__ == "__main__":
    results = test_wikitext2_decode_ppl(
        model_path="/aifs4su/guhao/Models/Llama-3.2-1B-Instruct",
        # model_path="/aifs4su/guhao/checkpoints/llama3-1b-instruct-indexer_score",
        # model_path="/aifs4su/guhao/checkpoints/llama3-1b-dma_8gpu",
        compression_ratios=[1.0, 0.8, 0.6, 0.4],
        press_type="dma_score",
        context_length=2048,
        decode_length=256,
        num_samples=5,
        device="cuda:0",
        use_flash_attention=False
    )