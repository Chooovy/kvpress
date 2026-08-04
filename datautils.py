import torch
from datasets import load_dataset
from itertools import chain
from pathlib import Path
import os
from typing import List, Dict, Any
import numpy as np

CACHE_DIR = "/aifs4su/guhao/KVCache/dma_kvcache_eviction/cache"

# ==================== SFT 数据集处理 ====================

def format_longbench_v2_for_sft(example):
    """
    LongBench-v2: 多选题格式，构建 SFT 样本
    输入: context + question + choices
    输出: answer (A/B/C/D)
    """
    # 构建 prompt
    context = example.get("context", "")
    question = example.get("question", "")
    choices = []
    for choice_key in ["choice_A", "choice_B", "choice_C", "choice_D"]:
        choice_val = example.get(choice_key, "")
        if choice_val:
            choices.append(f"{choice_key[-1]}. {choice_val}")
    choices_text = "\n".join(choices)
    
    prompt = f"""Based on the following context, answer the question by selecting the correct option.

Context:
{context}

Question: {question}

Options:
{choices_text}

Answer:"""
    
    answer = example.get("answer", "")
    
    return {
        "input": prompt,
        "output": answer,
        "length": len(context.split()) if context else 0,  # 近似 word count
    }


def format_longbench_for_sft(example):
    """
    LongBench: 问答/生成格式，构建 SFT 样本
    输入: context + input (question/query)
    输出: answers[0] (取第一个答案)
    """
    context = example.get("context", "")
    input_text = example.get("input", "")
    answers = example.get("answers", [])
    answer = answers[0] if answers else ""
    
    # 根据数据集类型构建不同的 prompt
    dataset_name = example.get("dataset", "")
    
    if "passage_count" in dataset_name:
        prompt = f"""Count the number of passages in the following context.

Context:
{context}

{input_text}

Answer:"""
    elif "passage_retrieval" in dataset_name:
        prompt = f"""Find and retrieve the relevant passage from the context.

Context:
{context}

Query: {input_text}

Relevant Passage:"""
    elif any(x in dataset_name for x in ["qasper", "narrativeqa", "hotpotqa", "triviaqa", "multifieldqa"]):
        prompt = f"""Answer the question based on the given context.

Context:
{context}

Question: {input_text}

Answer:"""
    elif any(x in dataset_name for x in ["gov_report", "multi_news", "qmsum", "vcsum"]):
        prompt = f"""Summarize the following content.

Content:
{context}

{input_text}

Summary:"""
    elif any(x in dataset_name for x in ["lcc", "repobench"]):
        prompt = f"""Complete the code based on the given context.

Context:
{context}

{input_text}

Completion:"""
    elif any(x in dataset_name for x in ["trec", "lsht"]):
        prompt = f"""Classify the following based on the context.

Context:
{context}

{input_text}

Classification:"""
    elif "samsum" in dataset_name:
        prompt = f"""Summarize the dialogue.

Dialogue:
{context}

{input_text}

Summary:"""
    else:
        # 通用格式
        prompt = f"""Based on the following context, complete the task.

Context:
{context}

Task: {input_text}

Response:"""
    
    return {
        "input": prompt,
        "output": answer,
        "length": example.get("length", len(context.split())),
    }


def load_longbench_v2_for_sft(
    tokenizer,
    pt_context_len=None,
    max_samples=None,
    preprocessing_num_workers=4,
    overwrite_cache=False,  # 新增
):
    """
    加载 LongBench-v2 用于 SFT
    """
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # 缓存文件名
    cache_suffix = f"_{pt_context_len}" if pt_context_len else "_auto"
    cache_file = f'{CACHE_DIR}/longbench_v2_sft{cache_suffix}.cache'
    
    # 如果缓存存在，直接加载
    if os.path.exists(cache_file) and not overwrite_cache:
        print(f"Loading LongBench-v2 from cache: {cache_file}")
        cached_data = torch.load(cache_file, weights_only=False)
        print(f"Loaded from cache - Train: {len(cached_data['train'])}, Eval: {len(cached_data['test'])}")
        return cached_data['train'], cached_data['test']
    
    print("Loading LongBench-v2 dataset...")
    dataset = load_dataset('THUDM/LongBench-v2', split='train', cache_dir=CACHE_DIR)
    
    # 转换格式
    dataset = dataset.map(
        format_longbench_v2_for_sft,
        remove_columns=dataset.column_names,
        num_proc=preprocessing_num_workers,
        desc="Formatting LongBench-v2 for SFT",
    )
    
    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))
    
    # 自动检测最大长度
    if pt_context_len is None:
        print("Auto-detecting max context length from dataset...")
        sample_size = min(500, len(dataset))
        sample_indices = list(range(sample_size))
        sample_data = dataset.select(sample_indices)
        
        max_token_len = 0
        for example in sample_data:
            full_text = example["input"] + example["output"] + tokenizer.eos_token
            tokens = tokenizer(full_text, truncation=False, padding=False)
            max_token_len = max(max_token_len, len(tokens["input_ids"]))
        
        pt_context_len = ((max_token_len // 128) + 1) * 128
        pt_context_len = min(pt_context_len, 131072)
        print(f"Auto-detected max context length: {max_token_len} tokens, using pt_context_len={pt_context_len}")
    
    # Tokenize for SFT
    def tokenize_sft(examples):
        full_texts = [
            inp + out + tokenizer.eos_token 
            for inp, out in zip(examples["input"], examples["output"])
        ]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=pt_context_len,
            padding=False,
        )
        
        labels = []
        for i, (inp, out) in enumerate(zip(examples["input"], examples["output"])):
            input_ids = tokenized["input_ids"][i]
            input_only = tokenizer(inp, truncation=True, max_length=pt_context_len, padding=False)
            input_len = len(input_only["input_ids"])
            label = [-100] * input_len + input_ids[input_len:]
            label = label[:len(input_ids)]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input", "output", "length"],
        num_proc=preprocessing_num_workers,
        desc="Tokenizing LongBench-v2",
    )
    
    split = tokenized_dataset.train_test_split(test_size=min(100, len(tokenized_dataset) // 10), seed=42)
    
    # 保存缓存
    torch.save({'train': split['train'], 'test': split['test']}, cache_file)
    # print(f"LongBench-v2 cached to {cache_file}")
    
    # print(f"LongBench-v2 loaded - Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split['train'], split['test']


def load_longbench_for_sft(
    tokenizer,
    subsets=None,
    pt_context_len=None,
    max_samples_per_subset=None,
    preprocessing_num_workers=4,
    use_extended=False,
    overwrite_cache=False,  # 新增
):
    """
    加载 LongBench 用于 SFT，支持缓存
    """
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # 缓存文件名
    cache_suffix = f"_{pt_context_len}" if pt_context_len else "_auto"
    ext_suffix = "_extended" if use_extended else ""
    cache_file = f'{CACHE_DIR}/longbench_sft{cache_suffix}{ext_suffix}.cache'
    
    # 如果缓存存在，直接加载
    if os.path.exists(cache_file) and not overwrite_cache:
        # print(f"Loading LongBench from cache: {cache_file}")
        cached_data = torch.load(cache_file, weights_only=False)
        # print(f"Loaded from cache - Train: {len(cached_data['train'])}, Eval: {len(cached_data['test'])}")
        return cached_data['train'], cached_data['test']
    
    if subsets is None:
        if use_extended:
            subsets = [
                "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
                "gov_report", "multi_news", "trec", "triviaqa", "samsum", 
                "passage_count", "passage_retrieval_en", "lcc", "repobench-p"
            ]
        else:
            subsets = [
                "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", 
                "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", 
                "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", 
                "lsht", "passage_count", "passage_retrieval_en", 
                "passage_retrieval_zh", "lcc", "repobench-p"
            ]
    
    all_data = []
    
    for subset in subsets:
        subset_name = f"{subset}_e" if use_extended else subset
        try:
            print(f"Loading LongBench/{subset_name}...")
            data = load_dataset('THUDM/LongBench', subset_name, split='test', cache_dir=CACHE_DIR)
            
            data = data.map(
                format_longbench_for_sft,
                num_proc=preprocessing_num_workers,
                desc=f"Formatting {subset_name}",
            )
            
            if max_samples_per_subset and len(data) > max_samples_per_subset:
                data = data.select(range(max_samples_per_subset))
            
            all_data.append(data)
            # print(f"  Loaded {len(data)} samples from {subset_name}")
            
        except Exception as e:
            print(f"  Warning: Failed to load {subset_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No LongBench subsets were loaded successfully")
    
    from datasets import concatenate_datasets
    combined = concatenate_datasets(all_data)
    print(f"Combined LongBench dataset: {len(combined)} samples")
    
    columns_to_keep = ["input", "output", "length"]
    columns_to_remove = [c for c in combined.column_names if c not in columns_to_keep]
    combined = combined.remove_columns(columns_to_remove)
    
    # 自动检测最大长度
    if pt_context_len is None:
        print("Auto-detecting max context length from dataset...")
        sample_size = min(500, len(combined))
        sample_indices = list(range(sample_size))
        sample_data = combined.select(sample_indices)
        
        max_token_len = 0
        for example in sample_data:
            full_text = example["input"] + example["output"] + tokenizer.eos_token
            tokens = tokenizer(full_text, truncation=False, padding=False)
            max_token_len = max(max_token_len, len(tokens["input_ids"]))
        
        pt_context_len = ((max_token_len // 128) + 1) * 128
        pt_context_len = min(pt_context_len, 131072)
        print(f"Auto-detected max context length: {max_token_len} tokens, using pt_context_len={pt_context_len}")
    
    def tokenize_sft(examples):
        full_texts = [
            inp + out + tokenizer.eos_token 
            for inp, out in zip(examples["input"], examples["output"])
        ]
        
        tokenized = tokenizer(
            full_texts,
            truncation=True,
            max_length=pt_context_len,
            padding=False,
        )
        
        labels = []
        for i, (inp, out) in enumerate(zip(examples["input"], examples["output"])):
            input_ids = tokenized["input_ids"][i]
            input_only = tokenizer(inp, truncation=True, max_length=pt_context_len, padding=False)
            input_len = len(input_only["input_ids"])
            label = [-100] * input_len + input_ids[input_len:]
            label = label[:len(input_ids)]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    tokenized_dataset = combined.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input", "output", "length"],
        num_proc=preprocessing_num_workers,
        desc="Tokenizing LongBench",
    )
    
    split = tokenized_dataset.train_test_split(test_size=min(200, len(tokenized_dataset) // 10), seed=42)
    
    # 保存缓存
    torch.save({'train': split['train'], 'test': split['test']}, cache_file)
    # print(f"LongBench cached to {cache_file}")
    
    # print(f"LongBench loaded - Train: {len(split['train'])}, Eval: {len(split['test'])}")
    return split['train'], split['test']


# ==================== 原有代码 ====================

def load_datasets_for_training(
    dataset_name,
    tokenizer,
    model_family="llama",
    pt_context_len=512,
    eval_dataset_size=200,
    max_train_samples=None,
    max_eval_samples=None,
    preprocessing_num_workers=4,
    overwrite_cache=False,
    # LongBench 特有参数
    longbench_subsets=None,
    longbench_use_extended=False,
    longbench_auto_context_len=True,  # 新增：是否自动检测 LongBench 的上下文长度
):
    """
    支持: c4, wikitext, alpaca, redpajama, smollm-corpus, longbench, longbench-v2
    
    Args:
        longbench_auto_context_len: 当使用 longbench/longbench-v2 时，是否自动检测最大上下文长度。
                                   如果为 True，会忽略 pt_context_len 参数，自动使用数据集最大长度。
    """
    
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    
    # LongBench 系列单独处理（SFT 格式）
    if dataset_name == 'longbench-v2':
        # 如果启用自动检测，传 None；否则使用用户指定的值
        context_len = None if longbench_auto_context_len else pt_context_len
        return load_longbench_v2_for_sft(
            tokenizer=tokenizer,
            pt_context_len=context_len,
            max_samples=max_train_samples,
            preprocessing_num_workers=preprocessing_num_workers,
            overwrite_cache=overwrite_cache,  # 新增
        )
    
    if dataset_name == 'longbench':
        context_len = None if longbench_auto_context_len else pt_context_len
        return load_longbench_for_sft(
            tokenizer=tokenizer,
            subsets=longbench_subsets,
            pt_context_len=context_len,
            max_samples_per_subset=max_train_samples,
            preprocessing_num_workers=preprocessing_num_workers,
            use_extended=longbench_use_extended,
            overwrite_cache=overwrite_cache,  # 新增
        )
    
    # 以下是原有的预训练数据集处理逻辑
    # Include tokenizer info to avoid loading caches tokenized with a different vocab.
    tok_name = getattr(tokenizer, "name_or_path", "tokenizer")
    tok_name = str(tok_name).replace("/", "_").replace(" ", "_")
    tok_vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    cache_file = f'{CACHE_DIR}/dataset_{model_family}_{dataset_name}_{pt_context_len}_{tok_name}_vocab{tok_vocab}.cache'
    
    # 如果缓存存在，直接加载
    if os.path.exists(cache_file) and not overwrite_cache:
        print(f"Loading dataset from cache: {cache_file}")
        cached_data = torch.load(cache_file, weights_only=False)
        train_dataset = cached_data['train']
        eval_dataset = cached_data['validation']
        # print(f"Loaded from cache - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    else:
        # print(f"Loading {dataset_name} dataset...")
        
        # 加载原始数据集
        if dataset_name == 'c4':
            dataset = load_dataset(
                "allenai/c4", "en",
                data_files={
                    "train": "en/c4-train.00000-of-01024.json.gz",
                    "validation": "en/c4-validation.00000-of-00008.json.gz",
                },
                cache_dir=CACHE_DIR,
                verification_mode="no_checks",
            )
                
        elif dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=CACHE_DIR)

        elif dataset_name == 'redpajama':
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", cache_dir=CACHE_DIR)
            if "validation" not in dataset.keys():
                dataset["validation"] = dataset["train"].select(range(eval_dataset_size))
                dataset["train"] = dataset["train"].select(range(eval_dataset_size, len(dataset["train"])))
        
        elif dataset_name == 'smollm-corpus':
            # 新增：支持 smollm-corpus
            print("Loading smollm-corpus (fineweb-edu-dedup subset)...")
            full_dataset = load_dataset(
                "HuggingFaceTB/smollm-corpus", 
                "fineweb-edu-dedup",
                split="train",
                cache_dir=CACHE_DIR,
                num_proc=preprocessing_num_workers
            )
            
            # 分割 train 和 validation
            print(f"Total samples: {len(full_dataset)}, splitting into train/validation...")
            dataset = full_dataset.train_test_split(
                test_size=eval_dataset_size, 
                shuffle=True, 
                seed=42
            )
            dataset = {
                "train": dataset["train"], 
                "validation": dataset["test"]
            }
            print(f"Split complete - Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")
                
        elif dataset_name == 'alpaca':
            dataset = load_dataset("tatsu-lab/alpaca", cache_dir=CACHE_DIR)
            ALPACA_PROMPT_DICT = {
                "prompt_input": (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
                ),
                "prompt_no_input": (
                    "Below is an instruction that describes a task. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Response: "
                ),
            }
            def extract_alpaca_dataset(example):
                if example.get("input", "") != "":
                    prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
                else:
                    prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
                return {'input': prompt_format.format(**example), 'output': example['output']}
            
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
            if "validation" not in dataset.keys():
                dataset = dataset["train"].train_test_split(test_size=eval_dataset_size, shuffle=True, seed=42)
                dataset = {"train": dataset["train"], "validation": dataset["test"]}
        elif dataset_name in ['longalpaca', 'longalpaca-12k']:
            print("Loading Yukang/LongAlpaca-12k...")
            dataset = load_dataset("Yukang/LongAlpaca-12k", cache_dir=CACHE_DIR)
            def extract_longalpaca(example):
                return {
                    # 问题直接用 instruction，忽略原 input 字段
                    "input": example.get("instruction", ""),
                    "output": example.get("output", ""),
                }
            
            dataset = dataset.map(
                extract_longalpaca,
                remove_columns=[c for c in dataset["train"].column_names if c not in ["instruction", "input", "output"]],
            )
            if "validation" not in dataset.keys():
                dataset = dataset["train"].train_test_split(test_size=eval_dataset_size, shuffle=True, seed=42)
                dataset = {"train": dataset["train"], "validation": dataset["test"]}
            
            # Tokenize into input_ids/labels
            def tokenize_sft(examples):
                full_texts = [
                    inp + out + tokenizer.eos_token 
                    for inp, out in zip(examples["input"], examples["output"])
                ]
                tokenized = tokenizer(
                    full_texts,
                    truncation=True,
                    max_length=pt_context_len,
                    padding=False,
                )
                labels = []
                for i, (inp, out) in enumerate(zip(examples["input"], examples["output"])):
                    input_ids = tokenized["input_ids"][i]
                    input_only = tokenizer(inp, truncation=True, max_length=pt_context_len, padding=False)
                    input_len = len(input_only["input_ids"])
                    label = [-100] * input_len + input_ids[input_len:]
                    label = label[:len(input_ids)]
                    labels.append(label)
                tokenized["labels"] = labels
                return tokenized
            
            dataset = {
                "train": dataset["train"].map(
                    tokenize_sft,
                    batched=True,
                    remove_columns=["input", "output"],
                    num_proc=preprocessing_num_workers,
                    desc="Tokenizing LongAlpaca train",
                ),
                "validation": dataset["validation"].map(
                    tokenize_sft,
                    batched=True,
                    remove_columns=["input", "output"],
                    num_proc=preprocessing_num_workers,
                    desc="Tokenizing LongAlpaca validation",
                ),
            }
        else:
            raise ValueError(
                f"Dataset {dataset_name} not supported. Choose from: c4, wikitext, alpaca, longalpaca-12k, "
                "redpajama, smollm-corpus, longbench, longbench-v2"
            )

        # 对于预训练数据集 (c4, wikitext, redpajama, smollm-corpus)，进行tokenization和分块
        if dataset_name in ["c4", "wikitext", "redpajama", "smollm-corpus"]:
            print(f"Tokenizing and chunking {dataset_name} with context length {pt_context_len}...")
            column_names = list(dataset["train"].features)
            text_column_name = "text" if "text" in column_names else column_names[0]
            
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name])
            
            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                desc="Tokenizing dataset",
            )
            
            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                if total_length >= pt_context_len:
                    total_length = (total_length // pt_context_len) * pt_context_len
                result = {
                    k: [t[i : i + pt_context_len] for i in range(0, total_length, pt_context_len)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result
            
            dataset = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=preprocessing_num_workers,
                load_from_cache_file=not overwrite_cache,
                desc=f"Grouping texts in chunks of {pt_context_len}",
            )
        
        # 保存缓存
        cache_data = {
            'train': dataset['train'],
            'validation': dataset['validation']
        }
        torch.save(cache_data, cache_file)
        print(f"Dataset cached to {cache_file}")
        
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']
    
    # 限制样本数量
    if max_train_samples is not None and len(train_dataset) > max_train_samples:
        train_dataset = train_dataset.select(range(max_train_samples))
    if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    print(f"Final dataset sizes - Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset




class SimplePaddingCollator:
    def __init__(self, tokenizer, padding="longest", pad_to_multiple_of=None, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
        self.unk_token_id = getattr(tokenizer, "unk_token_id", None)
        self._warned_invalid_ids = False
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch_size = len(features)
        max_length = max(len(f["input_ids"]) for f in features)
        
        if self.pad_to_multiple_of is not None and max_length % self.pad_to_multiple_of != 0:
            max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        
        first_elem = features[0]["input_ids"]
        if isinstance(first_elem, torch.Tensor):
            dtype = first_elem.dtype
            device = first_elem.device
        else:
            dtype = torch.long
            device = torch.device("cpu")
        
        batch_input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=dtype, device=device)
        batch_attention_mask = torch.zeros((batch_size, max_length), dtype=dtype, device=device)
        
        has_labels = "labels" in features[0]
        if has_labels:
            batch_labels = torch.full((batch_size, max_length), self.label_pad_token_id, dtype=dtype, device=device)
        
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attention_mask = feature.get("attention_mask", None)
            
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=dtype)
            if self.vocab_size:
                invalid = (input_ids < 0) | (input_ids >= self.vocab_size)
                if invalid.any():
                    if not self._warned_invalid_ids:
                        print(
                            f"[SimplePaddingCollator] Found out-of-range token ids; "
                            f"clamping to unk/pad (vocab_size={self.vocab_size}).",
                            flush=True,
                        )
                        self._warned_invalid_ids = True
                    fill_id = self.unk_token_id if self.unk_token_id is not None else self.pad_token_id
                    input_ids = input_ids.masked_fill(invalid, int(fill_id))
            if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask, dtype=dtype)
            
            seq_len = len(input_ids)
            
            batch_input_ids[i, :seq_len] = input_ids
            
            if attention_mask is not None:
                batch_attention_mask[i, :seq_len] = attention_mask
            else:
                batch_attention_mask[i, :seq_len] = 1
            
            if has_labels:
                labels = feature["labels"]
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels, dtype=dtype)
                batch_labels[i, :seq_len] = labels
        
        result = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
        }
        
        if has_labels:
            result["labels"] = batch_labels
        
        return result