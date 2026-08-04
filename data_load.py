import pandas as pd
from datasets import Dataset
from datasets import concatenate_datasets
from datautils import load_datasets_for_training

def load_ruler(tokenizer, args):
    """
    RULER-style retrieval dataset loader (local parquet).

    Expected columns (see `evaluation/ruler-4k.parquet`):
      - context: str
      - question: str
      - answer_prefix: str
      - answer: np.ndarray/list (take first element)
      - task: str

    We build an SFT sample that focuses supervision on the short answer tokens:
      input_text  = context + "\\n\\nQuestion: ...\\n\\n" + answer_prefix + " "
      output_text = answer[0] + eos

    To avoid poisoning training with truncated retrieval signals, we DROP samples
    whose total token length would exceed `args.pt_context_len`.
    """
    # Support training on multiple context-length variants (e.g., 4k + 8k) by concatenating parquet files.
    # Priority:
    #   1) args.ruler_parquet_paths (list[str])
    #   2) args.ruler_parquet_path (str)
    #   3) auto: use local ruler-4k + ruler-8k if present
    parquet_paths = getattr(args, "ruler_parquet_paths", None)
    if parquet_paths is None:
        one = getattr(args, "ruler_parquet_path", None)
        if one:
            parquet_paths = [one]
        else:
            parquet_paths = [
                "/aifs4su/guhao/KVCache/kvpress/evaluation/ruler-4k.parquet",
                "/aifs4su/guhao/KVCache/kvpress/evaluation/ruler-8k.parquet",
            ]

    dfs = []
    for p in parquet_paths:
        try:
            # Read only required columns to reduce IO/memory.
            dfs.append(pd.read_parquet(p, columns=["context", "question", "answer_prefix", "answer", "task", "max_new_tokens"]))
        except Exception:
            # Fallback: read full parquet if columns are not supported (or differ).
            dfs.append(pd.read_parquet(p))
    df = pd.concat(dfs, ignore_index=True)

    # Optional task filter.
    tasks = getattr(args, "ruler_tasks", None)
    if tasks:
        df = df[df["task"].isin(list(tasks))].reset_index(drop=True)

    def _answer0(x):
        try:
            return str(x[0])
        except Exception:
            return str(x)


    supervise_prefix = bool(getattr(args, "ruler_supervise_prefix", False))
    if supervise_prefix:
        # More supervision signal (useful when answer is only 1-2 tokens).
        # Model learns to generate answer_prefix + answer; labels cover a longer span.
        df["input_text"] = df.apply(
            lambda r: f"{r['context']}\n\n{r['question']}\n\n",
            axis=1,
        )
        df["output_text"] = df.apply(
            lambda r: f"{r['answer_prefix']} {_answer0(r['answer'])}{tokenizer.eos_token}",
            axis=1,
        )
    else:
        # Default: match eval usage where answer_prefix is provided as a fixed prefix,
        # supervise only the answer tokens.
        df["input_text"] = df.apply(
            lambda r: f"{r['context']}\n\n{r['question']}\n\n{r['answer_prefix']} ",
            axis=1,
        )
        df["output_text"] = df["answer"].apply(_answer0) + tokenizer.eos_token

    # Filter out truncated examples by measuring true token length.
    def get_total_token_length(row):
        full_text = row["input_text"] + row["output_text"]
        tokens = tokenizer(full_text, truncation=False, padding=False)
        return len(tokens["input_ids"])

    df["total_tokens"] = df.apply(get_total_token_length, axis=1)
    df = df[df["total_tokens"] <= int(args.pt_context_len)].copy()

    dataset = Dataset.from_pandas(df[["input_text", "output_text"]])
    eval_size = min(int(args.max_eval_samples) if args.max_eval_samples else 200, max(1, len(dataset) // 10))
    split = dataset.train_test_split(test_size=eval_size, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    def tokenize_sft(examples):
        full_texts = [inp + out for inp, out in zip(examples["input_text"], examples["output_text"])]
        tokenized = tokenizer(full_texts, truncation=True, max_length=int(args.pt_context_len), padding=False)
        labels = []
        for i, inp in enumerate(examples["input_text"]):
            input_ids = tokenized["input_ids"][i]
            input_only = tokenizer(inp, truncation=True, max_length=int(args.pt_context_len), padding=False)
            input_len = len(input_only["input_ids"])
            label = [-100] * input_len + input_ids[input_len:]
            label = label[: len(input_ids)]
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

    train_ds = train_ds.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input_text", "output_text"],
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing RULER train dataset",
    )
    eval_ds = eval_ds.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input_text", "output_text"],
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing RULER eval dataset",
    )

    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.select(range(args.max_eval_samples))

    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    print(f"RULER loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)} (paths={parquet_paths})")
    return train_ds, eval_ds

def load_math(tokenizer, args):
    parquet_path = "/aifs4su/guhao/RL/Data/46k.parquet"
    df = pd.read_parquet(parquet_path)
    def extract_content(messages):
        if isinstance(messages, list) and len(messages) > 0:
            return messages[0].get('content', '')
        return str(messages)
    
    df['input_text'] = df['prompt'].apply(extract_content)
    df['output_text'] = df['target'].apply(extract_content)
    
    def get_total_token_length(row):
        full_text = row['input_text'] + row['output_text'] + tokenizer.eos_token
        tokens = tokenizer(full_text, truncation=False, padding=False)
        return len(tokens['input_ids'])
    
    df['total_tokens'] = df.apply(get_total_token_length, axis=1)
    df_filtered = df[df['total_tokens'] <= args.pt_context_len].copy()
    
    dataset = Dataset.from_pandas(df_filtered[['input_text', 'output_text']])
    eval_size = min(args.max_eval_samples if args.max_eval_samples else 200, len(dataset) // 10)
    split = dataset.train_test_split(test_size=eval_size, seed=42)
    
    train_ds = split['train']
    eval_ds = split['test']
    
    def tokenize_sft(examples):
        full_texts = [
            inp + out + tokenizer.eos_token 
            for inp, out in zip(examples["input_text"], examples["output_text"])
        ]
        
        tokenized = tokenizer(full_texts, truncation=True, max_length=args.pt_context_len, padding=False)
        
        labels = []
        for i, (inp, out) in enumerate(zip(examples["input_text"], examples["output_text"])):
            input_ids = tokenized["input_ids"][i]
            input_only = tokenizer(inp, truncation=True, max_length=args.pt_context_len, padding=False)
            input_len = len(input_only["input_ids"])
            label = [-100] * input_len + input_ids[input_len:]
            label = label[:len(input_ids)]
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    train_ds = train_ds.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input_text", "output_text"],
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing train dataset",
    )
    
    eval_ds = eval_ds.map(
        tokenize_sft,
        batched=True,
        remove_columns=["input_text", "output_text"],
        num_proc=args.preprocessing_num_workers,
        desc="Tokenizing eval dataset",
    )
    
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.select(range(args.max_eval_samples))
    
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    
    print(f"Math SFT dataset loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds




def load_longbench_bundle(tokenizer, args):
    train_a, eval_a = load_datasets_for_training(
        dataset_name="longbench-v2",
        tokenizer=tokenizer,
        pt_context_len=args.pt_context_len,
        longbench_auto_context_len=False,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    train_b, eval_b = load_datasets_for_training(
        dataset_name="longbench",
        tokenizer=tokenizer,
        pt_context_len=args.pt_context_len,
        longbench_auto_context_len=False,
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    train_ds = concatenate_datasets([train_a, train_b])
    eval_ds = concatenate_datasets([eval_a, eval_b])

    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    
    return train_ds, eval_ds

def load_c4(tokenizer, args):
    train_ds, eval_ds = load_datasets_for_training(
        dataset_name="c4",
        tokenizer=tokenizer,
        pt_context_len=args.pt_context_len,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=False,
    )
    
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    
    print(f"C4 loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds

def load_wikitext(tokenizer, args):
    train_ds, eval_ds = load_datasets_for_training(
        dataset_name="wikitext",
        tokenizer=tokenizer,
        pt_context_len=args.pt_context_len,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=False,
    )
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    print(f"WikiText loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


def load_longalpaca(tokenizer, args):
    train_ds, eval_ds = load_datasets_for_training(
        dataset_name="longalpaca",
        tokenizer=tokenizer,
        pt_context_len=args.pt_context_len,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        preprocessing_num_workers=args.preprocessing_num_workers,
        overwrite_cache=False,
    )
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    print(f"LongAlpaca-12k loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds


def load_dump_data(tokenizer, args):
    from datasets import Dataset
    
    debug_texts = [
        "Hello, this is a test sentence for debugging.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a great programming language for machine learning.",
        "Debugging is an essential skill for developers.",
        "This is a simple dataset for testing purposes.",
    ] * 100
    
    def tokenize_debug(examples):
        tokenized = tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=args.pt_context_len, 
            padding=False
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    dataset = Dataset.from_dict({"text": debug_texts})
    
    dataset = dataset.map(
        tokenize_debug,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing debug dataset",
    )
    
    train_ds = dataset.select(range(min(3, len(dataset))))
    eval_ds = dataset.select([0])
    
    if args.max_train_samples and len(train_ds) > args.max_train_samples:
        train_ds = train_ds.select(range(args.max_train_samples))
    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.select(range(args.max_eval_samples))
    
    columns = [c for c in ["input_ids", "attention_mask", "labels"] if c in train_ds.column_names]
    train_ds.set_format(type="torch", columns=columns)
    eval_ds.set_format(type="torch", columns=columns)
    
    print(f"Debug dataset loaded - Train: {len(train_ds)}, Eval: {len(eval_ds)}")
    return train_ds, eval_ds