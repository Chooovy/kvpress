import os, csv, json
import argparse
from tqdm import tqdm
from datasets import load_dataset
import torch.multiprocessing as mp
import torch
import math
from model.model_factory import load_pretrained


class ForceOptionLogitsProcessor(torch.nn.Module):
    def __init__(self, tokenizer, prefix_ids):
        super().__init__()
        self.prefix_ids = prefix_ids    
        self.step_counter = 0          
        self.original_logits = []    
    def forward(self, input_ids, scores):
        if self.step_counter < len(self.prefix_ids):
            self.original_logits.append(scores.detach().clone())
            # Every token matches exactly the token in the prefix of same position
            forced_token_id = self.prefix_ids[self.step_counter]
            mask = torch.full_like(scores, float("-inf"))  
            mask[:, forced_token_id] = 0                  
            scores = scores + mask                         
            # print (f"scores: {scores}, prefix_ids: {self.prefix_ids[self.step_counter]}")
            self.step_counter += 1    
        else:
            pass
        return scores

# Cloze: Following https://arxiv.org/pdf/2406.07887v1#page=6.06, no choices are given in the prompt 
# template = \
# """$DOC$

# $Q$

# The correct answer is """
template = \
"""$DOC$

$Q$

"""


def evaluate_cloze(model, args, tokenizer, prompt, options, max_input_tokens = 16384, rank=0):
    # print(f"prompt: {prompt}, options: {options}")
    inputs = tokenizer(prompt, return_tensors="pt")
    # Trucate the input if it is too long
    input_ids = inputs["input_ids"]
    seq_length = input_ids.shape[1]  # [batch_size, seq_len]

    if seq_length > max_input_tokens:
        inputs["input_ids"] = input_ids[:, -max_input_tokens:]
        inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_tokens:]
    device=torch.device(f"cuda:{rank}")
    inputs.to(device)

    option_scores = []
    for option in options:
        option_id = tokenizer.encode(option)
        # option_id.to(device)
        option_len = len(option_id)
        option_processor = ForceOptionLogitsProcessor(tokenizer, option_id)

        generative = args.generative
        tqdm.write(f"inputs.shape: {inputs['input_ids'].shape}")
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=option_len, 
                logits_processor=[option_processor], 
                output_scores=True,    
                return_dict_in_generate=True,
                num_beams=1,
                do_sample=False,    # Use greedy search
                use_cache=generative
            )

        generated_ids = outputs.sequences[0, inputs.input_ids.shape[-1]:]
        if not torch.equal(generated_ids.cpu(), torch.tensor(option_id)):
            raise ValueError("Forced generation failed.")

        
        log_prob = 0.0
        # print(f"option_len: {option_len}, option_processor.original_logits: {len(option_processor.original_logits)}")
        assert len(option_processor.original_logits) == option_len
        for step in range(len(option_processor.original_logits)):
            logits = option_processor.original_logits[step][0]
            prob = torch.log_softmax(logits, dim=-1)
            target_token = option_id[step]
            log_prob += prob[target_token].item()

        avg_log_prob = log_prob / option_len
        option_scores.append(avg_log_prob)

    perplexities = [(-torch.tensor(score)).item() for score in option_scores]

    best_option_index = option_scores.index(max(option_scores))  
    return chr(65 + best_option_index)  # return 'A', 'B', 'C', or 'D'

def evaluate_cloze_fwd(model, tokenizer, prompt, options, max_input_tokens=16384, rank=0, pad_to_multiple=1):
    device = torch.device(f"cuda:{rank}")
    option_scores = []

    for option in options:
        prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=False).input_ids[0]
        prompt_len = len(prompt_ids)
        option_ids = tokenizer(option, return_tensors="pt", truncation=False).input_ids[0]
        option_len = len(option_ids)
        input_ids = torch.cat([prompt_ids, option_ids], dim=0)      # select the first squence in the batch (batch_size==1)

        # Truncate
        # if len(input_ids) > max_input_tokens:
        #     half = max_input_tokens // 2
        #     input_ids = torch.cat([input_ids[:half], input_ids[-half:]])
        if len(input_ids) > max_input_tokens:
            inputs = input_ids[-max_input_tokens:]

        # padding
        original_length = len(input_ids)
        target_length = math.ceil(original_length / pad_to_multiple) * pad_to_multiple
        padding_length = target_length - original_length
        # Pad the sequence with `pad_token_id`
        padding_tensor = torch.tensor([-100] * padding_length)
        input_ids = torch.cat([input_ids, padding_tensor])        
        
        actual_prompt_len = min(prompt_len, len(input_ids) - option_len)
        actual_option_len = len(input_ids) - actual_prompt_len

        # print(f"input_len: {len(input_ids)}")
        # print (f"input_ids: {tokenizer.decode(input_ids)}, option_len: {option_len}")
        inputs = {
            "input_ids": input_ids.unsqueeze(0).int().to(device),
        }

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            model.use_cache = False
            model.inference_segment=max_input_tokens
            model.eval()
            outputs = model(**inputs,
                        use_cache=False,
                        # inference_segment=max_input_tokens
                        )
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        log_prob = 0.0
        input_ids_cpu = input_ids.cpu().numpy() 
        total_len = input_ids_cpu.shape[0]
        for i in range(actual_prompt_len - 1, original_length - 1):
            target_token = int(input_ids_cpu[i + 1])
            neg_index = i - total_len
            log_prob += log_probs[neg_index][target_token]
        avg_log_prob = log_prob / actual_option_len
        option_scores.append(avg_log_prob)

    perplexities = [-score for score in option_scores]  # scores already in cpu mem

    for i, option in enumerate(options):
        print(f"Option {chr(65 + i)} PPL: {perplexities[i]:.4f} \t{option}")
    
    best_option_index = perplexities.index(min(perplexities))
    return chr(65 + best_option_index)


def get_pred(data, args, out_path, rank, pad_to_multiple=1):
    """
    Instantiate the model here. 1 get_pred should run on each gpu
    """
    fout = open(out_path, 'a', encoding='utf-8')
    print (f"get_pred len: {len(data)}")
    model = load_pretrained(args.model_type, args.checkpoint_path,  peft_adapter_path=args.peft_adapter_path)
    model.bfloat16()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.local_tokenizer)
    device = torch.device(f'cuda:{rank}')
    correct = 0
    total = 0
    with torch.cuda.device(rank):
        model.to(device)
        for item in tqdm(data):
            context = item['context']
            options = [item['choice_A'], item['choice_B'], item['choice_C'], item['choice_D'], ]
            prompt = template.replace('$DOC$', context.strip()).replace('$Q$', item['question'].strip())
            # For 0shot, no cot, no rag
            if args.generative:
                print("evaluating generative cloze")
                pred = evaluate_cloze(model, args, tokenizer, prompt, options, max_input_tokens=args.max_input_tokens, rank=rank)
            else:
                pred = evaluate_cloze_fwd(model, tokenizer, prompt, options, max_input_tokens=args.max_input_tokens, rank=rank, pad_to_multiple=pad_to_multiple)    

            item['pred'] = pred
            item['judge'] = item['pred'] == item['answer']
            print(f"pred: {item['pred']}, answer: {item['answer']}")
            item['context'] = context[:1000]
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            fout.flush()
            total += 1
            if item['judge']:
                correct += 1
            
            print({
                'acc': f"{correct / total:.2%}" if total > 0 else "0.00%",
                'correct': correct,
                'total': total
            })
def main():
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)
    mp.set_start_method('spawn', force=True)
    out_file = os.path.join(args.save_dir, "cloze_result.jsonl")
    if args.local_dataset is None:
        dataset = load_dataset('THUDM/LongBench-v2', split='train')
    else:
        dataset = json.load(open(args.local_dataset, 'r', encoding='utf-8'))

    data_all = [{"_id": item["_id"], "domain": item["domain"], "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length": item["length"], "question": item["question"], "choice_A": item["choice_A"], "choice_B": item["choice_B"], "choice_C": item["choice_C"], "choice_D": item["choice_D"], "answer": item["answer"], "context": item["context"]} for item in dataset]

    print(f"total data: {len(data_all)}")
    has_data = {}
    if os.path.exists(out_file):
        with open(out_file, encoding='utf-8') as f:
            has_data = {json.loads(line)["_id"]: 0 for line in f}

    data = []
    for item in data_all:
        if item["_id"] not in has_data:
            data.append(item)
    print(f"total lines to eval: {len(data)}")

    if args.n_proc == 1:
        get_pred(data, args, out_file, 0, pad_to_multiple=args.pad_to_multiple)
    else:
        data_subsets = [data[i::args.n_proc] for i in range(args.n_proc)]
        processes = []
        for rank in range(args.n_proc):
            p = mp.Process(target=get_pred, args=(data_subsets[rank], args, out_file, rank, args.pad_to_multiple))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--cot", "-cot", action='store_true') # set to True if using COT
    parser.add_argument("--no_context", "-nc", action='store_true') # set to True if using no context (directly measuring memorization)
    parser.add_argument("--rag", "-rag", type=int, default=0) # set to 0 if RAG is not used, otherwise set to N when using top-N retrieved context
    parser.add_argument("--n_proc", "-n", type=int, default=1)
    parser.add_argument("--local_dataset", type=str, default=None)
    parser.add_argument("--local_tokenizer", type=str, default=None)
    parser.add_argument("--max_input_tokens", type=int, default=65000)
    parser.add_argument("--pad_to_multiple", type=int, default=1)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--generative", action='store_true')    
    parser.add_argument('--peft_adapter_path', default=None, type=str, help='directory of the peft adapter')
    args = parser.parse_args()
    main()