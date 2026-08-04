from transformers import pipeline
from kvpress import ExpectedAttentionPress
from kvpress import LearnedScorePress
from kvpress.presses.learned_score_press import load_model_with_learned_press
from transformers import AutoTokenizer, AutoModelForCausalLM

# device = "cuda:0"
# model = "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
# model_kwargs = {"attn_implementation": "flash_attention_2", "torch_dtype": "bfloat16"}
# pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

# context = "Here is the question: "
# question = "What is the capital of France?"

# for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     press = ExpectedAttentionPress(compression_ratio=ratio)
#     answer = pipe(context, question=question, press=press)["answer"]
#     print(f"compression ratio: {ratio}\n")
#     print(answer)
#     print("-" * 100)





# # model_path = "/aifs4su/guhao/checkpoints/dma_8gpu"
# # model_path = "/aifs4su/guhao/checkpoints/learned_press"
# model_path = "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
# model_kwargs = {"attn_implementation": "eager", "torch_dtype": "bfloat16", "device_map": "cuda:0"}
# model, tokenizer = load_model_with_learned_press(model_path, model_kwargs=model_kwargs)
# # model, tokenizer = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs), AutoTokenizer.from_pretrained(model_path)

# context = "Here is the question: "
# question = "What is the capital of France?"
# full_text = context + question
# inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

# outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)
# answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(answer)
# press = False
# for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     press = LearnedScorePress(compression_ratio=ratio)
#     with press(model):
#         press = True
#         outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.eos_token_id)

#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     if press:
#         print(f"using LearnedScorePress")
#     print(f"compression ratio: {ratio}\n")
#     print(f"using checkpoint: {model_path}")
#     print(answer)
#     print("-" * 100)


# model_path = "/aifs4su/guhao/Models/Llama-3.2-1B-Instruct"
model_path = "/aifs4su/guhao/checkpoints/llama3-1b-instruct-learned-score"
model_kwargs = {"attn_implementation": "eager", "dtype": "bfloat16", "device_map": "cuda:0"}
model, tokenizer = load_model_with_learned_press(model_path, model_kwargs=model_kwargs)

pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)

context = "Here is the question: "
question = "What is the capital of France?"

for ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    press = LearnedScorePress(compression_ratio=ratio)
    result = pipe(context, question=question, press=press, max_new_tokens=100)
    print(f"Compression ratio: {ratio}\n")
    print(f"Answer: {result['answer']}")
    print("-" * 100)

# pipe = pipeline(
#     "kv-press-text-generation",
#     model=model_path,
#     device="cuda:0",
#     model_kwargs={"attn_implementation": "eager", "torch_dtype": "bfloat16"}
# )

# press = LearnedScorePress(compression_ratio=0.5)
# result = pipe(context, question=question, press=press, max_new_tokens=100)
# print(result['answer'])