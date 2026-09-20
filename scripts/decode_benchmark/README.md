# H20 FA2 decode 测速

这里提供此前 **Full-KV / IndexMem++ 原始 eager FA2 路径**的可移植测速入口，方便复现和定位瓶颈。
本次发布只整理路径、参数检查、调度和汇总；模型 forward、状态恢复、计时循环与原实验保持一致。
不包含 FA3 实验，也不包含此前改变生成轨迹的 decode 优化版。

## 此前已经发布到哪里

| 内容 | 已发布位置 | 本次处理 |
| --- | --- | --- |
| FA2 显存修改 | 本仓库 `refactor/gqa_indexer`：[32dfc111](https://github.com/Chooovy/kvpress/commit/32dfc1111b7ac473c19266792d1859f73a7ec132) | 已在当前分支，不重复提交或复制运行时代码 |
| 正式计时、输入、环境、校验、OOM 和原始脚本 | 论文仓库：[FA2 实验归档](https://github.com/XintongYang1010/-ICLR27-IndexMem-/tree/1788164e2f7bb896dc7f38b266e66eb309893db5/results/decode_h20_memory_fixed_20260918_2220)，最初结果提交 `07d9f978` | 保留在原仓库；这里提供能在其他机器运行的入口 |
| 最新图与结论解读 | 论文仓库：[1788164e](https://github.com/XintongYang1010/-ICLR27-IndexMem-/tree/1788164e2f7bb896dc7f38b266e66eb309893db5/results/decode_h20_seqlen_figures_20260918) | 不重复上传图片和绘图脚本 |

历史 `benchmark.py` 的 SHA-256 是
`323bf87321095b7228ada46340ad71e401fb96cffe94ec84da4e161cffead348`。
本目录的 `benchmark.py` 从它整理而来：改为导入所在仓库，增加参数和输入长度检查，删除未用于这次对照的窗口方法，
并按仓库格式排版。`summarize.py` 保留原始逐条校验与汇总公式，替换了依赖旧实验目录的读取入口。
`prepare.py`、`run_matrix.py` 将原先针对补测、特定机器的准备和调度逻辑整理为显式路径与 GPU UUID 参数。
历史占卡管理、SSH 收集和临时诊断脚本仍可在归档中查看，不作为可移植测速依赖。

## 运行时代码改了什么、为什么改、结果如何

`32dfc111` 已经包含两个改动：

- [`qi_flex_attention.py`](../../kvpress/presses/gqa_indexer/qi_flex_attention.py)：deadline 的整数直方图按 1,024 ranks
  分块处理，保留原来的整数累计顺序和 dtype，减少长输入时的临时张量。131,071 keys、8 heads 的 H20 检查中，
  **该 helper 的额外分配峰值**由 44.02 GiB 降至 0.35 GiB。
- [`evict_runner.py`](../../kvpress/presses/gqa_indexer/evict_runner.py)：消费完 prefill 捕获的 hidden state / kwargs 后释放引用，
  并在 `finally` 中移除临时 hook；KV、router 缓存仍按原逻辑保留。

这解决的是 **prefill 的内存峰值**，没有修改 decode 的评分、淘汰、CMP、精度或 FA2 内核。
16 个原先 OOM 的 128K/256K IndexMem++ 测点中有 15 个完成；Qwen3-8B / 256K / B=4 仍在 prefill OOM。
此前校验包括精确 helper 对照和 36 条一致的历史 256-step 输出轨迹；原实现长输入已 OOM，不能声称那些位置
也完成了原版完整模型轨迹对照。证据见[验证归档](https://github.com/XintongYang1010/-ICLR27-IndexMem-/tree/1788164e2f7bb896dc7f38b266e66eb309893db5/results/decode_h20_memory_fixed_20260918_2220/validation)。

最终 FA2 汇总是 **96 条方法结果：79 成功、17 OOM，1,185 次有效正式计时**；不完整测点的 5 次计时未纳入速度。
IndexMem++ 在双方均成功的 32 个配置中有 14 个更快。固定缓存使其 decode 吞吐随输入长度较平稳，但小 batch 的
评分、淘汰和调度开销可能使它慢于 Full-KV；这不是所有配置都加速的结论。原结果复用了既有成功测量，部分配对
来自不同轮次或 GPU，汇总中保留了这一信息。这次脚本发布没有新增 GPU 性能结果。

## 固定协议

- Qwen3-4B / 8B，BF16，FlashAttention **2.8.3**；PyTorch **2.8.0+cu128**、Transformers **4.57.3**、Triton **3.4.0**。
  在已有对应 CUDA 环境中从仓库根目录执行 `python -m pip install -e . --no-deps`。
  `reference.json` 记录实际模型配置、权重文件哈希、scorer revision 和环境版本；准备阶段强制核对。
- 默认 8K / 16K / 32K / 64K / 128K / 256K × batch 1 / 4 / 16 / 32 × 两模型 × 两方法：48 对、96 条方法结果。
  `K = 1024`。早期 8K/16K、batch 1/4/8/16/32/64 矩阵可以通过 CLI 参数运行。
- 三条 GovReport 输入，从原始数据的 **零基序号** 1、10、14 起按顺序串接，文档之间插入 EOS `151645`；
  每个长度取同一条 256K token 流的前缀。使用归档输入，不重新 tokenize 或替换文本。
- IndexMem++：平均预算 2,048 / KV head（含 64 CMP slots），mass allocation，floor 512，sink 4，local 128。
  每条序列有独立物理缓存；Full-KV 使用 `DynamicCache`。
- 每个输入预热两次、正式重复五次，每次 256 个完整 decode 步；greedy 选择 token，不因 EOS 提前结束。
- **计时外**：prefill、初次压缩、batch 状态复制/恢复、首 token。
  **计时内**：每步 model forward、评分、持续淘汰、`finish_step()` / CMP、token 选择和原有 trace 收集。
  计时前后同步 CUDA。每次恢复相同初始状态，并检查有限 logits、预算、独立页面、CMP 增量和重复输出哈希。
- 汇总为 **整批总生成 tokens ÷ 正式计时总秒数**，不是各次 token/s 的算术平均。
  OOM / 错误记录阶段和堆栈，不填零；只对双方成功的配置计算加速比。
- 保留模型原始 RoPE 和 40,960 位置配置；64K–256K 是吞吐压力测试，不代表长上下文质量验证。

## 准备输入与权重

以下命令从仓库根目录执行。先取得本地 `Qwen/Qwen3-4B`、`Qwen/Qwen3-8B` 底模目录；底模所有已记录文件均须通过
`reference.json` 的 SHA-256 校验。不要把模型或 scorer 权重加入 Git。

```bash
mkdir -p /data/indexmem-fa2-assets
# 使用已登录且能访问论文仓库的 gh；也可直接取已有归档中的同名文件。
gh api -H 'Accept: application/vnd.github.raw+json' \
  'repos/XintongYang1010/-ICLR27-IndexMem-/contents/results/decode_h20_memory_fixed_20260918_2220/raw/assets/prompts256k.json?ref=1788164e2f7bb896dc7f38b266e66eb309893db5' \
  > /data/indexmem-fa2-assets/prompts256k.json

hf download marcusguhao/Qwen3-4B-gate-IndexMem-rvkl16k-local128-b256-decay final.pt \
  --revision 7ffd3bf7ceb6f3ab0895ec619a3de0c8e0061721 --local-dir /data/indexmem-fa2-assets/indexmem4b
hf download marcusguhao/Qwen3-8B-gate-IndexMem final.pt \
  --revision 3e97c529bb03045394520c37c46a5e38d6484cbf --local-dir /data/indexmem-fa2-assets/indexmem8b

python scripts/decode_benchmark/prepare.py \
  --root /data/indexmem-fa2-run \
  --model-4b /data/models/Qwen3-4B --model-8b /data/models/Qwen3-8B \
  --checkpoint-4b /data/indexmem-fa2-assets/indexmem4b/final.pt \
  --checkpoint-8b /data/indexmem-fa2-assets/indexmem8b/final.pt \
  --prompts /data/indexmem-fa2-assets/prompts256k.json
```

`--root` 必须为空；准备脚本校验输入/权重/环境，创建轻量 symlink，并保存代码哈希、Git revision 和运行时代码 diff。
运行时再次校验，避免修改代码后误复用旧结果。修改代码做优化实验时，准备新的 run 目录。
准备脚本不要求访问 GPU，也不自动下载安装包或权重。

## 测速与汇总

先查看设备，再把下面的 UUID 替换为自己已释放、可用的 GPU；脚本不操作占卡 supervisor，也不会结束其他 GPU 进程。
一个 UUID 对应一个独立 worker，同配置的 Full-KV、IndexMem++ 在同一张卡上顺序运行，不同配置并行。

```bash
nvidia-smi -L
python scripts/decode_benchmark/run_matrix.py --root /data/indexmem-fa2-run --dry-run
python scripts/decode_benchmark/run_matrix.py --root /data/indexmem-fa2-run \
  --gpus GPU-REPLACE-WITH-FIRST-UUID GPU-REPLACE-WITH-SECOND-UUID

python scripts/decode_benchmark/summarize.py --root /data/indexmem-fa2-run
```

`--dry-run` 只打印计划，不校验输入，也不启动 GPU。正式运行限制 CPU 线程，逐任务保存 `results/*.jsonl` 和
`logs/*.log`。OOM/普通任务错误后继续其余配置；发现所选 GPU 上有外部进程时停止本轮调度，只清理本轮子进程。
`Ctrl-C` / SIGTERM 同样清理本轮子进程。调度器用文件锁防止同一目录被两个调度器并发写入。

重启相同命令时跳过已成功测点，保留失败/中断记录；显式指定 `--only` 才重试失败测点，仍不会重测已完成结果：

```bash
python scripts/decode_benchmark/run_matrix.py --root /data/indexmem-fa2-run \
  --gpus GPU-REPLACE-WITH-UUID --only 8b_l262144_b4_indexmem

# 早期短序列矩阵；使用独立 run 目录可避免与长度扫描的结果混淆。
python scripts/decode_benchmark/run_matrix.py --root /data/indexmem-fa2-run \
  --gpus GPU-REPLACE-WITH-UUID --lengths 8192 16384 --batches 1 4 8 16 32 64

# 单测一个方法；同样使用新文件名，不覆盖已有 JSONL。
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-UUID OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
python scripts/decode_benchmark/benchmark.py --root /data/indexmem-fa2-run \
  --size 8b --length 32768 --batch 16 --method indexmem \
  --output /data/indexmem-fa2-run/results/8b_l32768_b16_indexmem.attempt1.jsonl
```

汇总脚本生成 `summary.csv`、`summary.json`、`attempts.json`；严格要求完整的 3 × 5 次正式计时和 3 × 2 次预热。
`benchmark.py` 的 `--steps/--documents/--repeats/--warmups` 可用于短 smoke，但其非正式协议日志不能混入上述正式汇总。
显存列是 decode 计时段的分配/保留峰值；不要把它解释为整个请求的 prefill 峰值。
绘图沿用此前论文仓库的[绘图 CLI](https://github.com/XintongYang1010/-ICLR27-IndexMem-/blob/1788164e2f7bb896dc7f38b266e66eb309893db5/results/decode_h20_seqlen_figures_20260918/plot_seqlen.py)。

## 本次发布验证

对原脚本排除上述路径/输入检查、未使用的窗口分支和局部变量重命名后，Full-KV / IndexMem++ 执行与计时 AST 一致。
新汇总入口重读全部 96 条历史结果，79 条成功记录的吞吐、延迟和显存与原汇总一致；17 条 OOM 均没有速度值。
CLI、Black、Flake8 通过；本地 CPU 子进程验证了并行、同卡配对、OOM 后继续、完成后跳过、显式重试和输入漂移拒绝。
这些是发布/调度检查，没有重新跑 GPU 性能实验；临时验证代码没有提交到仓库。
