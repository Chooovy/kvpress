# 目标

在做 KV-cache eviction 后，避免被 evict 的信息永久丢失：

* **保留**一部分“重要 token”的 KV（`K_save, V_save`）用于精确 softmax attention；
* **把被 evict 的 KV**（`K_evict, V_evict`）写入一个**小的 per-sequence memory 状态**（latent state），后续 query 来时通过 memory 读出一个补偿项 `M(q)`，与 `o_save` 融合，近似 full attention 输出。

---

# 核心思想：memory 必须是“有状态”的（stateful）

只训练一个静态网络 `M(q)`（推理时不更新、不接收 evict 内容）只能学到“平均补偿”，**不能记住本序列中新出现的 evicted KV**。
所以采用 **stateful memory**：推理时只更新一个小状态 `A`（不做反向，不更新主模型参数），让 `A` 真实承载被 evict 的 KV 信息。

---

# Memory 结构（推荐默认）

对每个 Transformer layer、每个 attention head 维护一套 memory 状态（也可以 layer 内共享，先按 head 做最直观）：

* 选择一个特征映射 `φ(·)`：把 key/query 投影到较小维度 `dφ`（可选 `dφ = d_head` 或更小）。
* Memory state：

  * `A ∈ R^{dφ × d_v}`：存“key→value”的关联累积（fast-weights 矩阵）
  * `b ∈ R^{dφ}`：可选，存归一化分母（用于稳定缩放/近似 softmax mass）
* 额外的稳定组件：

  * `gate g(q) ∈ [0,1]`：控制 memory 注入强度（按 head 的标量、或按 token/head 的向量都可）

> 状态 `A,b` 是 **per-sequence** 的：每次新请求/新 sample 初始化为 0；在 prefill+decode 过程中持续更新。

---

# 写入（eviction 时如何把 KV “存进 memory”）

当你的 eviction policy 决定从 KV-cache 中移除一批 token（或 block）时，对每个被移除 token 的 `(k_evict, v_evict)` 执行写入更新。推荐两个写入规则：**additive（简单）**和 **delta-rule（更强，抗干扰）**。默认建议 delta-rule。

## 写入规则（additive / Hebbian）

对每个被 evict 的 token：

* `A ← λ A + η · φ(k) ⊗ v`
* `b ← λ b + η · φ(k)`（若启用分母）

其中：

* `λ ∈ (0,1]`：遗忘系数（防止 A 无界增长 / 腾空间）
* `η > 0`：写入强度（可设常数，也可由 token importance / eviction score 决定）


---

# 读出（attention 时如何用 memory 补回 evicted 部分）

对每个 attention 计算位置（每个 query `q`）：

1. 正常算保留部分（save）的 attention：

* `o_save(q) = softmax(q K_save) V_save`（你的标准实现）

2. 从 memory 读出补偿向量：

* 先算 `u = φ(q)`
* 若启用分母：

  * `m(q) = (u^T A) / (u^T b + ε)`
* 若不启用分母：

  * `m(q) = u^T A`
    （`m(q)` 的维度与 `o_save` 对齐：通常是 head 的 value dim）

3. 融合（推荐带 gate，避免 scale 问题）：

* `o(q) = o_save(q) + g(q) ⊙ m(q)`

> gate `g(q)` 可以是：每 head 一个标量；或由 `q`、token importance、甚至“evict 总量”等特征预测。目标是：当 evicted 信息几乎无用时 `g≈0`，当缺失很大时 `g` 提升。

---

# 训练目标（让 memory 学的“就是缺失贡献”，对齐 full attention）

你提的方案 A（residual distillation）是正确方向，但要强调：**M 必须依赖 state（A,b）**，否则还是“静态补偿器”。

## Teacher 信号

训练阶段需要一个 teacher 输出 `o_full(q)`（可以是真正 full KV attention，或更大 budget 的 cache/更高保真近似）。同时计算 `o_save(q)`（按你的 eviction policy 只对 save 做 attention）。

定义 residual：

* `Δ(q) = o_full(q) - o_save(q)`

## 主损失（核心）

让 memory 分支（含 gate）拟合 residual：

* `L_main = MSE( g(q) ⊙ m(q), Δ(q) )`

这一步自动把 softmax 归一化缺失、missing-mass、以及 evicted 内容的贡献都蒸馏进 memory 读出里。


---

# 推理流程（重点：不做反向，只更新状态）

对每条请求/序列：

1. 初始化每层每 head 的 `A=0, b=0`
2. Prefill 阶段：

   * 逐 token 生成 KV
   * 触发 eviction 时，把被 evict 的 `(k,v)` 按写入规则更新到 `(A,b)`
3. Decode 阶段：

   * 对每个新 token 的 query：

     * 用 `K_save,V_save` 做 `o_save`
     * 用当前 `(A,b)` 读 `m(q)`
     * 融合得到输出 `o`
   * 当 cache 继续满、继续 evict 时，持续写入 `(A,b)`

> 关键：推理时 memory “学习”的唯一形式就是更新 `A,b`（状态更新），不需要梯度、不改主模型参数。

---

# 实用建议（避免踩坑）

* **把 memory 只挂在部分 layer（比如中后层）**通常更划算：早期层更像词法/局部特征，长期语义记忆更需要后层。
* `dφ` 不必等于 `d_head`：可以更小（低秩）来省算力/省显存，但会损失可召回精度。
* eviction policy 若本身就按“低注意力/低重要度”驱逐，additive 版本就可能够用；若你要救 needle / 实体绑定 / KV 精准回忆，优先 delta-rule + decay +（可选）分母。
* 如果你发现 memory 输出过大/不稳定：优先加 gate，其次启用分母 `b`，再调小 `η` 或增大 decay（减小 `λ`）。
