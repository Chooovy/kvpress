# Query-Independent Indexer with Cross-Replay Training

## Goal

目标是训练一个 **query-independent per-token indexer**：

\[
s_i = f_\theta(h_i)
\]

其中每个 token 的 score 在 token 写入 KV cache 时即可确定，之后不依赖未来 query。

这种形式天然适合 **KV eviction**：

- token 一旦被判定低价值，可以永久删除；
- 不需要每个新 query 重新计算 token importance；
- inference 时直接对 \(s_i\) 做 Top-K / threshold 即可压缩 KV cache。

---

## Why the Indexer Architecture Is Not the Core

当 indexer 被限制为 query-independent：

\[
s_i=f_\theta(h_i)
\]

其表达形式本身可以非常简单，例如：

\[
s_i=w^\top h_i
\]

或一个小 MLP。

此时真正关键的问题不是：

> 如何设计一个更复杂的 scorer？

而是：

> **什么训练目标能够让 \(s_i\) 学到“这个 token 对未知未来 query 是否值得保留”？**

因此核心应从 **indexer architecture** 转向 **supervision / loss semantics**。

---

## Ordinary LM Loss Learns Future Predictive Utility

普通 causal LM loss：

\[
\mathcal L_{\rm LM}
=
-\sum_t \log p(x_{t+1}|x_{\le t})
\]

可以端到端训练 query-independent score。

对于 token \(i\)，其梯度来自后续 token：

\[
\frac{\partial \mathcal L}{\partial s_i}
\sim
\sum_{t>i}
U(KV_i,q_t)
\]

因此 LM loss 学到的物理意义是：

> **token \(i\) 对自然 future continuation 的预测价值。**

但它有两个限制：

1. **只有 causal triangle supervision**  
   token \(i\) 只能被 \(i\) 之后的 query 监督。

2. **future-query distribution mismatch**  
   训练时的 query 是自然 next-token continuation，  
   但 eviction 后的 KV cache 需要服务未来未知的 QA / retrieval / reasoning queries。

因此 LM loss 更接近：

\[
s_i
\approx
\mathbb E_{q\sim Q_{\rm next-token}}
[U(KV_i,q)]
\]

而不一定是最理想的 query-agnostic eviction importance。

---

## Why Cross-Replay Loss Is Better Aligned

构造：

\[
[C;C']
\]

其中第一遍 \(C\) 正常 prefill，产生待压缩的 KV cache：

\[
\{KV_i\}_{i=1}^N
\]

以及 query-independent scores：

\[
s_i=f_\theta(h_i)
\]

训练时不直接 hard-evict，而是得到 soft gate：

\[
g_i = G(s_i)
\]

第二遍 \(C'\) **只能 attend 第一遍 \(C\) 的 gated KV cache**，不能 attend 自己的历史 KV。

例如：

\[
A_{ji}
=
\operatorname{softmax}
\left(
q_j^{\prime\top}k_i
+
\alpha\log(g_i+\epsilon)
\right)
\]

然后只对第二遍计算 reconstruction LM loss：

\[
\mathcal L_{\rm replay}
=
-\sum_j \log p(x'_{j+1}|KV_C^{gated},x'_j)
\]

### 核心优势：Triangle → Rectangle

普通 LM：

\[
KV_i
\leftarrow
\{q_{i+1},\ldots,q_N\}
\]

只有后续 query 能监督 token \(i\)。

Cross-replay：

\[
KV_i
\leftarrow
\{q'_1,\ldots,q'_N\}
\]

第一遍所有 KV 都可以被第二遍所有 replay queries 使用。

因此 supervision 从：

\[
\text{causal triangle}
\]

变成：

\[
\text{full cross-context rectangle}
\]

同一个 \(s_i\) 会聚合整段 context 中所有 replay query 对它的需求：

\[
s_i
\sim
\sum_{j=1}^{N}
U(KV_i,q'_j)
\]

这更接近 query-independent eviction 真正需要学习的：

> **一个 token 对一组未知未来 queries 的总体可用价值。**

---

## Interpretation

可以统一写成：

\[
s_i
=
\mathbb E_{q\sim Q}
[U(KV_i,q)]
\]

不同 loss 的核心区别其实是 **训练时采用什么 query distribution \(Q\)**：

### Ordinary LM

\[
Q=Q_{\rm next-token}
\]

学习：

> token 对自然未来 continuation 的价值。

### Cross-Replay

\[
Q=Q_{\rm reconstruction}
\]

学习：

> token 对整个 context 被重新读取 / 恢复时的总体价值。

对于 **query-independent KV eviction**，后者通常更对齐，因为 eviction 的目标本身就是：

\[
\text{保留一个固定 KV subset}
\rightarrow
\text{服务多个未知未来 queries}
\]

---

## Recommended Formulation

保持 indexer 极简：

\[
s_i=w^\top h_i
\]

训练时使用：

\[
\text{soft budgeted gate}
+
\text{cross-only replay}
+
\mathcal L_{\rm replay}
\]

即：

\[
C
\rightarrow
s_{1:N}
\rightarrow
g_{1:N}
\rightarrow
KV_C^{gated}
\rightarrow
C'
\rightarrow
\mathcal L_{\rm replay}
\]

其中需要额外加入 **cache budget / concentration constraint**，避免所有 \(g_i\) 都接近 1。

Inference 时再直接：

\[
\operatorname{TopK}(s)
\rightarrow
\text{hard KV eviction}
\]

---

## Core Hypothesis

> **对于 query-independent indexer，architecture 不是核心，training objective 才是核心。**

普通 LM loss 学习的是：

\[
\text{future predictive utility}
\]

而 cross-replay loss 将所有 replay queries 对同一个固定 KV cache 的需求聚合起来，学习的是更接近：

\[
\boxed{
\text{query-agnostic reusable cache utility}
}
\]

因此它理论上比 ordinary causal LM loss 更适合训练可直接用于 KV eviction 的 query-independent indexer。
