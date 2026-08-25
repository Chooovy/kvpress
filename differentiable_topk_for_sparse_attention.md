# Differentiable Top-K for Sparse Attention Indexers
## 从 MoE 可微路由到 Sparse Attention Top-K 的实现说明

> 目标读者：一个**没有此前 MoE Top-K 讨论上下文**、但需要直接实现 sparse attention 中可微 Top-K / E2E indexer 训练方案的 agent。  
> 目标：把 MoE、subset sampling、differentiable sorting/ranking 中与 **Top-K 离散选择可训练**有关的方法，统一成适合 sparse attention indexer 的实现框架，并给出推荐优先级。

---

# 0. 任务背景与核心目标

我们考虑一个 sparse attention indexer。

对 query \(q_t\)，indexer 给历史 token / KV 一个分数：

\[
s_{t,j}=f_\theta(q_t, k_j^{idx})
\]

或更一般地：

\[
s_{t,j}=f_\theta(h_t,h_j,\text{history})
\]

最终推理阶段需要选择固定预算 \(K\) 个 token：

\[
S_t=\operatorname{TopK}(s_t),\qquad |S_t|=K
\]

然后只对这些 KV 做真实 attention：

\[
o_t=
\operatorname{softmax}\left(
\frac{q_tK_{S_t}^\top}{\sqrt d}
\right)V_{S_t}
\]

训练目标希望直接使用最终语言模型损失：

\[
\mathcal L_{\rm LM}
\]

而不是让 indexer 单纯蒸馏 backbone attention weights。

---

# 1. 第一性原理：真正的问题不是 gate，而是离散 subset 的 credit assignment

真正目标：

\[
\min_\theta
\mathbb E_x
\left[
\mathcal L_{\rm LM}
\left(
\operatorname{Model}(x;\operatorname{TopK}(s_\theta))
\right)
\right]
\]

问题在于：

\[
m_j=\mathbf 1[j\in\operatorname{TopK}(s)]
\]

几乎处处满足：

\[
\frac{\partial m_j}{\partial s_i}=0
\]

所以：

\[
\frac{\partial\mathcal L}{\partial\theta}
=
\frac{\partial\mathcal L}{\partial m}
\frac{\partial m}{\partial s}
\frac{\partial s}{\partial\theta}
\]

中的

\[
\frac{\partial m}{\partial s}
\]

被 hard Top-K 截断。

因此所有“Top-K 可微分方案”本质上都在解决：

\[
\boxed{
\text{如何为离散 selection 构造一个可用的 gradient estimator / relaxation}
}
\]

---

# 2. 一个重要建模原则：selection 与 attention contribution 应尽量解耦

不推荐默认采用：

\[
a_j=q^\top k_j
\]

\[
\tilde a_j=a_j+\log g_j
\]

再让 \(g_j\) 同时承担：

1. token 是否被选中；
2. token 被选中后在 attention 中贡献多少。

原因：

- 所有 gate 同比例缩放对 softmax attention 不可辨识；
- constant gate 是天然 no-op configuration；
- selection 与 contribution 混在一起后，训练目标有额外 gauge freedom；
- 推理真正需要的是一个 subset / ranking，不需要 gate 的绝对幅值。

更干净的定义：

\[
s
\rightarrow
S=\operatorname{TopK}(s)
\rightarrow
\operatorname{SparseAttention}(S)
\]

其中：

- **indexer score 只决定 selection / ranking**
- **原 attention logits 决定 selected token 的 contribution**

这是后续所有方案的推荐默认建模方式。

---

# 3. 总体方法谱系

可以把 Top-K 可训练方案分为 7 类：

1. **Selected-gate proxy**
2. **Straight-Through Estimator (STE)**
3. **Routing-gradient estimator**
4. **Counterfactual / dense-backward approximation**
5. **Continuous differentiable Top-K**
6. **Stochastic exact-\(K\) subset + marginal gradient**
7. **Remove Top-K：continuous sparse routing**

对 sparse attention indexer 最值得优先实现的是：

\[
\boxed{\text{Candidate STE}}
\]

\[
\boxed{\text{SparseMixer / GRIN-like routing gradient}}
\]

\[
\boxed{\text{Sander / LapSum differentiable Top-K}}
\]

\[
\boxed{\text{SIMPLE / ProbMoE exact-K subset}}
\]

---

# 4. Baseline 0：传统 selected-gate proxy

传统 sparse MoE / sparse attention 常见做法：

\[
S=\operatorname{TopK}(s)
\]

然后只在 selected items 内：

\[
\pi_i=
\operatorname{softmax}_{i\in S}(s_i)
\]

输出：

\[
y=\sum_{i\in S}\pi_i E_i(x)
\]

或 attention 中：

\[
o=
\sum_{j\in S}
\alpha_jV_j
\]

如果把 indexer score 再参与 selected branch 的计算，就有：

\[
\frac{\partial L}{\partial s_j}\neq0,\qquad j\in S
\]

但：

\[
\frac{\partial L}{\partial s_j}=0,\qquad j\notin S
\]

## 核心缺陷

它只能回答：

> selected token 内谁应该更大 / 更小？

却不能回答：

> 当前没选中的 token 是否应该替换某个 selected token？

即没有真正的 **selection boundary gradient**。

这是所有后续方法想解决的问题。

---

# 5. 方法族 A：Straight-Through Estimator (STE)

## 5.1 最基础 STE

定义 hard mask：

\[
m_h=\operatorname{TopKMask}(s)
\]

再定义一个 differentiable soft surrogate：

\[
m_s=\operatorname{SoftTopK}(s)
\]

组合：

\[
m=
m_s+\operatorname{sg}(m_h-m_s)
\]

其中 \(\operatorname{sg}\) 是 stop-gradient。

于是：

### Forward

\[
m=m_h
\]

### Backward

\[
\frac{\partial m}{\partial s}
=
\frac{\partial m_s}{\partial s}
\]

这是最直接的：

\[
\boxed{\text{hard forward + soft backward}}
\]

---

## 5.2 推荐：不要只对 mask 做 STE，而对 attention output 做 STE

更稳定的形式：

\[
o_{\rm hard}
=
\operatorname{SparseAttention}(S_h)
\]

\[
o_{\rm soft}
=
\operatorname{SoftSparseAttention}(s)
\]

组合：

\[
o=
o_{\rm soft}
+
\operatorname{sg}(o_{\rm hard}-o_{\rm soft})
\]

这样：

- forward 下游 hidden state 完全来自真实 sparse model；
- backward 则通过 soft branch 给 indexer 梯度；
- normalization 等 attention 内部结构也包含在 surrogate 中。

推荐作为第一版实现。

---

## 5.3 Candidate-STE：适合长序列 sparse attention

不能对 \(L=64K\sim1M\) 所有 token 做 dense soft attention。

因此先取：

\[
C=\operatorname{TopM}(s)
\]

其中：

\[
M=2K\sim4K
\]

然后：

- hard forward：只用 Top-K；
- soft backward：只在 candidate Top-M 内构造 differentiable selection。

即：

\[
S=\operatorname{TopK}_C(s)
\]

\[
m_s=
\operatorname{SoftTopK}(s_C,K)
\]

复杂度从全局 \(L\) 降为 candidate size \(M\)。

---

## 5.4 Candidate pool 必须有 exploration

如果：

\[
C=\operatorname{TopM}(s)
\]

那么 \(C\) 之外 token 永远没有梯度。

推荐：

\[
C=
C_{\rm top}
\cup
C_{\rm random}
\cup
C_{\rm structural}
\]

例如：

- 70–90% current Top-M；
- 5–20% random token；
- recent tokens；
- sink tokens；
- chunk-stratified samples；
- high-attention oracle sample（只训练时）；
- periodically full candidate refresh。

关键原则：

\[
\boxed{\text{没有 support，就没有 credit assignment}}
\]

---

# 6. 方法族 B：DenseMixer / Dense Backprop 类型

这一族不是单纯把 Top-K smooth 化，而是显式估计：

> 如果一个当前没选中的 branch 被选中，它会怎样改变输出 / loss？

这正是 sparse attention indexer 最需要的 counterfactual information。

---

## 6.1 DenseMixer-style

MoE 中的思想：

- forward 仍然 hard Top-K；
- training 时额外计算更多 / 全部 experts；
- 用 STE 或 dense surrogate 给 router 更完整的 gradient；
- inference 无额外开销。

转到 sparse attention：

当前 selected：

\[
S=\operatorname{TopK}(s)
\]

候选：

\[
C=\operatorname{TopM}(s),\qquad M>K
\]

hard output：

\[
o_h=\operatorname{Attn}(S)
\]

训练时对 \(C\) 内所有 token 计算：

\[
o_C
\]

并据此构造 selection gradient。

### 最推荐的 sparse-attention 版本

不要 dense 到全序列，而是：

\[
\boxed{\text{Candidate-DenseMixer}}
\]

只 dense 到 \(M=2K\sim4K\)。

---

## 6.2 Counterfactual swap

直接构造：

\[
i\in S,\qquad j\in C\setminus S
\]

交换：

\[
S'=S-\{i\}+\{j\}
\]

计算：

\[
\Delta_{ij}
=
\mathcal L(S')-\mathcal L(S)
\]

如果：

\[
\Delta_{ij}<0
\]

说明 \(j\) 应该排在 \(i\) 前面：

\[
s_j>s_i
\]

可训练 pairwise ranking loss：

\[
L_{\rm rank}
=
w_{ij}
\operatorname{softplus}(s_i-s_j)
\]

其中：

\[
w_{ij}=|\Delta_{ij}|
\]

或 clip 后权重。

这是一种非常本质的 indexer 训练方式：

\[
\boxed{\text{直接学 ranking，而不是学 gate 幅值}}
\]

---

## 6.3 Dense Backprop-style approximation

真正计算每个 counterfactual swap 太贵。

MoE 中 Dense Backprop 的思想：

\[
E_i(x)
\]

没执行，也可以近似它的输出：

\[
\hat E_i(x)
\]

然后利用：

\[
\hat E_i(x)
\]

估计 router gradient。

在 sparse attention 中可以对应为：

\[
V_j,\quad
\Delta o_j,\quad
\Delta L_j
\]

的廉价预测器。

例如构造：

\[
\widehat{\Delta o}_j
=
g_\phi(h_q,h_j,v_j)
\]

或者：

\[
\widehat{\Delta L}_j
=
g_\phi(h_q,h_j,\text{summary})
\]

这个 proxy **只参与 backward / auxiliary target**，不参与 inference。

---

## 6.4 Default-output / EMA proxy

MoE 的另一种思路：

未执行 expert 的 output 用历史 EMA 近似。

Sparse attention 类比：

- 对 chunk / token 类型维护 expected value contribution；
- 对 unselected token 使用 running estimate；
- 仅用于 backward surrogate。

不过 token identity 变化大，因此比 MoE expert 更难直接用 EMA；
更适合 chunk-level / cluster-level approximation。

---

# 7. 方法族 C：SparseMixer / GRIN —— 直接估计 routing gradient

## 7.1 核心思想

不是把：

\[
\operatorname{TopK}
\]

替换成 soft version。

而是：

\[
\boxed{\text{forward 保持 sparse discrete routing，backward 直接构造 routing gradient estimator}}
\]

也就是显式估计：

\[
\widehat{
\frac{\partial L}{\partial s}
}
\]

使其包含由于 routing decision 改变而造成的 loss 变化。

---

## 7.2 SparseMixer

SparseMixer 的关键出发点：

传统 sparse routing 忽略：

\[
\frac{\partial m}{\partial s}
\]

SparseMixer 用数值 ODE / midpoint-style estimator 近似这部分梯度，同时保持 sparse forward compute。

对 sparse attention 的对应思想：

\[
S=\operatorname{TopK}(s)
\]

forward：

\[
o=\operatorname{Attn}(S)
\]

backward：

不把 Top-K 当 constant，而根据 output 对 routing decision 的敏感性构造：

\[
\hat g_s
\approx
\frac{\partial L}{\partial s}
\]

---

## 7.3 GRIN / SparseMixer-v2

GRIN 进一步强调：

\[
\boxed{\text{gating-weight gradient}\neq\text{routing gradient}}
\]

即：

selected token 内 softmax weight 的梯度，只是 routing-gradient 的 proxy。

真正需要的是：

\[
\text{如果 token 进入 / 离开 selected set，会发生什么？}
\]

这是 indexer E2E 训练最重要的 conceptual reference。

---

## 7.4 Sparse attention 实现建议

第一版不要直接复刻完整 SparseMixer 数学。

建议先实现一个可验证的 finite-difference routing estimator：

对 boundary token：

\[
i=s_{(K)},\qquad
j=s_{(K+1)}
\]

构造小 perturbation：

\[
s_i\leftarrow s_i-\epsilon
\]

\[
s_j\leftarrow s_j+\epsilon
\]

观察 selection flip 后：

\[
\Delta L
\]

再和你设计的 analytical estimator 比较。

这可以作为：

\[
\boxed{\text{gradient-estimator correctness oracle}}
\]

---

# 8. 方法族 D：Continuous differentiable Top-K

这一族直接定义一个连续算子：

\[
\tilde m(s)
\approx
\operatorname{TopKMask}(s)
\]

要求：

\[
\frac{\partial\tilde m}{\partial s}\neq0
\]

然后用普通 autograd。

主要方法包括：

1. DSelect-k
2. Optimal-Transport Soft Top-k
3. differentiable sorting / NeuralSort / SoftSort
4. Sander et al. sparse differentiable Top-k
5. LapSum / Fast LapSum
6. sparsemax / entmax / Sparsegen 类 sparse projection

---

# 9. DSelect-k

## 思路

不使用显式 hard Top-K，而是构造一个连续、可导、最多选择 \(k\) 个 expert 的 sparse gate。

核心特点：

- cardinality-aware；
- differentiable；
- 输出可以 sparse；
- 通过特殊 binary encoding 参数化 expert subset。

## 对 sparse attention 的适用性

概念上重要，但：

\[
L=64K\sim1M
\]

时 binary encoding / expert-index parameterization 不一定自然。

更适合作为：

- MoE related work；
- small candidate pool 内的 differentiable selector。

不建议直接作为全序列 indexer 第一版。

---

# 10. Optimal Transport Soft Top-k

把 Top-K 写成一个 transport / assignment 问题。

通过 entropic regularization：

\[
\operatorname{TopK}(s)
\rightarrow
\operatorname{SoftTopK}_\epsilon(s)
\]

当：

\[
\epsilon\to0
\]

时逼近 hard Top-K。

优点：

- 数学形式规整；
- 可通过 Sinkhorn / implicit differentiation 求梯度；
- 可以显式表达 fixed mass / budget。

缺点：

- 长序列成本高；
- 输出通常 dense；
- 不适合直接在 \(L=128K\) 上运行。

推荐只作为小 candidate pool 的 baseline。

---

# 11. Differentiable Sorting：NeuralSort / SoftSort 一类

Top-K 可以通过 differentiable sorting 获得。

大致：

\[
P_{\rm sort}(s)
\]

用 soft permutation matrix 近似排序。

然后：

\[
m_i=
\sum_{r=1}^{K}
P_{ri}
\]

得到 soft Top-K membership。

优点：

- conceptually simple；
- 可以直接把 ranking 变成 differentiable。

缺点：

- 常见实现复杂度较高；
- soft permutation 往往 dense；
- 对超长 sequence 不合适。

适合作为理论 baseline，不推荐直接大规模使用。

---

# 12. Sander et al.：Fast, Differentiable and Sparse Top-k

这是非常值得优先看的通用算子。

## 核心思想

把 Top-K 写成一个线性优化：

\[
\operatorname{TopK}(s)
=
\arg\max_{y\in P}
\langle s,y\rangle
\]

其中 \(P\) 是相应的 polytope / permutahedron。

加 regularization：

\[
y_\epsilon
=
\arg\max_{y\in P}
\langle s,y\rangle
-
\epsilon\Omega(y)
\]

得到：

- differentiable；
- sparse；
- 比普通 softmax 更贴近真正 Top-K。

## 对 sparse attention 的价值

这是非常自然的：

\[
\boxed{\text{sparse differentiable Top-K operator}}
\]

可直接替换：

```python
hard_mask = topk(scores)
```

为：

```python
soft_sparse_mask = sparse_differentiable_topk(scores, k)
```

第一版建议只在 candidate Top-M 内运行。

---

# 13. LapSum / SoftMoE / Fast LapSum

## 13.1 LapSum

LapSum 提供 differentiable ranking / Top-K mapping。

典型输出：

\[
p_i\in[0,1]
\]

并满足：

\[
\sum_i p_i=K
\]

因此是一个**固定 soft budget**。

---

## 13.2 SoftMoE 2026

SoftMoE 使用 LapSum-style soft Top-K，并进一步：

- 让 \(K\) / layer compute 可以学习；
- 通过 global compute budget 控制总激活；
- 对很小的 soft probabilities 再做 truncation 以获得 sparse execution。

需要注意：

\[
p_i\mathbf1[p_i>\tau]
\]

最终 threshold 仍然引入新的不可导边界。

因此 SoftMoE 更准确地说是：

\[
\boxed{\text{differentiable soft routing + hard truncation}}
\]

而不是完全消灭所有 selection discontinuity。

---

## 13.3 Fast LapSum

Fast LapSum 的核心价值在于：

\[
N
\]

可以非常大。

MoE 中 expert 数一般几十到几百，因此未必最需要；
但 sparse attention：

\[
L=128K,\;1M
\]

恰好非常适合这种 million-scale differentiable Top-K 算子。

如果要尝试：

\[
\boxed{\text{full-sequence differentiable Top-K}}
\]

Fast LapSum 是优先候选之一。

---

# 14. Sparsemax / Entmax / Sparsegen / LD-MoLE 类

这类方法不要求 fixed exact K，而是做 simplex sparse projection。

以 Sparsegen-like 形式：

\[
p=
\arg\min_{p\ge0,\mathbf1^\top p=1}
\|p-u\|^2-\lambda\|p\|^2
\]

解具有 threshold：

\[
p_i=
\left[
\frac{u_i-\tau}{1-\lambda}
\right]_+
\]

因此自然产生：

\[
p_i=0
\]

即 sparse routing。

## 优点

- differentiable；
- naturally sparse；
- active set 可以随 token / layer 动态变化。

## 缺点

不能天然保证：

\[
|S|=K
\]

所以更适合：

\[
\text{learnable sparsity}
\]

而不是：

\[
\text{fixed exact-K budget}
\]

如果 sparse attention inference 必须严格固定 K，则优先级低于 exact-K 方法。

---

# 15. 方法族 E：Stochastic relaxation —— Gumbel Top-K

## 15.1 Gumbel-Max

categorical sampling：

\[
i^*
=
\arg\max_i
(\log w_i+g_i)
\]

其中：

\[
g_i\sim\operatorname{Gumbel}(0,1)
\]

---

## 15.2 Gumbel Top-K

推广到 subset：

\[
S=
\operatorname{TopK}(s+g)
\]

再使用：

\[
\operatorname{RelaxedTopK}
\]

构造 differentiable approximation：

\[
\tilde m=
\operatorname{RelaxedTopK}(s+g,T)
\]

当：

\[
T\to0
\]

趋向 k-hot。

---

## 15.3 ST-Gumbel

forward：

\[
m_h=\operatorname{TopK}(s+g)
\]

backward：

\[
m_s=\operatorname{RelaxedTopK}(s+g,T)
\]

组合：

\[
m=
m_s+\operatorname{sg}(m_h-m_s)
\]

这是 stochastic STE。

## 优点

- 自带 exploration；
- 易于实现；
- fixed-K 很自然；
- 可以缓解 deterministic Top-K early lock-in。

## 缺点

- gradient variance；
- temperature tuning；
- train/inference distribution mismatch；
- 超长序列上 soft relaxation 成本仍要控制。

推荐作为很重要的 baseline。

---

# 16. 方法族 F：SIMPLE / ProbMoE —— exact-K discrete subset + exact marginal

这是理论上最符合“indexer 是 subset policy”这一观点的一族。

---

# 17. SIMPLE

定义离散变量：

\[
z\in\{0,1\}^{L}
\]

满足：

\[
\sum_i z_i=K
\]

从条件分布中采样：

\[
z\sim p_\theta(z\mid \sum z_i=K)
\]

forward 真正运行：

\[
\boxed{\text{discrete exact-K subset}}
\]

然后计算每个 item 的 marginal：

\[
\mu_i=P_\theta(z_i=1)
\]

构造 straight-through：

\[
z_{\rm ST}
=
\mu+\operatorname{sg}(z-\mu)
\]

### Forward

\[
z_{\rm ST}=z
\]

### Backward

\[
\frac{\partial z_{\rm ST}}{\partial\theta}
=
\frac{\partial\mu}{\partial\theta}
\]

这比 generic soft relaxation 更 principled，因为 backward 使用的是 subset distribution 的 exact marginal。

---

# 18. ProbMoE

ProbMoE 将上述思路直接搬进 MoE routing。

核心定义：

\[
p_\theta(S),\qquad |S|=K
\]

而不是每个 item 独立 gate：

\[
g_i\in[0,1]
\]

这是非常重要的建模区别。

对于 sparse attention indexer：

\[
S_q\sim p_\theta(S_q)
\]

\[
S_q\subseteq\{1,\dots,L\}
\]

\[
|S_q|=K
\]

然后真实 attention 只执行：

\[
S_q
\]

中的 KV。

---

## 18.1 为什么这比 independent gates 更本质

如果所有 score 相等：

\[
s_i=c
\]

那么只是：

\[
p(S)=\frac1{\binom LK}
\]

即：

\[
\boxed{\text{uniform uncertainty over K-subsets}}
\]

而不是退化回 dense attention。

只要不同 token 的真实 task utility 不同，LM loss 就会破坏这个对称性。

这是 exact-K subset formulation 最大的理论优势之一。

---

## 18.2 对 sparse attention 的主要难点

MoE：

\[
N_{\rm expert}\sim 8\text{-}256
\]

Sparse attention：

\[
L\sim 32K\text{-}1M
\]

exact marginal 动态规划可能不可直接扩展。

可考虑：

1. candidate Top-M；
2. chunk-level subset；
3. hierarchical subset；
4. approximate marginals；
5. low-rank / block factorization；
6. local exact + global proposal；
7. sampled candidates。

---

# 19. 方法族 G：直接去掉 Top-K —— continuous sparse routing

代表思路：

- ReMoE
- Sparsemax / Sparsegen
- DirMoE 的独立 stochastic selection
- Soft MoE / SMEAR / Lory（更激进地不做 discrete token routing）

---

# 20. ReMoE：ReLU Routing

直接：

\[
g_i=\operatorname{ReLU}(s_i)
\]

所以：

\[
s_i\le0\Rightarrow g_i=0
\]

自然 sparse。

通过 sparsity regularization / constrained budget 控制：

\[
\sum_i\mathbf1[g_i>0]
\]

或 expected compute。

优点：

- 不再有 Top-K；
- 正区间正常可导；
- dynamic K。

缺点：

- 不是 exact-K；
- 对固定 KV budget 的 sparse attention 不一定适合。

---

# 21. DirMoE：selection 与 contribution 分离

DirMoE 的重要思想不是具体分布，而是：

\[
\boxed{\text{selection 和 contribution 是两个不同随机变量}}
\]

例如：

\[
z_i\sim\operatorname{Bernoulli}
\]

决定选不选；

\[
w
\]

决定 selected expert 的 mixing weight。

对 sparse attention 的启发：

- indexer score：只决定 subset；
- 原 attention softmax：决定 selected KV contribution。

这支持前文推荐的：

\[
\boxed{\text{不要把 indexer score 再作为 log-gate 加进 attention logits}}
\]

---

# 22. 不应误认为“Top-K 可导”的方法

以下方法改善 routing，但**没有真正解决 Top-K selection gradient**：

## 22.1 Noisy Top-K

\[
s'=s+\epsilon
\]

再：

\[
\operatorname{TopK}(s')
\]

作用主要是：

- exploration；
- load balancing；
- 避免 deterministic early lock-in。

但：

\[
\frac{\partial\operatorname{TopK}}{\partial s}
\]

仍然为零。

---

## 22.2 Expert Choice Routing

只是改变：

\[
\text{token chooses experts}
\]

为：

\[
\text{expert chooses tokens}
\]

优化负载平衡和 capacity，但 Top-K 仍是 discrete。

---

## 22.3 BASE / balanced assignment

把 routing 变成 batch-level constrained assignment。

解决：

- expert capacity；
- load balance；
- token dropping。

不是直接的 Top-K differentiability 方法。

如果未来 sparse attention 做：

\[
\text{global KV budget allocation across queries}
\]

这类 structured assignment 才更相关。

---

# 23. Sparse attention 特有问题：MoE 方法不能直接照搬

MoE 和 sparse attention 有关键差别。

## 23.1 Candidate 数量

MoE：

\[
N=8\sim256
\]

Attention：

\[
L=32K\sim1M
\]

因此：

- DenseMixer 全算 expert 在 MoE 可接受；
- attention 不可能 full dense candidate backward。

必须用：

\[
\boxed{\text{candidate restriction / hierarchy}}
\]

---

## 23.2 每个 candidate 的“branch output”不同

MoE expert：

\[
E_i(x)
\]

是一个明确 branch。

Attention 中 token \(j\) 的作用与其他 selected token 联合决定：

\[
o(S)
=
\frac{
\sum_{j\in S}
e^{a_j}v_j
}{
\sum_{j\in S}e^{a_j}
}
\]

所以 token utility 不是独立 additive：

\[
u_j(S)
\]

依赖当前集合 \(S\)。

因此：

- pairwise swap；
- boundary perturbation；
- local subset evaluation；

比简单 per-token regression 更自然。

---

## 23.3 Top-K boundary 最重要

真正会改变 selection 的主要是：

\[
s_{(K-r)},\dots,s_{(K+r)}
\]

因此可以重点给 boundary gradient。

定义：

\[
B=
\{j:
|rank(j)-K|\le r
\}
\]

然后只在 \(B\) 做 expensive surrogate / counterfactual。

这可能是 sparse attention 相比 MoE 更重要的优化。

---

# 24. 一个关键理论视角：Indexer 本质是 learning-to-rank

最终 inference 只看：

\[
\operatorname{TopK}(s)
\]

所以：

\[
s
\]

和：

\[
as+b,\qquad a>0
\]

selection 完全相同。

因此 indexer 不需要 score calibration。

真正有意义的是：

\[
s_i-s_j
\]

尤其 boundary pair：

\[
i\in S,\quad j\notin S
\]

所以可把问题写成：

\[
\boxed{\text{learning-to-rank under compute budget}}
\]

而不是：

\[
\text{gate regression}
\]

这会自然避免很多 constant-gate / scale-degeneracy 问题。

---

# 25. 推荐实现路线：按优先级排序

## Priority 1：Candidate STE

最容易实现、最应该先跑。

### Forward

```python
scores = indexer(q, k_idx)
candidate_idx = topk(scores, M)
hard_idx = topk(scores[candidate_idx], K)

out_hard = sparse_attention(
    q,
    k[candidate_idx[hard_idx]],
    v[candidate_idx[hard_idx]],
)
```

### Backward surrogate

```python
soft_mask = differentiable_topk(
    scores[candidate_idx],
    K,
)

out_soft = weighted_candidate_attention(
    q,
    k[candidate_idx],
    v[candidate_idx],
    soft_mask,
)

out = out_soft + (out_hard - out_soft).detach()
```

目标：

```python
loss = lm_loss(model_output)
```

不再额外做 attention distillation。

---

# 26. Candidate STE 推荐细节

## Candidate size

开始：

\[
M=2K
\]

再试：

\[
4K
\]

## Candidate construction

```text
80% indexer Top-M
10% random
5% recent/local
5% stratified/chunk candidate
```

## Temperature schedule

soft Top-K relaxation 若有温度 \(T\)：

开始高一些：

\[
T_0\approx1
\]

逐渐：

\[
T\downarrow0.1
\]

不要过快变硬，否则 gradient 很快消失。

---

# 27. Priority 2：Candidate-DenseMixer / Pairwise Swap

只对 boundary 做 counterfactual。

取：

\[
B_{\rm in}
=
\text{bottom-r selected}
\]

\[
B_{\rm out}
=
\text{top-r unselected}
\]

例如：

\[
r=8\sim64
\]

随机采：

\[
(i,j)\in B_{\rm in}\times B_{\rm out}
\]

计算 swap loss：

\[
\Delta_{ij}
=
L(S-i+j)-L(S)
\]

pairwise target：

\[
y_{ij}
=
\mathbf1[\Delta_{ij}<0]
\]

若 \(j\) 更好：

\[
L_{\rm pair}
=
w_{ij}
\operatorname{softplus}(s_i-s_j)
\]

若 \(i\) 更好：

\[
L_{\rm pair}
=
w_{ij}
\operatorname{softplus}(s_j-s_i)
\]

---

# 28. Priority 3：Sander sparse differentiable Top-K / Fast LapSum

用于替换简单 sigmoid-threshold surrogate。

目标接口：

```python
soft_mask = differentiable_topk(
    scores,
    k=K,
    method="sparse_topk"  # or lapsum
)
```

期望性质：

```text
0 <= soft_mask <= 1
sum(soft_mask) ~= K
gradient exists around ranking boundary
prefer sparse support
```

推荐：

- 小规模先实现 Sander-style；
- 超长序列尝试 Fast LapSum；
- 先在 candidate Top-M 内验证，再考虑全序列。

---

# 29. Priority 4：SIMPLE / ProbMoE exact-K subset

更研究型、理论最干净。

目标：

```python
subset, marginals = sample_exact_k_subset(
    logits=scores,
    k=K,
)
```

构造：

```python
mask = marginals + (subset - marginals).detach()
```

Forward：

```text
mask == sampled exact-K subset
```

Backward：

```text
gradient flows through exact/approx marginals
```

Sparse attention：

```python
out = sparse_attention(
    q,
    k[subset],
    v[subset],
)
```

主要工程问题：

- large-L exact marginals；
- memory；
- dynamic programming cost。

推荐先在：

\[
M=2K\sim4K
\]

candidate pool 内实现 exact-K subset。

---

# 30. Priority 5：SparseMixer / GRIN-like custom routing gradient

如果前面的 continuous relaxation 有明显 bias，可以做。

推荐 workflow：

1. 先构造 true boundary swap oracle；
2. 测量现有 STE gradient 与 true \(\Delta L\) 的相关性；
3. 设计新的 routing-gradient estimator；
4. 比较：
   - cosine similarity；
   - rank correlation；
   - sign accuracy；
   - downstream LM loss；
   - retrieval quality。

核心 scientific question：

\[
\boxed{\text{我们的 gradient estimator 是否真的恢复了 discrete selection 的 marginal utility？}}
\]

---

# 31. 可加入的 LM-gradient utility self-distillation

如果不想每次做 swap，可以利用 local gradient。

假设给每个 token 一个 infinitesimal additive bias：

\[
a_j\leftarrow a_j+b_j
\]

attention：

\[
\alpha_j=
\operatorname{softmax}(a+b)_j
\]

输出：

\[
o=\sum_j\alpha_jv_j
\]

有：

\[
\frac{\partial o}{\partial b_j}
=
\alpha_j(v_j-o)
\]

所以：

\[
\frac{\partial L}{\partial b_j}
=
\alpha_j
\left(
\frac{\partial L}{\partial o}
\right)^\top
(v_j-o)
\]

定义一阶 token utility：

\[
\hat u_j
=
-
\frac{\partial L}{\partial b_j}
\]

然后训练 indexer ranking：

\[
s_i>s_j
\quad\text{if}\quad
\hat u_i>\hat u_j
\]

这不是 attention-weight distillation，而是：

\[
\boxed{\text{LM-gradient self-distillation}}
\]

它可作为辅助 loss，与主 LM loss 一起使用。

---

# 32. 推荐总 loss

第一版：

\[
L=L_{\rm LM}
\]

第二版：

\[
L=
L_{\rm LM}
+
\lambda_{\rm rank}L_{\rm pair}
\]

第三版：

\[
L=
L_{\rm LM}
+
\lambda_{\rm util}L_{\rm utility-rank}
+
\lambda_{\rm budget}L_{\rm budget}
\]

如果 exact-K 已结构性保证：

\[
L_{\rm budget}
\]

不需要。

---

# 33. 不推荐默认采用的 anti-collapse trick

以下方法可以做 baseline，但不应作为核心解决方案：

## 33.1 Softmax normalize gate

\[
g=\operatorname{softmax}(s)
\]

只能解决共同 scale / shift 的 identifiability。

不能从根本上解决：

\[
s_1=\dots=s_L
\]

导致 uniform / no-op 的问题。

---

## 33.2 固定 anchor token

例如强制一个 sink token gate 为 1。

本质是 gauge fixing。

不是 selection-learning 的核心机制。

---

## 33.3 Gate entropy regularization

例如：

\[
L_{\rm entropy}
=
H(g)
\]

可以防 uniform 或过尖，但依赖 hyperparameter，且和最终 LM utility 不完全一致。

只建议做稳定训练辅助项。

---

## 33.4 独立 \(L_0\) penalty

\[
L=
L_{\rm LM}
+\lambda\sum_jP(g_j\ne0)
\]

问题：

- \(\lambda\) 难调；
- 容易全开 / 全关；
- 不是 exact-K。

如果 inference 固定 K，更推荐直接把：

\[
|S|=K
\]

编码进变量空间。

---

# 34. Exact-K 是最自然的 constraint

对于 inference 固定 KV budget：

\[
|S|=K
\]

训练最好也保持 exact-K，而不是通过 penalty 间接逼近。

推荐变量空间：

\[
m\in\{0,1\}^L
\]

subject to：

\[
\mathbf1^\top m=K
\]

或者：

\[
S\in\binom{[L]}{K}
\]

这样：

- 不存在 all-on；
- 不存在 all-off；
- constant score 只表示 uniform uncertainty；
- selection objective 与 inference 一致。

---

# 35. 推荐实验矩阵

至少比较：

| 方法 | Forward exact K | Unselected gradient | Extra compute | Train-test mismatch | 推荐 |
|---|---:|---:|---:|---:|---:|
| attention distill | ✓ | teacher 提供 | 中 | 有 objective mismatch | baseline |
| log gate | ✓/否 | ✗ | 低 | 中 | baseline |
| Candidate STE | ✓ | candidate 内 ✓ | 低-中 | 低 | ★★★★★ |
| ST-Gumbel Top-K | ✓ | candidate 内 ✓ | 中 | 中 | ★★★★ |
| Sander sparse Top-K | soft/sparse | ✓ | 中 | 中 | ★★★★★ |
| LapSum | soft budget | ✓ | 中 | 中 | ★★★★★ |
| Fast LapSum | soft budget | ✓ | 低-中 | 中 | ★★★★★ |
| Candidate DenseMixer | ✓ | candidate 内 ✓ | 中 | 低 | ★★★★★ |
| Pairwise Swap | ✓ | sampled pairs | 高 | 无 | ★★★★★ oracle |
| SIMPLE / ProbMoE | **✓ exact-K** | ✓ marginals | 中-高 | 极低 | ★★★★★ |
| SparseMixer / GRIN | ✓ | ✓ estimator | 低-中 | 极低 | ★★★★★ |
| ReMoE / Sparsegen | ✗ dynamic K | ✓ | 低 | inference需适配 | ★★★ |

---

# 36. 关键评测：不要只看最终 accuracy

为了理解“Top-K 可微方案是否真的学对”，建议额外测：

## 36.1 Gradient vs true swap oracle

采 boundary pair：

\[
(i,j)
\]

真实：

\[
\Delta L_{ij}
\]

估计：

\[
\hat g_{ij}
=
\hat g_j-\hat g_i
\]

测：

- Pearson correlation；
- Spearman correlation；
- sign accuracy；
- Top-r pair ranking accuracy。

---

## 36.2 Oracle recovery

定义 full-attention / expensive counterfactual oracle subset：

\[
S^*
\]

测：

- Recall@K；
- Precision@K；
- NDCG；
- boundary swap regret。

---

## 36.3 LM loss regret

定义：

\[
R=
L(S_{\rm indexer})-L(S_{\rm oracle})
\]

这是比 attention-weight KL 更接近真实目标的指标。

---

## 36.4 Selection stability

测相邻 training step：

\[
J(S_t,S_{t+1})
\]

防止 stochastic / soft method selection 抖动过大。

---

## 36.5 Candidate miss rate

如果使用 Top-M candidate：

\[
\operatorname{MissRate}
=
\frac{
|S^*\setminus C|
}{
|S^*|
}
\]

这个指标非常重要。

如果 oracle token 根本不在 candidate pool：

\[
\text{任何 backward estimator 都救不了}
\]

---

# 37. 推荐 ablation

必须至少包含：

1. attention distill；
2. LM loss + log gate；
3. LM loss + Candidate STE；
4. LM loss + ST-Gumbel；
5. LM loss + sparse differentiable Top-K；
6. LM loss + pairwise swap auxiliary；
7. LM loss + exact-K subset；
8. frozen backbone vs jointly train backbone；
9. candidate size \(M/K=1,2,4,8\)；
10. random exploration ratio；
11. soft temperature；
12. exact-K vs dynamic-K；
13. per-token vs per-chunk selection；
14. boundary-only vs full-candidate backward。

---

# 38. 注意 routing absorption / backbone co-adaptation

如果 backbone 与 indexer 一起训练，可能出现：

\[
\boxed{\text{backbone 学会适应一个差 router，而不是 router 学会更好选择}}
\]

因此第一阶段实验建议：

```python
for p in backbone.parameters():
    p.requires_grad = False
```

但不要：

```python
with torch.no_grad():
    backbone(...)
```

因为梯度仍然必须经过 frozen backbone operations 回到 indexer。

正确：

```text
LM loss
   ↓
frozen backbone computational graph
   ↓
sparse attention selection surrogate
   ↓
indexer parameters
```

---

# 39. 推荐工程实现抽象

建议统一成如下接口。

```python
class TopKSelector(nn.Module):
    def forward(
        self,
        scores,          # [..., L]
        k: int,
        training: bool,
        candidate_idx=None,
    ):
        # Returns:
        #   hard_idx
        #   hard_mask
        #   surrogate_mask
        #   aux
        pass
```

不同方法只替换 selector。

---

# 40. Selector 1：Hard Top-K

```python
hard_idx = torch.topk(scores, k).indices
```

仅用于：

- inference；
- baseline。

---

# 41. Selector 2：Sigmoid threshold STE

```python
threshold = kth_largest(scores, k).detach()
soft = torch.sigmoid((scores - threshold) / temperature)

hard = topk_mask(scores, k)

mask = soft + (hard - soft).detach()
```

注意：

- soft sum 不一定等于 K；
- threshold gradient 是否 detach 要单独 ablate；
- 最好只在 candidate pool 内做。

---

# 42. Selector 3：ST-Gumbel Top-K

```python
noise = sample_gumbel(scores.shape)
perturbed = scores + noise

hard = topk_mask(perturbed, k)
soft = relaxed_topk(perturbed, k, temperature)

mask = soft + (hard - soft).detach()
```

---

# 43. Selector 4：Sparse differentiable Top-K

```python
soft_sparse = sparse_topk_operator(scores, k, reg_strength)
hard = topk_mask(scores, k)

mask = soft_sparse + (hard - soft_sparse).detach()
```

可以测试：

- pure soft forward；
- hard-forward ST；
- sparse support forward。

---

# 44. Selector 5：LapSum

```python
soft = lapsum_topk(scores, k)
```

理想性质：

```python
soft.min() >= 0
soft.max() <= 1
soft.sum(-1) == k
```

再选：

```python
hard = topk_mask(scores, k)
mask = soft + (hard - soft).detach()
```

---

# 45. Selector 6：Exact-K subset

```python
subset, marginals = exact_k_sample(scores, k)

mask = marginals + (subset - marginals).detach()
```

如果 full-L 太贵：

```python
candidate_idx = topk(scores, M).indices
subset_local, marginal_local = exact_k_sample(
    scores[candidate_idx],
    k
)
```

---

# 46. Attention integration 推荐方式

不要：

```python
attn_logits += log(indexer_gate)
```

推荐：

```python
scores = indexer(...)

hard_idx, hard_mask, surrogate_mask, aux = selector(
    scores, k
)

out_hard = sparse_attention(q, k_cache, v_cache, hard_idx)
```

如果用 output STE：

```python
out_soft = surrogate_attention(
    q,
    k_cache,
    v_cache,
    candidate_idx,
    surrogate_mask,
)

out = out_soft + (out_hard - out_soft).detach()
```

---

# 47. Surrogate attention 如何定义

一种简单方式：

\[
w_j=
m_j^{soft}
\exp(a_j)
\]

\[
\alpha_j=
\frac{w_j}{
\sum_i w_i
}
\]

即：

\[
\alpha
=
\operatorname{Normalize}
(
m^{soft}\odot e^a
)
\]

输出：

\[
o_{\rm soft}
=
\sum_j\alpha_jv_j
\]

注意：

- \(m^{soft}\) 只在 candidate 内；
- hard forward 不使用 \(m^{soft}\)；
- 避免 indexer score直接改变 hard attention semantics。

---

# 48. 另一种 surrogate：soft subset mixture

对若干 sampled subsets：

\[
S_1,\dots,S_R
\]

计算：

\[
o_r=\operatorname{Attn}(S_r)
\]

再：

\[
o_{\rm soft}
=
\sum_r p_\theta(S_r)o_r
\]

这个更接近 expected discrete policy，但计算成本高。

可以仅用于：

- oracle；
- 小模型；
- 少量 batch；
- gradient estimator validation。

---

# 49. 如果做 chunk sparse attention

如果每个 chunk 为一个候选：

\[
L_c=L/B
\]

则 exact-K / differentiable Top-K 会容易很多。

例如：

\[
L=128K,\quad B=64
\]

则：

\[
L_c=2048
\]

这是 SIMPLE / ProbMoE / OT / sorting 方法更现实的规模。

因此一个很重要的路线：

\[
\boxed{\text{先对 chunk 做可微 exact-K，chunk 内再正常 attention}}
\]

通常比 token-level full sequence 更容易实现。

---

# 50. Hierarchical selection

可以：

第一层：

\[
C=
\operatorname{TopKChunk}(s^{chunk})
\]

第二层：

\[
T_c=
\operatorname{TopKToken}(s^{token}|c)
\]

总 budget：

\[
\sum_c |T_c|=K
\]

两层分别可以用：

- differentiable Top-K；
- exact-K subset；
- STE。

这也是 scalable exact-K 的重要方向。

---

# 51. 推荐的第一轮实现组合

如果只有时间实现 4 个：

## A. Candidate STE

用于证明：

\[
\text{LM loss alone 能不能训练 indexer}
\]

## B. ST-Gumbel Candidate Top-K

用于测试 exploration 是否关键。

## C. Sparse differentiable Top-K / Fast LapSum

用于测试 continuous relaxation 是否比 simple STE 稳。

## D. Pairwise Swap Oracle

不是主训练方法，而是评价其它 gradient estimator 是否正确。

---

# 52. 推荐的第二轮研究组合

如果第一轮有效：

## E. SIMPLE / ProbMoE candidate exact-K

最 principled。

## F. SparseMixer / GRIN-like custom gradient

最接近真正 discrete routing gradient。

## G. LM-gradient utility self-distillation

成本比真实 swap 更低。

---

# 53. 建议论文/方法阅读清单

以下是需要优先查阅的方法名，agent 实现时应搜索原论文和官方代码，而不要只依赖本说明中的二手描述：

### MoE / routing gradient

- SparseMixer / *Sparse Backpropagation for MoE Training*
- GRIN / SparseMixer-v2
- Dense Backpropagation for Sparsely-Gated MoE
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
- DenseMixer
- ProbMoE
- DSelect-k
- ReMoE
- DirMoE
- LD-MoLE
- COMET
- SoftMoE (2026, LapSum-based)
- Soft MoE (2024, slot-based; different paper)
- SMEAR
- Lory
- BASE Layers
- Expert Choice Routing
- Maximum Score Routing

### Generic differentiable Top-K / subset selection

- SIMPLE: A Gradient Estimator for k-Subset Sampling
- Reparameterizable Subset Sampling via Continuous Relaxations
- Gumbel-TopK / RelaxedTopK
- Differentiable Top-k Operator with Optimal Transport
- Fast, Differentiable and Sparse Top-k: a Convex Analysis Perspective
- NeuralSort
- SoftSort
- LapSum
- Fast LapSum
- Sparsemax
- Entmax
- Sparsegen

---

# 54. 最终推荐的研究 framing

不要把问题表述成：

> 如何给 Top-K 一个可导近似？

更好的 framing：

\[
\boxed{
\text{How to optimize a discrete compute-allocation policy under an exact attention budget?}
}
\]

其中：

- indexer = policy / ranker；
- selected KV subset = action；
- sparse attention compute = budgeted computation；
- LM loss = task reward / objective；
- Top-K differentiability = credit assignment。

因此几条方法对应：

### STE

\[
\text{biased low-variance gradient estimator}
\]

### Gumbel

\[
\text{stochastic reparameterized relaxation}
\]

### SparseMixer / GRIN

\[
\text{routing-gradient estimator}
\]

### DenseMixer / Counterfactual

\[
\text{counterfactual branch evaluation}
\]

### SIMPLE / ProbMoE

\[
\text{exact-cardinality stochastic policy + marginal gradient}
\]

### Sander / LapSum

\[
\text{continuous differentiable surrogate for the constrained selection operator}
\]

---

# 55. 最重要的实现原则总结

1. **把 indexer 定义成 ranking / subset selector，不是 attention gate。**
2. **尽量结构性保证 exact-K，而不是靠 penalty 调 sparsity。**
3. **hard forward 最能减少 train-test mismatch。**
4. **backward 的核心是让 Top-K 外尤其 boundary token 获得 credit。**
5. **candidate pool 必须有 exploration，否则未入池 token 永远学不到。**
6. **长序列不能照搬 MoE 全专家 dense-backward，必须 candidate / chunk / hierarchy。**
7. **优先验证 gradient estimator 是否与真实 swap \(\Delta L\) 一致。**
8. **frozen backbone 是验证 indexer credit assignment 的最佳第一阶段设置。**
9. **不要把 attention-weight distillation 当作唯一 teacher；LM-loss marginal utility 更接近真实目标。**
10. **最终最干净的 formulation 是：exact-K subset + LM loss。**

---

# 56. 给实现 agent 的直接任务建议

建议按以下顺序开始代码：

```text
Step 1
实现统一 TopKSelector 接口。

Step 2
实现 hard Top-K baseline。

Step 3
实现 Candidate STE：
hard forward + sigmoid/soft-topk backward。

Step 4
实现 ST-Gumbel Candidate Top-K。

Step 5
实现一个 sparse differentiable Top-K operator
优先参考 Sander / LapSum / Fast LapSum。

Step 6
实现 boundary pairwise swap oracle，
用于评估 gradient estimator。

Step 7
在 candidate pool 内实现 SIMPLE / ProbMoE-style exact-K sampling + marginals。

Step 8
比较：
LM loss、
Top-K recall、
swap regret、
gradient-sign accuracy、
训练稳定性、
额外 FLOPs / memory。

Step 9
如果 STE 与 swap oracle 相关性差，
再实现 SparseMixer / GRIN-style routing-gradient estimator。

Step 10
扩展到 chunk / hierarchical Top-K，降低长序列成本。
```

---

# 57. 推荐默认配置

第一版可以从：

```yaml
selector: candidate_ste
candidate_ratio: 4        # M = 4K
candidate_top_fraction: 0.85
candidate_random_fraction: 0.10
candidate_structural_fraction: 0.05

hard_k: K

soft_topk:
  method: sigmoid_threshold
  temperature_init: 1.0
  temperature_final: 0.1
  temperature_decay: cosine

forward:
  hard_sparse_attention: true

backward:
  use_output_ste: true

backbone:
  frozen: true
  keep_computation_graph: true

loss:
  lm_loss: 1.0
  pairwise_swap: 0.0     # 后续开启
  utility_rank: 0.0      # 后续开启
```

---

# 58. 最后一句

最值得坚持的原则是：

\[
\boxed{
\text{不要想办法“让 gate 看起来不退化”，而要直接优化真正的 Top-K compute allocation。}
}
\]

如果最终 sparse attention 推理阶段执行的是：

\[
S=\operatorname{TopK}(s),\qquad |S|=K
\]

那么训练阶段最自然的数学对象也应当是：

\[
\boxed{\text{ranking / exact-K subset}}
\]

而不是一组独立、绝对值有额外自由度的 soft gates。
