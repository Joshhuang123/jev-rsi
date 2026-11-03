# Prompt 消融 · 不训练也能拿回 policy 那一组

> 2026-11-02。基于 `results/policy诊断.md` 的诊断结论做的下一步。
> **只改 system prompt，其余逐字复用 tev1 官方协议**（temperature=0 / max_tokens=8 /
> logprobs=true / top_logprobs=5 / 关闭思考 / regex 约束）。

---

## 一句话结论

**policy 组从 40.7% 拉到 69.3%，只靠改 system prompt，零训练。**

诊断说「模型不会识别信息缺失」，改 prompt 让它先核查事实 → **unknown 题从 36% 涨到 84%**。

---

## 三个变体（policy 150 题）

| 变体 | system prompt 要点 | policy 准确率 | 二值题 | unknown 题 |
|---|---|---:|---:|---:|
| **base**（官方协议） | 原文：「选一个字母」 | 40.7% | 43% | 36% |
| **factcheck** | 加「先查事实，缺了就选'无法确定'」 | 48.0% | 32% ↓ | **80%** ↑ |
| **balanced** | 加「先查事实」**＋**「事实齐全时必须给出真实答案，不得选'无法确定'」 | **69.3%** | **62%** ↑ | **84%** ↑ |

**+28.6 个点，零训练。**

### 混淆矩阵对比

**base（40.7%）** — 真实 unknown 只有 18/50 对，模型大量误选 ineligible：

|  | eligible | ineligible | unknown |
|---|---:|---:|---:|
| eligible | **25** | 14 | 11 |
| ineligible | 5 | **18** | 27 |
| unknown | 13 | 19 | **18** |

**balanced（69.3%）** — 两头都涨：

|  | eligible | ineligible | unknown |
|---|---:|---:|---:|
| eligible | **36** | 7 | 7 |
| ineligible | 2 | **26** | 22 |
| unknown | 3 | 5 | **42** |

---

## 最重要的发现：factcheck 单用会「矫枉过正」

只看 factcheck 的结果，很容易得出错误结论：

| | base | factcheck | balanced |
|---|---:|---:|---:|
| unknown 题 | 36% | **80%** | **84%** |
| 二值题（eligible/ineligible） | **43%** | 32% ↓ | 62% ↑ |
| 净结果 | 40.7% | 48.0% | 69.3% |

**factcheck 单独用，unknown 题暴涨 44 个点，但二值题掉了 11 个点** ——
模型学会了「拿不准就选 unknown」，代价是把本来能算对的题也推给了 unknown。
最大的单一错误模式 `ineligible → unknown` 从 27 条涨到 35 条。

**balanced 加的那句反向约束**：
> "If every required fact is present, you MUST evaluate the rule and give the true answer;
> do not answer 'cannot be determined' in that case."

把二值题从 32% 拉回 62%（还高于 base 的 43%），同时 unknown 保持 84%。

**这解释了为什么 SFT 能做到 100%**：SFT 不只是教模型「要查事实」，
它用数据分布同时教会了「什么时候该说不知道、什么时候必须给出答案」。
**纯 prompt 改写只能逼近这个行为，做不到完美平衡 —— 因为 prompt 没法携带正确的不确定率。**

---

## 对「差距 20.7 个点」这个结论的修正

原说法：零训练 67.3% vs 上游微调 88.0%，差 20.7 个点。

**加上这次消融后，这个说法要再修正一层**：

| | policy (150) | 其余五源 (850) | 合计 test |
|---|---:|---:|---:|
| 零训练 · 官方协议 | 40.7% | 75.9% | **67.3%** |
| 零训练 · balanced prompt | **69.3%** | 待测 | 待测 |
| 上游微调 | 100% | 87.0% | **88.0%** |

**「差 20.7 个点」这个数字里，有一大块不是「模型不够好」，而是「协议没告诉它该怎么做」。**

---

## 对下一步决策的影响

原来的三条路：

| 方案 | 判断 |
|---|---|
| A. 自己训一版 Tev1 4B | **仍然值得做**，但目标变了 —— 不是「教会它决策」，是「教会它平衡 unknown 的使用率」 |
| B. 换更强底座 | 值得一试 |
| **C2. 改 prompt** | **✅ 已做完，收益 +28.6 点，零成本** |

**新的优先级**：
1. **先测 balanced prompt 在全部 1000 题 + policy_transfer 上的表现** —— 确认这个增益是局部的（只在 policy）还是可迁移的。
2. 如果 balanced 在别处也有增益 → **基线要重测**，之前所有基于 67.3% 的结论都要更新。
3. 然后再决定要不要训 —— 训的目标会更清楚：**补上 balanced prompt 补不了的那部分**。

---

## 踩坑（值得记）

**用 ollama 的 OpenAI 兼容层 `/v1/chat/completions` 关不掉思考**：
`chat_template_kwargs.enable_thinking=false` 这个字段它不认，
必须转成顶层 `think: false` **且走原生 `/api/chat`**。
不转的话模型输出 "Thinking Process:"、`content` 全空、`finish_reason=length`，
**答案全错（0/150）但 HTTP 200、不报错** —— 看起来像「模型很笨」，实际是传输层没配对。

`scripts/eval_local.py` 里已经踩过这个坑（它转了 `think` 并用原生接口），
新写脚本时直接照抄，别重新踩。

---

## 复现

```bash
cd ~/WorkBuddy/jev-rsi
python3 eval_prompt_variant.py \
  --model qwen3.5:4b-mlx \
  --records ~/WorkBuddy/tev1/data/v1/records/test.jsonl \
  --output /tmp/bal_policy.jsonl --concurrency 3 --variant balanced
```
