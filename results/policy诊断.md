# 诊断 · 零训练模型在 policy 上错在哪

> 2026-11-02。基于本机实测的 1000 条逐条结果（`baseline-test-1000.jsonl`），
> 与官方 `data/v1/records/test.jsonl` 对齐后逐条分析。

## 一句话结论

**零训练模型「会算布尔逻辑，但不会识别信息缺失」。**

policy 组 150 题，模型正确 61 题（40.7%）。拆开看：

| 题型 | 题数 | 模型正确 | 正确率 |
|---|---:|---:|---:|
| **二值题**（真实是 eligible / ineligible） | 62 | 43 | **69%** |
| **三值题**（真实是 unknown） | 50 | 18 | **36%** |

**二值逻辑能做对七成，三值的 unknown 基本是瞎猜。**

---

## 混淆矩阵（行=正确，列=模型预测）

|  | eligible | ineligible | unknown | 合计 |
|---|---:|---:|---:|---:|
| **eligible** | **25** | 14 | 11 | 50 |
| **ineligible** | 5 | **18** | 27 | 50 |
| **unknown** | 13 | 19 | **18** | 50 |

**读这张表**：

1. **对角线 25 / 18 / 18 = 61**（模型的 61 题全在这三格）。
2. **`ineligible → unknown` 有 27 题** —— 最大的单一错误模式。
   模型知道自己算不出 eligible，于是退到 unknown。**但很多题明明能算出 ineligible。**
3. **`unknown → ineligible` 19 题** —— 模型在「事实缺失」时**没有选 unknown，而是硬判了一个值**。
4. **模型选 unknown 共 56 次，真实只有 50 次** —— 过度倾向 unknown（但精度只有 32%）。

---

## 根因：模型没有检查「这个事实给了没有」

抽 3 条模型误判的 unknown 题：

```
policy:  Eligible if and only if (days_since_purchase <= 233 AND item_unopened is true)
         OR order_value_usd >= 1810.
         Unspecified facts are unknown; do not assume they are false.
record:  { days_since_purchase: 229, order_value_usd: 1809 }
                                   ↑ item_unopened 根本没给
正确答案: unknown   |   模型选了: ineligible
```

```
policy:  Eligible if and only if (days_since_purchase <= 63 AND NOT (item_unopened is true)).
record:  { days_since_purchase: 58, order_value_usd: 11787 }
                                   ↑ item_unopened 根本没给
正确答案: unknown   |   模型选了: ineligible
```

**三条全是同一个错误**：`item_unopened` 这个事实在 `record` 里缺失，
按「未指定的事实是未知的」应当选 unknown，模型却代入了一个隐含假设去判 ineligible。

**这不是逻辑能力问题，是「先做事实核查、再做逻辑求值」这个流程缺失。**
模型直接跳到求值，把缺失的事实当成了 false。

---

## 这决定了「值不值得训」

| 如果微调买到的是… | 对本问题的效果 |
|---|---|
| 更好的布尔推理 | **收益有限** —— 69% 已经不算差 |
| **「事实缺失 → 选 unknown」这个行为** | **收益巨大** —— 50 题从 36% 到 100%，policy 组直接 40.7% → ~93% |

**上游微调版在 policy 上是 100%。** 也就是说，
**这 59.3 个点的差距，几乎全部可以用「学会在事实缺失时说不知道」来解释。**

而这恰恰是 SFT 最擅长的事 —— 它不需要模型变得更聪明，
只需要把「遇到缺失事实就选 unknown」这个行为模式灌进去。

---

## 由此得出的行动建议

**方案 C 的诊断做完了，结论是：值得训（方案 A），而且训练目标很明确。**

训练数据里要保证：
1. **unknown 的样本比例足够高** —— 训练集 policy 1500 条里，
   真实答案 unknown 的约 1/3，模型得先学会这个先验；
2. **不能只学「unknown 是安全选项」** —— 零训练模型已经过度倾向 unknown 了，
   得同时压住 `ineligible → unknown` 那 27 题。

**在此之前，还有一件零成本的事值得做**：
既然诊断发现「模型没做事实核查」，那**在 prompt 里显式加一步**可能就能拿回一部分 ——
比如要求它先列出规则需要哪些事实、record 给了哪些、缺哪个，再求值。
**这是纯 prompt 改动，不用训任何模型就能测。**

如果这样就能把 policy 从 40.7% 拉到 70%+，
那结论会变成「**这个任务不需要 SFT，需要更好的协议**」——
这比「训一版 Tev1」便宜两个数量级。

→ 下一步就跑这个 prompt 消融（方案 C2），再决定要不要动训练。

---

## 复现

```bash
cd ~/WorkBuddy/jev-rsi-publish
python3 - <<'PY'
import json, pathlib
from collections import Counter
recs=[json.loads(l) for l in pathlib.Path('results/baseline-test-1000.jsonl').read_text().splitlines() if l.strip()]
te={json.loads(l)['id']: json.loads(l) for l in pathlib.Path('/Users/junhaidemacbookair/WorkBuddy/tev1/data/v1/records/test.jsonl').read_text().splitlines() if l.strip()}
pol=[r for r in recs if r['id'] in te and te[r['id']].get('source')=='policy']
cm=Counter()
for r in pol:
    t=te[r['id']]
    gold=[o['key'] for o in t['options'] if o['label']==t['answer']][0]
    cm[(gold, r.get('predicted_key'))]+=1
for g in ('eligible','ineligible','unknown'):
    print(g, [cm[(g,p)] for p in ('eligible','ineligible','unknown')])
PY
```
