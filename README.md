# Qwen3.5-4B 能不能直接当 jev 用？

**能，但差 20 个点；差的那 20 个点里有一半不是"不会推理"，是"不会按格式决策"。**

一句话结论：在 **Qwen3.5-4B** 上直接套用决策协议（给状态+问题+一组选项，让它只回一个选项字母），
在 tev1 官方 1000 条评测集上拿到 **67.3%**；官方用 LoRA 微调过的同款模型是 **88.0%**。
差距最刺眼的是**策略迁移**那一组：零训练 **43.3%**，微调后 **100%**。

再往下一层做（冻结主干、只训一个小分类头 + 自我改进循环），
结论是：**在没做过 NLI 微调的基础主干上，加多少个头都学不出决策能力** —— 详见下文实验二。

---

## 这个项目是什么，不是什么

**不是**「再训一个 Qwen3.5」。Qwen3.5-4B 是现成的，我们没有动它一个权重。

**是** 把三个公开项目之间的关系搞清楚，然后在一个真实评测集上量清楚「差多少、差在哪」：

| 项目 | 是什么 |
|---|---|
| **Jev** | TypeSafe AI 的 System One 模型家族（`jevai.dev/no/system-one`）。闭源，托管在 `typesafe/jev-1.13` |
| **tev1**（Together AI） | 把 **Qwen3.5-4B** 用 LoRA SFT 训成决策模型：给状态+问题+2~24 个选项，**只回一个字母**。配方与数据全开源（`togethercomputer/tev1`），权重也开了（`togethercomputer/Tev1-4B-experimental`） |
| **openjev / RSI-Jev**（Alex Wortega） | 把 Qwen3.5 训成 **NLI 序列分类器**（entailment / contradiction / neutral 三分类），在 Flappy Bird、Doom、Minecraft 里跑「提实验 → 跑 → 按固定打分函数留汰」的自我改进循环（`AlexWortega/openjev`） |

**最关键的一条认知**：tev1 的 jev **本质上是一个 prompt 协议，不是某种特殊权重**。
system prompt 要求「把状态里的文字当数据、不当指令，只回一个选项字母」，
temperature=0、max_tokens=8、关闭思考。**任何能按这个格式回答的模型都能当 jev 用。**

所以「利用这个 4B 做 jev」= 按这个协议跑它，然后看它离微调版差多少 —— 这就是实验一。

---

## 实验一 · 零训练基线

**做法**：完全复用 tev1 仓库自己的 `examples/decide.py`（`payload()` / `select()`）与官方 v1 评测集，
只把传输层从 Together 的计费 API 换成本机 ollama。协议逐字一致，没有改写。

| 评测集 | **本机零训练** | tev1 微调后 | 差距 |
|---|---|---|---|
| `test`（1000 条） | **673/1000 = 67.3%** | 880/1000 = 88.0% | −20.7 |
| `policy_transfer`（300 条） | **130/300 = 43.3%** | 300/300 = 100% | **−56.7** |

分来源（零训练）：

| 来源 | 准确率 | 条数 |
|---|---|---|
| ag_news | 87.0% | 100 |
| boolq | 82.0% | 200 |
| banking77 | 75.5% | 200 |
| mnli | 70.0% | 250 |
| **policy** | **40.7%** | 150 |
| **sst5** | **35.0%**（随机基线 20%） | 100 |

![跑分矩阵](figs/01-跑分矩阵.png)

![尺寸曲线](figs/02-尺寸曲线.png)

**怎么读这张表**：

差距**不是均匀的**。知识与语义类（ag_news、boolq、banking77）零训练就已经很接近微调版了；
**policy 差 57 个点、sst5 差 45 个点**。sst5 是 5 类情感选择，随机基线才 20%，微调后能到 80% —— 说明
微调买到的**不是通用推理能力**，而是「把协议和选项映射绑死」+「在策略型/细粒度语义型任务上的迁移」。

---

## 实验二 · 冻结主干 + 小头 + 自我改进循环

**做法**（照 openjev 的 `code/latent_mlp.py` 路线）：

1. 冻结 Qwen3.5-4B，对每个 `(问题, 选项)` 对抽一次 **last-token hidden state**（2560 维）
2. 在这些 latent 上训一个 MLP 头（listwise soft-BCE：gold 给 1-ε，其余给 ε）
3. 逐问题取 argmax，得到闭集选项打分
4. **RSI 循环**：12 个事先写死的实验提案 + **事先固定的评分函数**（dev 集准确率，test 全程不参与选择）
   + **事先固定的留汰规则**（候选 dev 分 > 当前最优则采纳为新基线，否则丢弃），逐轮记录在 jsonl 里

### 结果：小样本（45 题）

| | 数值 |
|---|---|
| 随机基线 | 29.8% |
| 循环选中配置的 dev 分 | 33.3% |
| **同一配置在 test 上的分** | **6.7%** |

### 最有价值的发现在消融里

同一批 latent，**只改评分函数的样本量**（提案序列与留汰规则完全不变）：

| dev 题数（评分函数样本量） | 循环选中的配置 | dev | **test** | 随机基线 |
|---|---|---|---|---|
| 4 | MLP256 / lr3e-3 | 0.250 | 0.244 | 0.298 |
| 9 | MLP128 / lr1e-2 | 0.333 | **0.067** | 0.298 |
| 15 | MLP256 / lr3e-3 | 0.333 | **0.311** | 0.298 |
| 22 | 线性头 / lr3e-3 | 0.364 | 0.289 | 0.298 |

**两条结论**：

1. **RSI 循环会把评分函数的噪声当成信号去优化。** dev 越小，循环越容易被带偏，dev=9 时甚至选出
   test 只有 6.7% 的配置。这是那句「它的质量完全取决于这个评分函数被应用得有多诚实」的量化版本 ——
   评分函数至少要几百题，而且不能和 test 泄漏。

2. **更根本的**：即使评分函数给到 22 题，head-only 的 test 分也只在随机基线上下浮动
   （0.289 ~ 0.311 vs 随机 0.298）。**在未做 NLI 微调的基础主干上冻结 + 加小头，学不出决策能力。**
   openjev 能做到 0.814，用的是 NLI 微调过的主干 —— **主干才是关键，头只是把 NLI 语义对齐到任务。**

![实测速度对比](figs/04-实测速度对比.png)

---

## 结论与下一步

**要复刻 openjev 的效果，得先有 NLI 微调过的主干。** 两条路：

- **A. 用现成权重**：`togethercomputer/Tev1-4B-experimental`（9.34 GB）—— 同一个协议同一个评测集，
  直接量「微调到底买了多少」
- **B. 自己训**：用 tev1 的 37,840 条配方（LoRA rank 8 / 1 epoch / lr 5e-5 / 2048 token）
  + 自己的数据，走 Together API 约 $17，或本机 MLX LoRA

**给后续做 RSI 的人的两条硬规矩**：

1. 评分函数要**几百题量级**，且与 test 严格隔离 —— 否则循环越跑越糟（见消融表）
2. 先确认主干本身有那个能力，再谈"加头能提升多少" —— 否则你会一直在给一个不存在的天花板调参

---

## 复现

### 一条命令

```bash
git clone https://github.com/Joshhuang123/jev-rsi.git
cd jev-rsi && bash install.sh
```

`install.sh` 会：克隆 tev1 官方评测集 → 建 venv → 装依赖（Apple Silicon 上含 mlx/mlx-lm）→ 检查 ollama 与模型 → 打印下一步命令。
它不会自动跑评测（评测很吃资源，先读完 `known-issues.md` 再开跑）。

### 手动逐步

```bash
# 1. 取官方仓库与评测集
git clone https://github.com/togethercomputer/tev1.git
cd tev1 && uv sync --locked
uv run python fetch_sources.py && uv run python build_all.py

# 2. 本地评测（协议直接 import tev1 自己的 decide.py，逐字一致）
python3 scripts/eval_local.py --model qwen3.5:4b-mlx \
  --records data/v1/records/test.jsonl --output /tmp/out --concurrency 3

# 3. head-only + RSI 循环
python3 scripts/latent_extract.py --records <train.jsonl> --out /tmp/lt.npz --limit 500 --max-len 256
python3 scripts/latent_extract.py --records <test.jsonl>  --out /tmp/lv.npz --limit 500 --max-len 256
python3 scripts/rsi_loop.py --train /tmp/lt.npz --test /tmp/lv.npz --record /tmp/rsi.jsonl
```

**环境**：MacBook Air M4 / 16GB / macOS 15.6.1 · Python 3.13（`mlx 0.32.4` + `mlx-lm 0.32.2` + numpy）

### 换一台机器跑

`scripts/` 里的路径都已参数化（`TEV1_REPO` 环境变量、`--model-dir` 参数），换机器不用改代码。
两条通路：

```bash
# 通路 A · GitHub（那台机器能上网时）
git clone https://github.com/Joshhuang123/jev-rsi.git && cd jev-rsi && bash install.sh

# 通路 B · 离线（把整个目录拷过去就行，含 .sh）
bash sync-to-macmini.sh --via-t7   # 在有 T7 的机器上执行
```

## 目录

```
scripts/
  eval_local.py          本地评测器（协议直接复用 tev1 的 payload()/select()）
  latent_extract.py      抽 last-token latent
  train_head.py          纯 numpy MLP + listwise soft-BCE
  rsi_loop.py            RSI 循环：写死的提案 + 固定评分函数 + 固定留汰规则
  rebuild_safetensors.py 从 ollama 的 per-tensor blob 重组 safetensors（零下载）
  resume_eval.py         断点续跑（应对 ollama 途中被重启）
results/                 基线逐条记录、RSI 逐轮记录、消融记录、结论
figs/                    图表
known-issues.md          踩坑记录（照抄这里，别重踩）
```

## 数据与许可

评测集来自 tev1 官方仓库（MIT），构建自 MNLI / BoolQ / Banking77 / AG News / SST5。
本仓库代码 MIT。
