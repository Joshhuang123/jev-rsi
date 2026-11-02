# 踩坑记录

按「症状 → 原因 → 改法」记。都是在这台机器上真实撞过的。

---

## 1. 官方 registry 慢到不能用，但 HF 镜像很快

**症状**：`ollama pull qwen3.5:4b` 跑了 30 分钟只下了 400MB；同一时刻
`mirrors.aliyun.com` 跑到 23.7 MB/s。

**原因**：ollama registry 在境外，跨境链路被压到 ~120 KB/s。**不是本地网络问题** ——
同一个 Wi-Fi 下国内源是满速。

**改法**：走 `hf-mirror.com`（`https://hf-mirror.com/<repo>/resolve/main/<file>`），
实测 4.73 MB/s，比官方快 39 倍。

**顺带**：`hf-mirror.com` 自己也会 **429 限流**（连着下几个大文件时），
断了重试即可，别以为是网络挂了。

---

## 2. 国内源必须加进 NO_PROXY，否则被本机代理打断

**症状**：tev1 的 `fetch_sources.py` 挂在 `httpx.ConnectError: [SSL: UNEXPECTED_EOF_WHILE_READING]`，
同样的代码在没挂代理的机器上是好的。

**原因**：本机有 `HTTP_PROXY/HTTPS_PROXY=http://127.0.0.1:7897`（系统代理也开着）。
**国内源不需要走代理**，被塞进代理反而被 TLS 打断。

**改法**：把国内源加进 `NO_PROXY`：

```bash
export NO_PROXY="localhost,127.0.0.1,::1,hf-mirror.com,mirrors.aliyun.com,pypi.tuna.tsinghua.edu.cn,www.modelscope.cn"
export no_proxy="$NO_PROXY"
```

判断依据：清华 pypi 走代理 10.8s、直连 2.7s。

---

## 3. HF 模型卡的跑分表分「Non-Thinking」和「Thinking」两段

**症状**：抓同一个基准（MMLU-Pro）的分，0.8B 卡给 29.7、2B 卡给 42.3，看着像数据打架。

**原因**：**不是冲突，是两种推理模式。** 每张模型卡的表里，同一基准名会出现两次：
前一段是关闭思考的分数，后一段是开思考的分数。

**改法**：抓表时按分段行切分，别把两次出现当成同一行。

---

## 4. ollama 的 `/api/embed` 只服务 embedding 类模型

**症状**：`POST /api/embed` 对 `qwen3.5:4b-mlx` 返回
`{"error":"This server does not support embeddings. Start it with --embeddings"}`。

**原因**：纯 causal LM 没有 pooling 头，ollama 不给它算 embedding。
**`--embeddings` 这个 flag 在 0.40 已经不存在了**（`ollama serve --help` 里没有），
报错信息是历史遗留。

**改法**：想拿 hidden state 就得绕开 ollama，走 MLX / transformers。

---

## 5. ollama 的 MLX 模型是「每张量一个 blob」，但可以零下载重组

**症状**：想把 ollama 里那个 730 层的 MLX 模型喂给 mlx-lm，没有整文件可用。

**原因**：ollama 0.40 导入 safetensors 时，把每个张量（或一组）拆成独立 blob 存。
但**每个 blob 本身仍是合法的 safetensors 文件**（自带 header + 数据）。

**改法**（`scripts/rebuild_safetensors.py`）：读 manifest 拿到所有 tensor 层，
逐个读出 header 里的 `(name, dtype, shape, data_offsets)`，重排成一个新的 safetensors。
**省掉 3GB 下载。**

**改法二（同样有效但要下权重）**：`mlx-community/Qwen3.5-4B-4bit` 走 ModelScope，
实测 2.88 MB/s（`https://www.modelscope.cn/api/v1/models/<repo>/repo?Revision=master&FilePath=<file>`）。

---

## 6. 重组后键名要改两处

**症状**：重组完喂给 mlx-lm，报
`ValueError: Received 498 parameters not in model`（498 = 924 个张量 − 426 个期望）。

**原因**：两套命名对不上 ——

| ollama 里的名字 | mlx-lm 期望的 |
|---|---|
| `language_model.model.X` | `model.X`（去掉 VL 包装层） |
| `model.X.weight.scale` | `model.X.scales`（mlx 量化层用复数） |
| `model.X.weight.bias` | `model.X.biases` |

视觉塔（`vision_tower.*`，297 个张量）直接丢掉 —— mlx-lm 的 VL 类本来就跳过它，纯文本任务用不上。

**另外**：mlx-lm 的 `Model`（VL 包装类）会**再补一层** `language_model.` 前缀，
所以纯文本任务要**直接用 `TextModel`**，并按 config 顶层的 `quantization` 调 `nn.quantize`。

---

## 7. `safe_open(framework="mlx")` 读不了 bfloat16

**症状**：`TypeError: data type 'bfloat16' not understood`。

**原因**：safetensors 的 mlx 后端遇到 bf16 会退回 numpy，而 numpy 没有 bf16。

**改法**：用 mlx 自己的加载器 `mx.load(path)`。

---

## 8. macOS 15 上，非交互 CLI 进程被定位服务挡住

**症状**：在 agent 里跑 `networksetup -setairportnetwork` 报 `-3900 tmpErr`；
`networksetup -getairportnetwork` 报 `You are not associated with an AirPort network`；
系统日志里是：

```
APP NOT AUTHORIZED FOR LOCATION SERVICES, will not continue (status=0 ... req=GET SSID)
```

**原因**：macOS 14+ 起，连 Wi-Fi 需要定位权限，**非交互 CLI 进程默认没有**。

**注意**：这会让 CLI 侧的报错**看起来像模型答错，其实是权限门**。排查时先确认是不是这条。

**改法**：让命令跑在 Terminal.app 里（`osascript` 派发），Terminal 有权限。
同理 `security find-generic-password` 读 Wi-Fi 密码也会被挡 —— 而且这些条目在
**数据保护钥匙串**里，CLI 根本读不到。

---

## 9. 同一台机器上另一个 agent 在并行跑，会把环境搅乱

**症状（连续踩了三次）**：

- 评测跑到一半进程没了 → 查日志发现 **ollama 被重启**（PID 从 55252 变成 2108）
- `/private/tmp` 里 4.4G 莫名清空，包括发布脚本要用的 Chrome 登录态目录 `chrome-ops`
- `ollama list` 里的模型少了三个

**原因**：这台机器上 WorkBuddy AI 是并行跑的 agent，会重启 ollama、清临时目录、动模型。

**改法**：
- 评测脚本加重试（连接错误不算模型答错）
- 写 `scripts/resume_eval.py` 断点续跑
- 长任务**串行**跑，别和别的模型推理抢 GPU
- 干活前先 `df -h` 和 `top` 看一眼内存/磁盘，异常就先把该卸的模型卸了
  （`ollama stop <model>` 能立刻腾出几 GB）

---

## 10. 16GB 机器上跑模型抽取 latent 会被 OOM KILL

**症状**：抽到一半进程被杀（`Killed: 9`），内存 15G used / 446M unused / swap 13GB。

**原因**：ollama 里加载的模型 + MLX 进程里又加载一份 + 别的 agent 进程，一起把 unified memory 吃穿。

**改法**：
- **串行**：评测和抽取不要同时跑
- 抽取前 `ollama stop` 把用不到的模型卸掉
- latent **流式写盘**（`np.lib.format.open_memmap`），别全攒在内存里
- 磁盘也别忘了，抽取前先 `df -h`

---

## 11. RSI 循环会被评分函数的噪声带偏

不是代码坑，是**方法论坑**，但值得写下来。

dev 集只有 9 题时，循环选中的配置在 test 上只有 6.7%（比随机还低）；
dev 给到 15 题时是 31.1%。**循环会忠实地把评分函数的噪声当成信号去优化。**

**评分函数至少要几百题量级**，且必须与 test 严格隔离。
一个"看起来很严谨"的自我改进循环，评分函数样本量不够的话，跑得越久越糟。
