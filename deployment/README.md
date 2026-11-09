# 模型部署

本项目比较 JEV-9B 和 Qwen3.5-9B 在垃圾邮件、钓鱼邮件和毒性检测上的表现。一个本地进程提供两个推理入口，共用 AutoTrust 在 JEV-9B 发布包中提供的 Qwen3.5-9B 文本骨干，因此只下载一份骨干权重。JEV 路径启用 LoRA，使用原生决策头和官方温度。Qwen 路径禁用 LoRA，读取允许选项的原始 token logits，并在这些选项内计算 softmax 概率。[AutoTrust 模型卡](https://huggingface.co/autotrust/JEV-9B#plain-transformers--peft)

## 安装与启动

在包含 `requirements.txt` 的目录中运行与你的操作系统对应的终端命令。项目统一使用 Python 3.12 的 `.venv`。`requirements.txt` 提供基础开发与测试依赖，`requirements-cuda.txt` 在此基础上安装 GPU 推理依赖。

GPU 部署需要支持 BF16 和 CUDA 12.8 的 NVIDIA GPU 及驱动，并预留至少 30 GB 磁盘空间。首次启动下载约 18 GB，默认保存在 `.cache/huggingface/`。可以用 `--cache-dir` 指定其他位置，服务和验收脚本须使用同一路径。

### Windows（PowerShell）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m deployment.server
```

如果用 Conda 提供 Python，请先激活 Python 3.12 的环境，并将创建虚拟环境的命令中的 `py -3.12` 换成 `python`。

### Linux x86-64（Bash）

使用带 NVIDIA 驱动的 Linux 主机。Python 3.12 需包含 `venv` 模块。[PyTorch CUDA 安装说明](https://pytorch.org/get-started/previous-versions/#v2110) · [bitsandbytes 安装说明](https://huggingface.co/docs/bitsandbytes/v0.50.1/en/installation)

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.11.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-cuda.txt
.venv/bin/python -m pip check
.venv/bin/python -m deployment.server
```

服务监听 `127.0.0.1:8000`，可用 `--port` 改端口。按 Ctrl+C 停止。权重固定到 `7ac669b8fb476ba926491e3d1c39f9330d68dbed`，使用 NF4、二次量化和 BF16 计算。JEV 决策头保留 FP32。

### macOS 或无 NVIDIA GPU 的环境

基础环境可运行接口和概率计算测试。需要在 Windows 或 Linux 的 NVIDIA 主机上运行真实推理和验收。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -s deployment/tests -v
```

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest discover -s deployment/tests -v
```

## 接口

在运行服务的主机上另开终端调用两个模型。组装 few-shot 示例时，将示例和任务问题放进同一个 `prompt` 字段，并把待判定文本原样放进 `text`。

### PowerShell

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{
    type = 'noul'
    text = 'Congratulations! You won a $500 gift card. Click this link to claim it now.'
    prompt = 'Is this message spam?'
} | ConvertTo-Json
foreach ($model in @('jev', 'qwen')) {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/$model/predict" `
        -ContentType 'application/json; charset=utf-8' -Body $body | ConvertTo-Json -Depth 5
}
```

### Bash 或 zsh

```bash
curl -fsS http://127.0.0.1:8000/health
printf '\n'
body='{"type":"noul","text":"Congratulations! You won a $500 gift card. Click this link to claim it now.","prompt":"Is this message spam?"}'
for model in jev qwen; do
    curl -fsS -H 'Content-Type: application/json' --data "$body" \
        "http://127.0.0.1:8000/v1/$model/predict"
    printf '\n'
done
```

| 字段 | 含义 |
| --- | --- |
| `decision` | `noul` 为 `no` 或 `yes`，`score` 为概率最大的整数 `0`–`5`。平局取顺序最靠前的选项。 |
| `prob` | 按标签索引的完整概率分布。二分类的正类概率取 `prob["yes"]`，评估置信度时取预测标签对应的概率。 |
| `logits` | 按相同标签索引的原始分数。JEV 的值尚未除以官方温度，Qwen 的值是原始选项 token logits。评分任务的 JSON 对象键是字符串。 |
| `latency` | 服务内耗时（毫秒），包含 prompt 渲染、分词和 GPU 推理。计时从取得推理锁之后开始。 |

两个模型收到相同的 token 序列，二分类内部使用 `false/true`，响应映射为 `no/yes`。完整渲染后的输入上限默认是 4,096 tokens，超限返回 HTTP 413。无效请求返回 422，GPU 显存不足返回 503。遇到显存不足时，先释放 GPU 上的其他工作，再用同一 `--max-input-tokens` 限制重测两个模型。`/health` 返回环境信息和 PyTorch 统计的峰值显存。

## 验收

接口和概率计算测试可以用已安装依赖的虚拟环境运行。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s deployment/tests -v
```

```bash
.venv/bin/python -B -m unittest discover -s deployment/tests -v
```

真实验收需要重启服务，在对应的启动命令末尾加上 `--audit-inputs`。在另一个终端运行下列命令。脚本预热两种任务，校验概率与 logits 的关系、两路输入 token 的哈希及长度边界，输出四个真实样例结果和显存数据。

```powershell
New-Item -ItemType Directory -Force .cache | Out-Null
.\.venv\Scripts\python.exe -B -m deployment.smoke | Tee-Object .cache/smoke-results.json
```

```bash
mkdir -p .cache
set -o pipefail
.venv/bin/python -B -m deployment.smoke | tee .cache/smoke-results.json
```

### 实测结果

2026-11-09 在 Windows、RTX 5070 Ti Laptop GPU（12 GB）、驱动 616.92 和 Python 3.12.13 上通过。PyTorch 2.11.0 + CUDA 12.8、Transformers 5.16.1、PEFT 0.21.0、bitsandbytes 0.50.1。两条路径接受 4,096 tokens，超过上限都返回 HTTP 413。此次启动及验收期间，PyTorch 峰值分配 **10,288 MiB**，峰值保留 **11,586 MiB**。

| 预热后单次请求 | JEV-9B | Qwen3.5-9B |
| --- | --- | --- |
| 垃圾邮件（49 tokens） | `yes`，P(yes) 0.7083，245 ms | `yes`，P(yes) 0.6514，268 ms |
| 毒性评分（62 tokens） | `3`，P(3) 0.3503，254 ms | `0`，P(0) 0.2462，261 ms |
| 长度边界（4,096 tokens） | 通过，6.17 s | 通过，3.91 s |

演示可以使用接口示例中的垃圾邮件。两个模型都返回 `yes`，同时给出完整概率和原始 logits。表中是部署验收的单次观测值。
