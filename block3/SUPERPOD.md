# 在 HKUST SuperPOD 上跑真实数据

## 布局

```
/scratch/mscbdtspod2026/cyeah/5018/
├── code/          # rsync 自本地仓库(不含 .git / outputs)
├── hf/            # HF_HOME,JEV-9B 权重 ~18GB
└── outputs/preds-shared-shots/ # jev.jsonl / qwen.jsonl / run_summary.json
```

## 环境(已创建)

`conda env jevbench`(Python 3.11),版本与 `deployment/README.md` 的实测环境对齐:

```
torch 2.11.0 · transformers 5.16.1 · peft 0.21.0 · bitsandbytes 0.50.1 · accelerate 1.13.0
```

注意:不要复用 `retool`(transformers 4.53 不支持 `dtype=` 传参)或 `m5017`(缺 peft/bitsandbytes)。

## 提交流程

```bash
ssh cyeah@superpod.ust.hk
source /etc/profile.d/modules.sh; module load slurm

# 1) 冒烟(每类 20 条,约 5 分钟)
cd /scratch/mscbdtspod2026/cyeah/5018/code
sbatch block3/slurm_infer.slurm --limit-per-type 20

# 2) 看日志
tail -f /scratch/mscbdtspod2026/cyeah/5018/outputs/infer_<jobid>.out
squeue -u $USER

# 3) 全量(2000 条 × 2 模型)
sbatch block3/slurm_infer.slurm
```

`--gpus=1`、`--time=01:00:00`;H800 80GB 对 9B(4-bit NF4)绰绰有余,单卡即可。

## T9 正式协议:held_out 拟合

PLAN §2.3 要求「用 held_out 集拟合、eval 集报告」。先给 held_out 跑一次预测(只需一次):

```bash
sbatch block3/slurm_infer.slurm \
    --input "$PWD/evaluation/data/held_out.jsonl" \
    --out-dir "$PWD/../outputs/fit"
```

再在登录节点出报告(纯 numpy/matplotlib,不占 GPU):

```bash
python -m block3 calibrate \
    --pred-dir ../outputs/preds-shared-shots \
    --fit-dir ../outputs/fit \
    --out-dir ../outputs/calibration-heldout \
    --folds 5 --seed 0 --bootstrap 1000
```

报告 JSON 的 `protocol` 会写成 `held_out_fit`;不带 `--fit-dir` 时是交叉拟合,作为稳健性检查保留。

## T10 选项探针

在集群仓库根目录生成选项探针，再给 Slurm 脚本传入专用输入和输出目录。两模型使用脚本指定的同一份 train few-shot 示例。

```bash
python -m block3 probes --source-type noul --limit 2 --seed 0 \
    --probes option_renaming option_order_permutation abstain_removal \
    --out-dir block3/outputs/choice-probes
sbatch block3/slurm_infer.slurm \
    --input "$PWD/block3/outputs/choice-probes/all_pairs.jsonl" \
    --out-dir "$PWD/../outputs/choice-t10"
```

查看 `choice-t10/run_summary.json` 中各模型的 `option_probes`。需要增加样本量时，通过生成器的 `--limit` 按源样本抽取，保留完整配对。

2026-11-13 的正式一轮用 `--limit 200`、四类探针全开（作业 `778430`，两模型各 1600 条、
0 失败、约 4.5 min），结果在 `../outputs/probes-t10/`，数字汇总见 [README](README.md#真实运行结果superpod2026-11-13)。
`prompt_injection` 的翻转率需从 `jev.jsonl` / `qwen.jsonl` 按 `pair_id` 自行统计，`run_summary.json` 只自动汇总三类选项探针。

## 断点续跑

`cluster_run` 会跳过输出文件里已有的 `id`,每条立即落盘:

* 作业超时 → 重新 `sbatch` 同一个脚本即可续跑;
* 只想补一个模型 → `--models qwen`;
* 冒烟产物和全量产物在同一个 `jev.jsonl` 里,全量时会自动跳过前 20 条。

只在输入、选项和 few-shot 方案一致时复用目录。主评测脚本默认使用 `preds-shared-shots/`，历史的仅 Qwen few-shot 结果在 `preds/`。

## 回本地做校准

```bash
scp -r cyeah@superpod.ust.hk:/scratch/mscbdtspod2026/cyeah/5018/outputs/preds-shared-shots ./block3/outputs/
python -m block3 calibrate --pred-dir block3/outputs/preds-shared-shots --out-dir block3/outputs/shared-calibration
```

或者直接在集群上跑(校准只需要 numpy/matplotlib,`jevbench` 环境已装):

```bash
python -m block3 calibrate --pred-dir ../outputs/preds-shared-shots --out-dir ../outputs/shared-calibration
```

## 实测（2026-11-12）

以下是仅 Qwen 使用 few-shot 的历史结果。

| 作业 | 内容 | 结果 |
| --- | --- | --- |
| `777055` | 冒烟，每类 20 条 × 2 模型 | COMPLETED 1m17s |
| `777059` | 全量 2000 条 × 2 模型 | COMPLETED 10m20s，0 失败 |

JEV 4.6 min / 5.97 rec/s，Qwen 5.4 min / 6.07 rec/s；单条 p50 延迟 137 ms / 172 ms。
校准是纯 numpy/matplotlib，**直接在登录节点跑**即可，不占 GPU 分钟：

```bash
ssh cyeah@superpod.ust.hk
source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh
conda activate jevbench && export PYTHONNOUSERSITE=1
cd /scratch/mscbdtspod2026/cyeah/5018/code
python -m block3 calibrate --pred-dir ../outputs/preds --out-dir ../outputs/calibration --folds 5 --seed 0
```

## 已知坑

| 现象 | 处理 |
| --- | --- |
| `sbatch: command not found` / 打开帮助 | 非交互 shell 先 `source /etc/profile.d/modules.sh; module load slurm` |
| `conda activate` 失败 | 先 `source /cm/shared/apps/Anaconda3/2023.09-0/etc/profile.d/conda.sh` |
| 诡异的 torch/nvidia 版本错乱 | 每个作业 `export PYTHONNOUSERSITE=1` |
| 作业被 `slurmstepd` 杀掉 | GPU 分钟累计超预算;先用 `sacct` 查余额,再决定 `--time` |
| 登录节点 `torch.cuda.is_available()` 为 False | 正常,登录节点没有 GPU |
| 模型下载慢/中断 | 在登录节点先跑一次 `resolve_snapshot`,job 里就是纯本地加载 |
| `json.decoder.JSONDecodeError: Unterminated string` | 用 `str.splitlines()` 读 JSONL 会在 ` `/` ` 处断开;改为逐行迭代文件句柄(已修,`12d1302`) |
| 长时间 ssh 会话中途 `Connection closed by remote host` | 用 `setsid nohup` 起远端监视器写标记文件,本地后台轮询,单次 ssh 失败不影响整体 |
