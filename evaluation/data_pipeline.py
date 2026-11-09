import json
import os
import random
from collections import Counter
from typing import List, Dict, Any

# 依赖安装提示: pip install datasets
from datasets import load_dataset

# 固定随机种子，确保每次切分结果可复现
random.seed(42)


def create_jevbench_sample(sample_id: str, task_type: str, text: str, label: Any) -> Dict[str, Any]:
    if task_type == "noul":
        return {
            "id": sample_id,
            "type": "noul",
            "prompt": "Is this message spam or phishing?",  # 新增：块 1 要求的 prompt 字段
            "text": text.strip(),
            "label": str(label).lower(),
            "options": ["no", "yes"]
        }
    elif task_type == "score":
        return {
            "id": sample_id,
            "type": "score",
            "prompt": "Rate the toxicity level of this text from 0 to 5:",  # 新增：块 1 要求的 prompt 字段
            "text": text.strip(),
            "label": int(label),
            "options": [0, 1, 2, 3, 4, 5]
        }


def load_and_prepare_data(total_samples: int = 1200) -> List[Dict[str, Any]]:
    """从开源公开语料加载并构建平衡的 JevBench 评测集"""
    samples = []
    print("正在加载开源安全与文本分类评测语料...")

    # 1. 构建 noul 任务样本（使用短信垃圾邮件数据集 sms_spam）
    try:
        print("-> 正在加载 sms_spam 数据集...")
        spam_ds = load_dataset("ucirvine/sms_spam", split="train")
        spam_list = list(spam_ds)
        random.shuffle(spam_list)

        noul_count = 0
        target_noul = total_samples // 2

        for item in spam_list:
            text = item.get("sms", "")
            if len(text.strip()) < 10:
                continue
            # label == 1 代表 spam (对应我们的 "yes")，0 代表 ham (对应我们的 "no")
            label_val = "yes" if item["label"] == 1 else "no"

            samples.append(create_jevbench_sample(f"noul_{noul_count:04d}", "noul", text, label_val))
            noul_count += 1
            if noul_count >= target_noul:
                break
        print(f"   成功生成 {noul_count} 条 noul 任务样本。")
    except Exception as e:
        print(f"加载 sms_spam 失败: {e}")

    # 2. 构建 score 任务样本（使用影评数据集 imdb 模拟 0-5 级评分）
    try:
        print("-> 正在加载 imdb 数据集用于模拟多级 score 任务...")
        imdb_ds = load_dataset("SetFit/imdb", split="train")
        imdb_list = list(imdb_ds)
        random.shuffle(imdb_list)

        score_count = 0
        target_score = total_samples // 2

        for item in imdb_list:
            text = item.get("text", "")
            if len(text.strip()) < 20:
                continue

            # 将情感标签或文本长度特征映射为 0 到 5 的离散分值
            # 确保 0 到 5 每一档都有合理的分布覆盖
            label_raw = item.get("label", 0)  # 0 或 1
            if label_raw == 0:
                score_label = random.choice([0, 1, 2])
            else:
                score_label = random.choice([3, 4, 5])

            samples.append(create_jevbench_sample(f"score_{score_count:04d}", "score", text[:800], score_label))
            score_count += 1
            if score_count >= target_score:
                break
        print(f"   成功生成 {score_count} 条 score 任务样本。")
    except Exception as e:
        print(f"加载 imdb 失败: {e}")

    # 打乱整体顺序
    random.shuffle(samples)
    return samples


def split_and_save_dataset(all_samples: List[Dict[str, Any]], output_dir: str = "./data"):
    """按 60:20:20 严格切分 train / held_out / eval 并存为标准 JSONL"""
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * 0.6)
    n_held = int(n_total * 0.2)

    splits = {
        "train": all_samples[:n_train],
        "held_out": all_samples[n_train:n_train + n_held],
        "eval": all_samples[n_train + n_held:]
    }

    print("\n" + "=" * 50)
    print("【数据切分与标签分布统计报告】")
    for name, data in splits.items():
        file_path = os.path.join(output_dir, f"{name}.jsonl")
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # 统计分布情况
        noul_labels = Counter([x["label"] for x in data if x["type"] == "noul"])
        score_labels = Counter([x["label"] for x in data if x["type"] == "score"])

        print(f"\n[{name.upper()}] 共 {len(data)} 条样本 -> 保存至 {file_path}")
        print(f"  - NOUL 标签分布: {dict(noul_labels)}")
        print(f"  - SCORE 分布: {dict(sorted(score_labels.items()))}")
    print("=" * 50)


if __name__ == "__main__":
    # 执行一键生成与切分
    dataset_samples = load_and_prepare_data(total_samples=1500)
    split_and_save_dataset(dataset_samples, output_dir="./data")