import json
import os
import random
from collections import Counter
from typing import List, Dict, Any
from datasets import load_dataset

# 固定随机种子，确保每次切分结果可复现
random.seed(42)

DEFAULT_TOTAL_SAMPLES = 10000


def create_jevbench_sample(sample_id: str, task_type: str, text: str, label: Any, source: str = "unknown") -> Dict[str, Any]:
    if task_type == "noul":
        return {
            "id": sample_id,
            "type": "noul",
            "source": source,
            "prompt": "Is this message spam or phishing?",
            "text": text.strip(),
            "label": str(label).lower(),
            "options": ["no", "yes"]
        }
    elif task_type == "score":
        return {
            "id": sample_id,
            "type": "score",
            "source": source,
            "prompt": "Count the applicable toxicity categories (0 to 5): severe toxic, obscene, threat, insult, identity hate.",
            "text": text.strip(),
            "label": int(label),
            "options": [0, 1, 2, 3, 4, 5]
        }
def load_and_prepare_data(total_samples: int = DEFAULT_TOTAL_SAMPLES) -> List[Dict[str, Any]]:
    """Load three spam/phishing sources and Jigsaw toxicity category-count labels."""
    samples = []
    print("正在加载 SMS spam、Enron spam、phishing 与 Jigsaw Toxicity 语料...")

    # Split the requested total evenly between NOUL and SCORE, then distribute
    # any NOUL remainder across the three sources so the requested total is exact.
    noul_target = total_samples // 2
    target_score = total_samples - noul_target
    base_quota, extra_quota = divmod(noul_target, 3)
    sources = [
        ("ucirvine/sms_spam", "train", "sms_spam"),
        ("SetFit/enron_spam", "train", "enron_spam"),
        ("drorrabin/phishing_emails-data", "train", "phishing_email"),
    ]
    for source_index, (dataset_name, split_name, source_name) in enumerate(sources):
        noul_per_source = base_quota + int(source_index < extra_quota)
        if noul_per_source == 0:
            continue
        added = 0
        try:
            print(f"-> 正在加载 {source_name} ({dataset_name})...")
            try:
                dataset = load_dataset(dataset_name, split=split_name)
            except Exception as primary_error:
                if source_name != "phishing_email":
                    raise
                print(f"   主 phishing 语料加载失败，尝试备用集 jhonrayo99/phishing-email-balanced-6000: {primary_error}")
                try:
                    dataset = load_dataset("jhonrayo99/phishing-email-balanced-6000", split="train")
                    source_name = "phishing_email_fallback"
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise
            candidates = list(dataset)
            random.shuffle(candidates)
            for item in candidates:
                if source_name == "sms_spam":
                    text = str(item.get("sms", "")).strip()
                    raw_label = item.get("label", 0)
                    is_positive = int(raw_label) == 1
                elif source_name == "enron_spam":
                    text = str(item.get("text", "")).strip()
                    raw_label = item.get("label", 0)
                    is_positive = int(raw_label) == 1
                else:
                    text = str(item.get("text", "")).strip()
                    raw_label = str(item.get("email_type", item.get("label", ""))).lower()
                    is_positive = raw_label in {"phishing email", "phishing", "1"}
                if len(text) < 10:
                    continue
                label = "yes" if is_positive else "no"
                sample = create_jevbench_sample(f"noul_{source_name}_{added:05d}", "noul", text[:4000], label, source_name)
                samples.append(sample)
                added += 1
                if added >= noul_per_source:
                    break
            print(f"   {source_name}: 成功生成 {added} 条 NOUL 样本。")
            if added < noul_per_source:
                print(f"   警告：{source_name} 可用样本少于目标数量 {noul_per_source}。")
        except Exception as e:
            print(f"加载 {source_name} 失败: {e}")

    if not any(sample.get("source", "").startswith("phishing_email") for sample in samples):
        raise RuntimeError("未能加载任何 phishing 邮件样本；停止生成，避免产出缺少 phishing 覆盖的数据集。")
    print("-> 正在从 Hugging Face 加载 Arsive/toxicity_classification_jigsaw (train)...")
    try:
        jigsaw_dataset = load_dataset("Arsive/toxicity_classification_jigsaw", split="train")
    except Exception as e:
        raise RuntimeError(
            "加载 Hugging Face 数据集 Arsive/toxicity_classification_jigsaw 失败；"
            "请检查网络、datasets 依赖版本或 Hugging Face 连接。"
        ) from e

    jigsaw_rows = list(jigsaw_dataset)
    random.shuffle(jigsaw_rows)
    score_count = 0
    subtype_names = ["severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    for item in jigsaw_rows:
        text = str(item.get("comment_text", "")).strip()
        if len(text) < 2:
            continue
        try:
            subtype_labels = {name: int(item[name]) for name in subtype_names}
            toxic_label = int(item["toxic"])
        except (KeyError, TypeError, ValueError):
            continue
        if any(value not in (0, 1) for value in subtype_labels.values()) or toxic_label not in (0, 1):
            continue

        # SCORE is the count of five subcategory labels; the umbrella toxic
        # label is preserved separately and is not added to the score.
        score_label = sum(subtype_labels.values())
        sample = create_jevbench_sample(
            f"score_jigsaw_toxicity_{score_count:05d}",
            "score",
            text[:4000],
            score_label,
            "arsive_toxicity_classification_jigsaw",
        )
        sample["subtype_labels"] = subtype_labels
        sample["toxic_label"] = toxic_label
        samples.append(sample)
        score_count += 1
        if score_count >= target_score:
            break

    print(f"   Arsive Jigsaw Toxicity: 成功生成 {score_count} 条 score 样本。")
    if len(samples) != total_samples:
        raise RuntimeError(f"样本数量不足：请求 {total_samples} 条，实际生成 {len(samples)} 条；请检查数据源是否完整。")

    random.shuffle(samples)
    return samples
def split_and_save_dataset(all_samples: List[Dict[str, Any]], output_dir: str = "./data"):
    """按 60:20:20 严格切分 train / held_out / eval 并存为标准 JSONL"""
    os.makedirs(output_dir, exist_ok=True)

    # Deduplicate before splitting so identical examples cannot cross split boundaries.
    unique_samples = []
    seen = set()
    for sample in all_samples:
        key = (sample.get("type"), sample.get("text", "").strip())
        if key in seen:
            continue
        seen.add(key)
        unique_samples.append(sample)

    removed_duplicates = len(all_samples) - len(unique_samples)
    random.shuffle(unique_samples)
    all_samples = unique_samples
    if removed_duplicates:
        print(f"切分前移除重复样本: {removed_duplicates} 条")

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

        noul_labels = Counter([x["label"] for x in data if x["type"] == "noul"])
        score_labels = Counter([x["label"] for x in data if x["type"] == "score"])
        source_counts = Counter([x.get("source", "unknown") for x in data])

        print(f"\n[{name.upper()}] 共 {len(data)} 条样本 -> 保存至 {file_path}")
        print(f"  - NOUL 标签分布: {dict(noul_labels)}")
        print(f"  - SCORE 分布: {dict(sorted(score_labels.items()))}")
        print(f"  - 来源分布: {dict(source_counts)}")
    print("=" * 50)


if __name__ == "__main__":
    dataset_samples = load_and_prepare_data()
    split_and_save_dataset(dataset_samples, output_dir="./data")





