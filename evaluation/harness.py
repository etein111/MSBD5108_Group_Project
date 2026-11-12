import json
import time
import os
from typing import List, Dict, Any, Protocol


# 1. 定义块 3 适配器必须遵守的统一 Protocol
class ModelAdapter(Protocol):
    def predict(self, prompt: str, text: str, task_type: str, options: List[Any]) -> tuple:
        """
        统一接口规范：
        返回: (decision, prob, logits_dict, latency_seconds)
        """
        ...


# 2. 编写 Mock 适配器用于开发验证
class MockModelAdapter:
    def predict(self, prompt: str, text: str, task_type: str, options: List[Any]) -> tuple:
        import random
        import numpy as np

        if task_type == "noul":
            decision = random.choice(["yes", "no"])
            # 模拟生成对应 options 的完整概率分布
            raw_p = random.uniform(0.7, 0.99)
            if decision == "yes":
                prob_dict = {"no": round(1.0 - raw_p, 4), "yes": round(raw_p, 4)}
            else:
                prob_dict = {"no": round(raw_p, 4), "yes": round(1.0 - raw_p, 4)}
            logits = {"no": -1.2, "yes": 1.5}

        elif task_type == "score":
            decision = random.choice(options)
            # 模拟 6 分类的完整 softmax 概率分布
            raw_logits = np.array([random.random() for _ in options])
            exp_logits = np.exp(raw_logits - np.max(raw_logits))
            probs = exp_logits / np.sum(exp_logits)
            prob_dict = {str(opt): round(float(p), 4) for opt, p in zip(options, probs)}
            # 确保 decision 对应概率最高（满足 argmax 契约）
            prob_dict[str(decision)] = max(prob_dict.values())
            logits = {str(opt): float(l) for opt, l in zip(options, raw_logits)}
        else:
            decision = options[0] if options else "no"
            prob_dict = {str(opt): 1.0 / len(options) for opt in options}
            logits = {}

        # 增加断言：校验和约等于 1.0
        total_p = sum(prob_dict.values())
        assert abs(total_p - 1.0) < 1e-3, f"概率和不为 1: {total_p}"

        latency_seconds = 0.05
        return decision, prob_dict, logits, latency_seconds


# 3. Harness 评估引擎
class JevBenchHarness:
    def __init__(self, jsonl_path: str, adapter: ModelAdapter):
        self.jsonl_path = jsonl_path
        self.adapter = adapter

    def run(self) -> List[Dict[str, Any]]:
        results = []

        # 修正：将读取和循环逻辑正确缩进到 run 方法内部
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        for item in lines:
            sid = item["id"]
            ttype = item["type"]
            text = item["text"]
            prompt = item.get("prompt", "")
            gt = item["label"]
            opts = item.get("options", [])

            try:
                dec, prob, logits, lat = self.adapter.predict(prompt, text, ttype, opts)
                record = {
                    "id": sid,
                    "type": ttype,
                    "source": item.get("source", "unknown"),
                    "text": text,
                    "ground_truth": gt,
                    "prediction": dec,
                    "probability": prob,
                    "raw_logits": logits,
                    "latency_seconds": lat,
                    "status": "success"
                }
            except Exception as e:
                record = {
                    "id": sid,
                    "type": ttype,
                    "source": item.get("source", "unknown"),
                    "text": text,
                    "ground_truth": gt,
                    "prediction": None,
                    "probability": None,
                    "raw_logits": None,
                    "latency_seconds": 0.0,
                    "status": f"error: {str(e)}"
                }
            # Preserve Jigsaw ground-truth details for downstream audit;
            # these fields are not passed to the model adapter.
            for metadata_field in ("subtype_labels", "toxic_label"):
                if metadata_field in item:
                    record[metadata_field] = item[metadata_field]

            results.append(record)
        return results


if __name__ == "__main__":
    # 假设你刚刚用 T4 生成了 ./data/eval.jsonl
    test_jsonl_path = "./data/eval.jsonl"

    if os.path.exists(test_jsonl_path):
        adapter = MockModelAdapter()
        harness = JevBenchHarness(test_jsonl_path, adapter)
        eval_results = harness.run()
        print(f"Harness 测试成功！共输出了 {len(eval_results)} 条预测结果。")
        print("第一条样例结果:", eval_results[0] if eval_results else "无数据")
    else:
        print(f"未找到测试文件 {test_jsonl_path}，请先运行 T4 生成数据。")