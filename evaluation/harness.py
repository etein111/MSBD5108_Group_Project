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
        """
        Mock 适配器：模拟模型预测并返回规定格式
        """
        import random

        # 简单模拟不同任务的输出
        if task_type == "noul":
            decision = random.choice(["yes", "no"])
            probability = round(random.uniform(0.7, 0.99), 4)
            logits = {"no": -1.2, "yes": 1.5}
        elif task_type == "score":
            decision = random.randint(0, 5)
            probability = round(random.uniform(0.6, 0.95), 4)
            logits = {str(i): -random.random() for i in options}
        else:
            decision = "no"
            probability = 0.5
            logits = {}

        latency_seconds = 0.05  # 模拟 50ms 耗时

        return decision, probability, logits, latency_seconds


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
                predict_record = getattr(self.adapter, "predict_record", None)
                if callable(predict_record):
                    dec, prob, logits, lat = predict_record(item)
                else:
                    dec, prob, logits, lat = self.adapter.predict(prompt, text, ttype, opts)
                record = {
                    "id": sid,
                    "type": ttype,
                    "text": text,
                    "ground_truth": gt,
                    "prediction": dec,
                    "probability": prob,
                    "raw_logits": logits,
                    "latency": lat,
                    "status": "success"
                }
            except Exception as e:
                record = {
                    "id": sid,
                    "type": ttype,
                    "text": text,
                    "ground_truth": gt,
                    "prediction": None,
                    "probability": None,
                    "raw_logits": None,
                    "latency": 0.0,
                    "status": f"error: {str(e)}"
                }
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
