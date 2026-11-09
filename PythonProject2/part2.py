from T4 import load_and_prepare_data, split_and_save_dataset
from harness import JevBenchHarness, MockModelAdapter
from T8 import JevBenchMetrics
import json

if __name__ == "__main__":
    print("=== 第一步：运行 T4 生成并切分数据集 ===")
    raw_data = load_and_prepare_data(total_samples=10000)
    split_and_save_dataset(raw_data, output_dir="./data")

    print("\n=== 第二步：运行 T5 评测 Harness ===")
    adapter = MockModelAdapter()
    # 读取 T4 生成的 eval.jsonl 测试集
    harness = JevBenchHarness("./data/eval.jsonl", adapter)
    eval_results = harness.run()
    print(f"Harness 评估完成，共收集到 {len(eval_results)} 条预测记录。")

    print("\n=== 第三步：运行 T8 计算全部指标并生成可视化看板 ===")
    evaluator = JevBenchMetrics(eval_results)

    # 1. 计算并打印 §2.4 的全部数值指标
    all_metrics = evaluator.compute_all_metrics()
    print("【§2.4 全部指标数值汇总】:")
    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))

    # 2. 生成包含全部指标的可视化大图/看板
    dashboard_path = "./data/jevbench_evaluation_dashboard.png"
    evaluator.plot_comprehensive_dashboard(dashboard_path)


    print(f"\n🎉 块 2 全部流程顺利跑通！可视化看板已保存至: {dashboard_path}")