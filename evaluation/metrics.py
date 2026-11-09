import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
    brier_score_loss
)
from typing import List, Dict, Any

class JevBenchMetrics:
    def __init__(self, eval_results: List[Dict[str, Any]]):
        self.results = eval_results

    def compute_all_metrics(self) -> Dict[str, Any]:
        """按任务类型分别计算核心指标，并汇总整体表现"""
        noul_true, noul_pred, noul_probs = [], [], []
        score_true, score_pred, score_probs = [], [], []

        for item in self.results:
            if item.get("status") != "success":
                continue

            ttype = item["type"]
            gt = item["ground_truth"]
            pred = item["prediction"]
            prob = item.get("probability", 0.0)

            if ttype == "noul":
                noul_true.append(1 if str(gt).lower() == "yes" else 0)
                noul_pred.append(str(pred).lower())
                p_yes = prob if str(pred).lower() == "yes" else (1.0 - prob)
                noul_probs.append(p_yes)
            elif ttype == "score":
                score_true.append(int(gt))
                score_pred.append(int(pred))
                score_probs.append(float(prob))

        metrics_summary = {}

        # 1. NOUL 任务指标
        if noul_true:
            noul_true_bin = np.array(noul_true)
            noul_pred_bin = [1 if p == "yes" else 0 for p in noul_pred]

            exact_match = accuracy_score(noul_true_bin, noul_pred_bin)
            macro_f1 = f1_score(noul_true_bin, noul_pred_bin, average="macro", zero_division=0)
            calib_noul = self._compute_calibration(noul_true_bin, np.array(noul_probs), n_bins=15)

            metrics_summary["noul"] = {
                "sample_count": len(noul_true),
                "exact_match": float(exact_match),
                "macro_f1": float(macro_f1),
                "ece_15bin": calib_noul["ece"],
                "brier_score": calib_noul["brier_score"],
                "calibration_details": calib_noul
            }

        # 2. SCORE 任务指标
        if score_true:
            exact_match = accuracy_score(score_true, score_pred)
            mae = mean_absolute_error(score_true, score_pred)
            qwk = cohen_kappa_score(score_true, score_pred, weights="quadratic")

            metrics_summary["score"] = {
                "sample_count": len(score_true),
                "exact_match": float(exact_match),
                "mae": float(mae),
                "qwk": float(qwk),
                "true_values": score_true,
                "pred_values": score_pred
            }

        return metrics_summary

    def _compute_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> dict:
        brier = float(brier_score_loss(y_true, y_prob)) if len(y_true) > 0 else 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        n_samples = len(y_true)
        bin_accs, bin_confs, bin_sizes = [], [], []

        for i in range(n_bins):
            l_edge, u_edge = bin_edges[i], bin_edges[i+1]
            if i == n_bins - 1:
                in_bin = (y_prob >= l_edge) & (y_prob <= u_edge)
            else:
                in_bin = (y_prob >= l_edge) & (y_prob < u_edge)

            bin_size = np.sum(in_bin)
            bin_sizes.append(int(bin_size))

            if bin_size > 0:
                avg_conf = np.mean(y_prob[in_bin])
                avg_acc = np.mean(y_true[in_bin])
                ece += (bin_size / n_samples) * np.abs(avg_acc - avg_conf)
                bin_accs.append(float(avg_acc))
                bin_confs.append(float(avg_conf))
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        return {
            "ece": float(ece),
            "brier_score": brier,
            "bin_accs": bin_accs,
            "bin_confs": bin_confs,
            "bin_sizes": bin_sizes,
            "bin_edges": bin_edges.tolist()
        }

    def plot_comprehensive_dashboard(self, save_path: str = "./jevbench_evaluation_dashboard.png"):
        """绘制包含 §2.4 全部指标的可视化综合看板大图"""
        metrics = self.compute_all_metrics()
        has_noul = "noul" in metrics
        has_score = "score" in metrics

        if not has_noul and not has_score:
            print("没有找到有效的评测数据，无法生成可视化看板。")
            type(self)
            return

        # 创建一个 1 行 2 或 3 列的画布
        cols = 2 if (has_noul and has_score) else 1
        fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 6), dpi=300)
        if cols == 1:
            axes = [axes]

        ax_idx = 0

        # --- 子图 1：NOUL 任务指标与 Reliability 校准图 ---
        if has_noul:
            ax = axes[ax_idx]
            noul_data = metrics["noul"]
            calib = noul_data["calibration_details"]
            bin_edges = calib["bin_edges"]
            bin_accs = calib["bin_accs"]
            ece = noul_data["ece_15bin"]
            brier = noul_data["brier_score"]
            em = noul_data["exact_match"]
            f1 = noul_data["macro_f1"]

            bin_centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2.0
            ax.bar(bin_centers, bin_accs, width=1.0/len(bin_centers), alpha=0.7, edgecolor="black", color="royalblue", label="Model Outputs")
            ax.plot([0, 1], [0, 1], "--", color="red", label="Perfect Calibration")

            # 在图内用文本框把 Exact Match 和 Macro-F1 以及 Brier 标出来
            text_str = f"Exact Match: {em:.3f}\nMacro-F1: {f1:.3f}\nBrier Score: {brier:.3f}\nECE (15-bin): {ece:.4f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

            ax.set_xlabel("Confidence (Predicted Probability)", fontsize=11)
            ax.set_ylabel("Accuracy (True Positive Rate)", fontsize=11)
            ax.set_title("NOUL Task: Calibration & Metrics", fontsize=12, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax_idx += 1

        # --- 子图 2：SCORE 任务指标与误差分布图 ---
        if has_score:
            ax = axes[ax_idx]
            score_data = metrics["score"]
            t_vals = np.array(score_data["true_values"])
            p_vals = np.array(score_data["pred_values"])
            errors = p_vals - t_vals # 残差

            em = score_data["exact_match"]
            mae = score_data["mae"]
            qwk = score_data["qwk"]

            # 绘制残差分布直方图
            ax.hist(errors, bins=range(-5, 7), align='left', alpha=0.7, color='teal', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero Error')

            text_str = f"Exact Match: {em:.3f}\nMAE: {mae:.3f}\nQWK: {qwk:.3f}"
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)

            ax.set_xlabel("Prediction Error (Pred - True)", fontsize=11)
            ax.set_ylabel("Sample Count", fontsize=11)
            ax.set_title("SCORE Task: Error Distribution & Metrics", fontsize=12, fontweight='bold')
            ax.legend(loc="upper right")
            ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"🎉 综合可视化评估看板已成功保存至: {save_path}")

if __name__ == "__main__":
    mock_results = [
        {"id": "noul_001", "type": "noul", "ground_truth": "yes", "prediction": "yes", "probability": 0.92, "status": "success"},
        {"id": "noul_002", "type": "noul", "ground_truth": "no", "prediction": "no", "probability": 0.85, "status": "success"},
        {"id": "score_001", "type": "score", "ground_truth": 5, "prediction": 4, "probability": 0.80, "status": "success"},
        {"id": "score_002", "type": "score", "ground_truth": 0, "prediction": 0, "probability": 0.95, "status": "success"},
    ]

    evaluator = JevBenchMetrics(mock_results)
    summary = evaluator.compute_all_metrics()
    print("【T8 指标数值汇总】:")
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 生成包含全部指标的可视化大图
    evaluator.plot_comprehensive_dashboard("./data/jevbench_evaluation_dashboard.png")