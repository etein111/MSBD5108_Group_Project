import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
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
        noul_true, noul_probs_list, noul_preds = [], [], []
        score_true, score_probs_matrix, score_preds = [], [], []
        latencies = {"noul": [], "score": []}
        noul_by_source = {}

        for item in self.results:
            if item.get("status") != "success":
                continue

            ttype = item["type"]
            gt = item["ground_truth"]
            pred = item["prediction"]
            prob_dict = item.get("probability", {})  # 现在是字典
            latency_seconds = item.get("latency_seconds", item.get("latency"))
            if ttype in latencies and latency_seconds is not None:
                latencies[ttype].append(float(latency_seconds))

            # 契约校验：概率必须是完整分布，且预测类别必须对应最大概率。
            if not isinstance(prob_dict, dict) or not prob_dict:
                raise ValueError(f"{item.get('id', '<unknown>')}: probability 必须是非空字典")
            try:
                probabilities = {str(k): float(v) for k, v in prob_dict.items()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{item.get('id', '<unknown>')}: probability 含非数值") from exc
            values = np.array(list(probabilities.values()), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
                raise ValueError(f"{item.get('id', '<unknown>')}: probability 必须是 [0, 1] 内的有限数值")
            if not np.isclose(np.sum(values), 1.0, atol=1e-3):
                raise ValueError(f"{item.get('id', '<unknown>')}: probability 总和必须约为 1")
            pred_key = str(pred)
            if pred_key not in probabilities:
                raise ValueError(f"{item.get('id', '<unknown>')}: prediction 不在 probability 类别中")
            if not np.isclose(probabilities[pred_key], np.max(values), atol=1e-6):
                raise ValueError(f"{item.get('id', '<unknown>')}: prediction 必须对应最大概率类别")

            if ttype == "noul":
                noul_true.append(1 if str(gt).lower() == "yes" else 0)
                noul_preds.append(1 if str(pred).lower() == "yes" else 0)
                # 提取正类 "yes" 的概率作为校准基准
                p_yes = prob_dict.get("yes", 0.5)
                noul_probs_list.append(p_yes)
                source = str(item.get("source", "unknown"))
                group = noul_by_source.setdefault(source, {"true": [], "pred": [], "prob": []})
                group["true"].append(1 if str(gt).lower() == "yes" else 0)
                group["pred"].append(1 if str(pred).lower() == "yes" else 0)
                group["prob"].append(float(p_yes))

            elif ttype == "score":
                score_true.append(int(gt))
                score_preds.append(int(pred))
                # 将 score 的概率字典按 [0, 1, 2, 3, 4, 5] 顺序转为概率向量
                p_vec = [float(prob_dict.get(str(i), 0.0)) for i in range(6)]
                score_probs_matrix.append(p_vec)

        metrics_summary = {}

        # 1. NOUL 任务指标与校准
        if noul_true:
            y_t = np.array(noul_true)
            y_p = np.array(noul_preds)
            y_prob = np.array(noul_probs_list)

            exact_match = accuracy_score(y_t, y_p)
            macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)
            balanced_accuracy = balanced_accuracy_score(y_t, y_p)
            auroc = float(roc_auc_score(y_t, y_prob)) if len(np.unique(y_t)) == 2 else None
            calib_noul = self._compute_calibration(y_t, y_prob, n_bins=15)

            source_metrics = {}
            for source, group in sorted(noul_by_source.items()):
                src_true = np.array(group["true"])
                src_pred = np.array(group["pred"])
                src_prob = np.array(group["prob"])
                src_calib = self._compute_calibration(src_true, src_prob, n_bins=15)
                source_metrics[source] = {
                    "sample_count": int(len(src_true)),
                    "exact_match": float(accuracy_score(src_true, src_pred)),
                    "macro_f1": float(f1_score(src_true, src_pred, average="macro", zero_division=0)),
                    "balanced_accuracy": float(balanced_accuracy_score(src_true, src_pred)),
                    "auroc": float(roc_auc_score(src_true, src_prob)) if len(np.unique(src_true)) == 2 else None,
                    "ece_15bin": src_calib["ece"],
                    "brier_score": src_calib["brier_score"]
                }
            metrics_summary["noul"] = {
                "sample_count": len(noul_true),
                "exact_match": float(exact_match),
                "macro_f1": float(macro_f1),
                "exact_match_ci95": self._bootstrap_accuracy_ci(y_t, y_p),
                "balanced_accuracy": float(balanced_accuracy),
                "auroc": auroc,
                "ece_15bin": calib_noul["ece"],
                "brier_score": calib_noul["brier_score"],
                "calibration_details": calib_noul,
                "by_source": source_metrics
            }

        # 2. SCORE 任务指标与多分类 Brier / 校准
        if score_true:
            y_t = np.array(score_true)
            y_p = np.array(score_preds)
            P_matrix = np.array(score_probs_matrix) # 形状: (N, 6)

            exact_match = accuracy_score(y_t, y_p)
            mae = mean_absolute_error(y_t, y_p)
            qwk = cohen_kappa_score(y_t, y_p, weights="quadratic")

            # 多分类 Brier Score 计算: 1/N * sum_i sum_k (p_ik - y_ik)^2
            Y_onehot = np.eye(6)[y_t] # (N, 6)
            brier_score_multiclass = float(np.mean(np.sum((P_matrix - Y_onehot) ** 2, axis=1)))

            # 多分类置信度校准（取每行最大概率作为 confidence，预测是否等于 ground_truth 作为 accuracy）
            confidences = np.max(P_matrix, axis=1)
            correctness = (y_p == y_t).astype(int)
            calib_score = self._compute_calibration(correctness, confidences, n_bins=15)

            metrics_summary["score"] = {
                "sample_count": len(score_true),
                "exact_match": float(exact_match),
                "mae": float(mae),
                "qwk": float(qwk),
                "exact_match_ci95": self._bootstrap_accuracy_ci(y_t, y_p),
                "brier_score": brier_score_multiclass,
                "ece_15bin": calib_score["ece"],
                "calibration_details": calib_score,
                "true_values": score_true,
                "pred_values": score_preds
            }

        for task_name, task_latencies in latencies.items():
            if task_name in metrics_summary and task_latencies:
                metrics_summary[task_name]["latency_p50_seconds"] = float(np.percentile(task_latencies, 50))
                metrics_summary[task_name]["latency_p95_seconds"] = float(np.percentile(task_latencies, 95))
                metrics_summary[task_name]["gpu_seconds_per_1000"] = float(np.mean(task_latencies) * 1000)

        return metrics_summary

    @staticmethod
    def compare_paired(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], task_type: str, n_resamples: int = 10000) -> Dict[str, Any]:
        """Compare two models on exactly the same sample IDs."""
        a = {str(r["id"]): r for r in results_a if r.get("status") == "success" and r.get("type") == task_type}
        b = {str(r["id"]): r for r in results_b if r.get("status") == "success" and r.get("type") == task_type}
        if not a or set(a) != set(b):
            raise ValueError("两模型必须包含相同且非空的成功样本 ID")
        ids = sorted(a)
        truth = np.array([float(a[i]["ground_truth"]) if task_type == "score" else float(str(a[i]["ground_truth"]).lower() == "yes") for i in ids])
        pred_a = np.array([float(a[i]["prediction"]) if task_type == "score" else float(str(a[i]["prediction"]).lower() == "yes") for i in ids])
        pred_b = np.array([float(b[i]["prediction"]) if task_type == "score" else float(str(b[i]["prediction"]).lower() == "yes") for i in ids])
        rng = np.random.default_rng(42)
        n = len(ids)
        acc_a = float(np.mean(pred_a == truth))
        acc_b = float(np.mean(pred_b == truth))
        boot_indices = rng.integers(0, n, size=(2000, n))
        delta_boot = np.mean((pred_a[boot_indices] == truth[boot_indices]).astype(float) - (pred_b[boot_indices] == truth[boot_indices]).astype(float), axis=1)
        ci_low, ci_high = np.percentile(delta_boot, [2.5, 97.5])
        result = {"sample_count": n, "accuracy_a": acc_a, "accuracy_b": acc_b, "accuracy_delta_a_minus_b": acc_a - acc_b, "accuracy_delta_ci95": {"lower": float(ci_low), "upper": float(ci_high)}}
        if task_type == "noul":
            correct_a = pred_a == truth
            correct_b = pred_b == truth
            a_only = int(np.sum(correct_a & ~correct_b))
            b_only = int(np.sum(~correct_a & correct_b))
            discordant = a_only + b_only
            tail = min(a_only, b_only)
            p_value = min(1.0, 2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2 ** discordant)) if discordant else 1.0
            result.update({"test": "exact McNemar", "a_only_correct": a_only, "b_only_correct": b_only, "p_value": p_value})
        elif task_type == "score":
            abs_error_diff = np.abs(pred_a - truth) - np.abs(pred_b - truth)
            observed = abs(float(np.mean(abs_error_diff)))
            signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, n))
            permuted = np.abs(np.mean(signs * abs_error_diff, axis=1))
            p_value = float((np.sum(permuted >= observed) + 1) / (n_resamples + 1))
            result.update({"test": "paired permutation on MAE", "mae_a": float(np.mean(np.abs(pred_a - truth))), "mae_b": float(np.mean(np.abs(pred_b - truth))), "p_value": p_value})
        else:
            raise ValueError("task_type must be 'noul' or 'score'")
        return result
    def _bootstrap_accuracy_ci(self, y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 2000) -> Dict[str, float]:
        """Percentile 95% bootstrap CI for exact-match accuracy."""
        n = len(y_true)
        if n == 0:
            return {"lower": 0.0, "upper": 0.0}
        rng = np.random.default_rng(42)
        indices = rng.integers(0, n, size=(n_bootstrap, n))
        scores = np.mean(y_true[indices] == y_pred[indices], axis=1)
        lower, upper = np.percentile(scores, [2.5, 97.5])
        return {"lower": float(lower), "upper": float(upper)}

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
        {"id": "noul_001", "type": "noul", "ground_truth": "yes", "prediction": "yes", "probability": {"no": 0.08, "yes": 0.92}, "status": "success"},
        {"id": "noul_002", "type": "noul", "ground_truth": "no", "prediction": "no", "probability": {"no": 0.85, "yes": 0.15}, "status": "success"},
        {"id": "score_001", "type": "score", "ground_truth": 5, "prediction": 4, "probability": {"0": 0.02, "1": 0.03, "2": 0.04, "3": 0.06, "4": 0.80, "5": 0.05}, "status": "success"},
        {"id": "score_002", "type": "score", "ground_truth": 0, "prediction": 0, "probability": {"0": 0.95, "1": 0.01, "2": 0.01, "3": 0.01, "4": 0.01, "5": 0.01}, "status": "success"},
    ]

    evaluator = JevBenchMetrics(mock_results)
    summary = evaluator.compute_all_metrics()
    print("【T8 指标数值汇总】:")
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # 生成包含全部指标的可视化大图
    evaluator.plot_comprehensive_dashboard("./outputs/mock_jevbench_evaluation_dashboard.png")



