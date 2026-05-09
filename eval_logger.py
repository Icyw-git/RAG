import datetime
import json
import os
import shutil
from typing import Dict, List, Optional, Any


class EvalLogger:
    """评测日志管理器 - 结构化记录每次评测的运行配置、指标和预测结果"""

    def __init__(self, log_root: str = "eval_logs"):
        self.log_root = log_root # 日志的根目录
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") #保存测试时间
        self.run_dir = f"{log_root}/{self.run_id}"
        os.makedirs(self.run_dir, exist_ok=True) #创建当前的时间日志目录

    def log_config(self, config: dict):
        """保存运行配置"""
        with open(f"{self.run_dir}/config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2) #创建配置文件，保存当前测试的配置参数

    def log_metrics(self, metrics: dict):
        """保存评测指标"""
        with open(f"{self.run_dir}/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def save_predictions(self, pred_path: str):
        """复制预测结果到日志目录"""
        if os.path.exists(pred_path):
            shutil.copy2(pred_path, f"{self.run_dir}/predictions.jsonl") #使用shutil.copy2保留原文件的元数据（如修改时间等）

    def save_badcases(self, badcases: list):
        """保存错误样本"""
        with open(f"{self.run_dir}/badcases.jsonl", "w", encoding="utf-8") as f:
            for case in badcases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")

    def update_summary_csv(self, config: dict, metrics: dict):
        """更新汇总CSV（所有 runs 的指标横向对比）"""
        csv_path = f"{self.log_root}/summary.csv"

        # 展平嵌套的 metrics（如 "hit_source@5" 键名中的 @ 保留）
        flat_config = self._flatten_dict(config, prefix="cfg") #加上前缀名，区分配置和指标
        flat_metrics = self._flatten_dict(metrics, prefix="")

        header = ["run_id"] + list(flat_config.keys()) + list(flat_metrics.keys()) #组装表头
        row = [self.run_id] + list(flat_config.values()) + list(flat_metrics.values()) #组装当前行数据

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            if not file_exists: #第一次创建文件的时候写入表头
                f.write(",".join(header) + "\n")
            f.write(",".join(str(v) for v in row) + "\n") #之后就直接写入数据行

        print(f"📊 Summary updated: {csv_path}")

    def print_summary(self, metrics: dict):
        """打印终端摘要"""
        print("\n" + "=" * 50)
        print(f"📋 Run ID: {self.run_id}")
        print("=" * 50)
        for key, val in metrics.items():
            if isinstance(val, float):
                print(f"  {key:20s}: {val:.4f}")
            else:
                print(f"  {key:20s}: {val}")
        print(f"📁 Logs saved to: {self.run_dir}")

    @staticmethod
    def _flatten_dict(d: dict, prefix: str = "") -> dict: #这里使用了静态方法，表示只作为该类使用的工具函数
        """将嵌套字典展平为单层（用于 CSV 列名）"""
        flat = {}
        for key, val in d.items():
            flat_key = f"{prefix}_{key}" if prefix else key
            if isinstance(val, dict):
                flat.update(EvalLogger._flatten_dict(val, prefix=flat_key)) #递归展平嵌套的字典
            else:
                flat[flat_key] = val
        return flat

    @classmethod
    def load_summary(cls, log_root: str = "eval_logs") -> list: #这里使用了类方法，表示该方法与类相关但不依赖于实例，可以直接通过类调用，利于继承
        """加载所有历史 summary.csv 数据"""
        import csv
        csv_path = f"{log_root}/summary.csv"
        if not os.path.exists(csv_path):
            print(f"No summary.csv found at {csv_path}")
            return []

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    @classmethod
    def print_comparison(cls, log_root: str = "eval_logs"):
        """打印所有 runs 的对比表"""
        rows = cls.load_summary(log_root)
        if not rows:
            return

        print("\n" + "=" * 80)
        print("📊 ALL RUNS COMPARISON")
        print("=" * 80)

        # 提取所有列名
        headers = list(rows[0].keys())
        # 格式化打印
        col_widths = {h: max(len(h), 12) for h in headers}
        for row in rows:
            for h in headers:
                col_widths[h] = max(col_widths.get(h, 12), len(str(row[h])), 12)

        # 打印表头
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))

        # 打印数据
        for row in rows:
            data_line = " | ".join(str(row[h]).ljust(col_widths[h]) for h in headers)
            print(data_line)

        print("=" * 80)