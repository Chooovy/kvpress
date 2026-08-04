#!/usr/bin/env python3
"""
自动扫描目录并计算所有 metrics.json 的平均分数，输出格式化表格
使用方法：
    python cal_avg_auto.py                    # 扫描当前目录
    python cal_avg_auto.py /path/to/results   # 扫描指定目录
"""
import json
import sys
import os
from pathlib import Path
from typing import Tuple

def read_config(config_path: str) -> Tuple[str, str, str, str, str]:
    """
    读取 config.yaml 并返回 (model, cr, press, dataset, split)
    """
    config = {}
    try:
        # 读取整个文件内容
        with open(config_path, 'r') as f:
            content = f.read()
            
        # 简单解析 key: value
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if ':' in line:
                parts = line.split(':', 1)
                key = parts[0].strip()
                val = parts[1].strip()
                config[key] = val
        
        # 提取 Model (取路径最后一部分)
        model_path = config.get('model', 'Unknown')
        model_name = os.path.basename(model_path)
        
        # 提取 CR
        cr = config.get('compression_ratio', 'N/A')
        
        # 提取 Press 并根据 init command 处理后缀 (匹配截图中的 naming convention)
        press_name = config.get('press_name', 'Unknown')
        
        # 提取 Dataset
        dataset = config.get('dataset', 'Unknown')

        # 提取 Split (data_dir)
        split = config.get('data_dir', 'Unknown')
        
        # # 在整个内容中查找 press_init_command 的参数
        # if 'mean_head=True' in content:
        #     press_name += '_head_mean'
        # elif 'layer_running_mean=True' in content:
        #     press_name += '_layer_mean'
            
        return model_name, cr, press_name, dataset, split
        
    except Exception:
        return "Unknown", "N/A", "Unknown", "Unknown", "Unknown"

def calculate_average(file_path: str) -> float:
    """
    计算 metrics.json 平均分，返回百分制数值 (0-100)
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = []
        for metrics in data.values():
            if isinstance(metrics, dict) and "string_match" in metrics:
                values.append(metrics["string_match"])

        if not values:
            return None

        # 转换为百分制，匹配截图格式
        return (sum(values) / len(values))

    except Exception:
        return None

def find_and_process_metrics(base_dir: str = "."):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: {base_dir} not found")
        return
    
    results = []
    
    # 遍历所有子目录 (例如 ruler__...)
    for item in base_path.iterdir():
        if not item.is_dir():
            continue
            
        metrics_file = None
        # 1. 检查直接子目录下的 metrics.json
        if (item / "metrics.json").exists():
            metrics_file = item / "metrics.json"
        else:
            # 2. 检查 /1/, /2/ 等数字子目录下的 metrics.json
            for i in range(1, 10):
                if (item / str(i) / "metrics.json").exists():
                    metrics_file = item / str(i) / "metrics.json"
                    break
        
        if metrics_file:
            # config.yaml 通常在实验根目录下 (即 item 下)
            config_file = item / "config.yaml"
            results.append({
                "metrics": str(metrics_file),
                "config": str(config_file) if config_file.exists() else None
            })

    if not results:
        print(f"No metrics found in {base_dir}")
        return

    # 打印表头
    print(f"{'model':<30} {'dataset':<15} {'split':<10} {'cr':<6} {'press':<70} {'avg':<10}")
    
    data_rows = []
    for entry in results:
        avg = calculate_average(entry["metrics"])
        if avg is None:
            continue
            
        if entry["config"]:
            model, cr, press, dataset, split = read_config(entry["config"])
        else:
            model, cr, press, dataset, split = "Unknown", "N/A", "Unknown", "Unknown", "Unknown"
            
        data_rows.append({
            "model": model,
            "dataset": dataset,
            "split": split,
            "cr": cr,
            "press": press,
            "avg": avg
        })
    
    # 排序：Model -> Dataset -> Split -> CR (数值) -> Press
    def sort_key(x):
        try:
            cr_val = float(x["cr"])
        except ValueError:
            cr_val = -1.0
        return (x["model"], x["dataset"], x["split"], cr_val, x["press"])
        
    data_rows.sort(key=sort_key)
    
    for row in data_rows:
        print(f"{row['model']:<30} {row['dataset']:<15} {row['split']:<10} {str(row['cr']):<6} {row['press']:<70} {row['avg']:.1f}")

if __name__ == "__main__":
    main_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    find_and_process_metrics(main_dir)
