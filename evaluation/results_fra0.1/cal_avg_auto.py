#!/usr/bin/env python3
"""
自动扫描目录并计算所有 metrics.json 的平均分数
使用方法：
    python cal_avg_auto.py                    # 扫描当前目录
    python cal_avg_auto.py /path/to/results   # 扫描指定目录
"""
import json
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple


def extract_compression_ratio(dirname: str) -> float:
    """从目录名中提取压缩率"""
    match = re.search(r'__cr([\d.]+)', dirname)
    if match:
        return float(match.group(1))
    return 0.0


def calculate_average(file_path: str) -> Tuple[float, int]:
    """
    计算 metrics.json 文件中所有 string_match 值的平均数
    
    Returns:
        (average_score, task_count) 或 (None, 0) 如果出错
    """
    if not os.path.exists(file_path):
        print(f"Warning: File not found at {file_path}")
        return None, 0

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = []
        
        for key, metrics in data.items():
            if isinstance(metrics, dict) and "string_match" in metrics:
                score = metrics["string_match"]
                values.append(score)

        if not values:
            return None, 0

        avg_score = sum(values) / len(values)
        return avg_score, len(values)

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
        return None, 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, 0


def find_and_process_metrics(base_dir: str = "."):
    """
    在指定目录下查找所有包含 metrics.json 的子目录并处理
    支持直接子目录或嵌套在 /1/, /2/ 等子目录下的 metrics.json
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return
    
    # 查找所有包含 metrics.json 的目录
    results = []
    
    for item in base_path.iterdir():
        if item.is_dir():
            metrics_file = item / "metrics.json"
            # 首先检查直接子目录下的 metrics.json
            if metrics_file.exists():
                results.append((str(item), str(metrics_file)))
            else:
                # 如果没找到，检查 /1/, /2/, /3/ 等子目录
                found_any = False
                for i in range(1, 10):  # 检查 1-9
                    nested_metrics = item / str(i) / "metrics.json"
                    if nested_metrics.exists():
                        # 使用父目录名作为显示名，但添加子目录编号
                        results.append((f"{str(item)}/{i}", str(nested_metrics)))
                        found_any = True
                
                # 如果数字子目录都没找到，尝试递归查找第一层子目录
                if not found_any:
                    for subitem in item.iterdir():
                        if subitem.is_dir():
                            sub_metrics = subitem / "metrics.json"
                            if sub_metrics.exists():
                                results.append((f"{str(item)}/{subitem.name}", str(sub_metrics)))
    
    if not results:
        print(f"No directories with metrics.json found in {base_dir}")
        return
    
    # 按目录名排序（尝试按压缩率排序）
    results.sort(key=lambda x: (extract_compression_ratio(x[0]), x[0]))
    
    print("=" * 80)
    print(f"Found {len(results)} result directories in: {base_path.absolute()}")
    print("=" * 80)
    print()
    
    # 收集所有结果以便最后汇总
    summary = []
    
    for dir_path, metrics_path in results:
        # 使用相对于 base_path 的路径作为显示名
        try:
            rel_path = Path(dir_path).relative_to(base_path)
            dir_name = str(rel_path)
        except ValueError:
            dir_name = os.path.basename(dir_path)
        
        cr = extract_compression_ratio(dir_name)
        
        print(f"📁 {dir_name}")
        avg_score, task_count = calculate_average(metrics_path)
        
        if avg_score is not None:
            print(f"   ├─ Tasks: {task_count}")
            print(f"   ├─ Average string_match: {avg_score:.4f}")
            print(f"   └─ Compression Ratio: {cr}")
            summary.append((dir_name, cr, avg_score, task_count))
        else:
            print(f"   └─ Failed to process metrics")
        print()
    
    # 打印汇总表格
    if summary:
        print("=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(f"{'Compression Ratio':<20} {'Avg Score':<15} {'Tasks':<10}")
        print("-" * 80)
        
        for dir_name, cr, score, count in summary:
            cr_str = f"{cr:.2f}" if cr > 0 else "N/A"
            print(f"{cr_str:<20} {score:.4f}{'':<10} {count:<10}")
        
        print("=" * 80)


def main():
    """主函数"""
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        base_dir = "."
    
    find_and_process_metrics(base_dir)


if __name__ == "__main__":
    main()
