import json
import sys
import os

def calculate_average(file_path):
    """
    计算 metrics.json 文件中所有 string_match 值的平均数
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        values = []
        print(f"--- Processing: {os.path.basename(os.path.dirname(os.path.dirname(file_path)))} ---")
        
        for key, metrics in data.items():
            if isinstance(metrics, dict) and "string_match" in metrics:
                score = metrics["string_match"]
                values.append(score)
                # print(f"{key}: {score}") # 取消注释以查看每个任务的详细分数

        if not values:
            print("No 'string_match' keys found in the file.")
            return

        avg_score = sum(values) / len(values)
        print(f"Tasks count: {len(values)}")
        print(f"Average string_match: {avg_score:.4f}")
        print("-" * 30)

    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    default_ratio = ("0.25", "0.50", "0.75", "0.90")
    for ratio in default_ratio:
        # default_path = f"ruler__4096__llama3.1-8b-instruct-query_indexer_score_larger-6140__query_indexer_score_block__cr{ratio}__compressed_questions/metrics.json"
        # default_path = f"ruler__4096__llama3.1-8b-instruct-memory-larger-9210__query_indexer_score_block__cr{ratio}__compressed_questions/metrics.json"
        # default_path = f"ruler__4096__llama3.1-8b-instruct-query-indexer_score_long__query_indexer_score_block__cr{ratio}__compressed_questions/metrics.json"
        default_path = f"ruler__4096__llama3.1-8b-instruct-query-indexer_score_long__query_indexer_kvzip__cr{ratio}__compressed_questions/metrics.json"

        calculate_average(default_path)