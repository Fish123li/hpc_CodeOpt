import json
import random

# 配置你的最终文件路径
INPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_final.jsonl"


def inspect_dataset():
    data = []
    invalid_count = 0
    empty_output_count = 0

    print(f"正在读取 {INPUT_FILE} 进行最终质检...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                record = json.loads(line)

                # 检查关键字段
                if 'instruction' not in record or 'output' not in record:
                    invalid_count += 1
                    continue

                # 检查内容是否为空
                if not record['output'].strip():
                    empty_output_count += 1
                    continue

                data.append(record)
            except json.JSONDecodeError:
                invalid_count += 1
                print(f"Line {i + 1}: JSON 解析失败")

    print("-" * 30)
    print(f"总行数: {len(data) + invalid_count + empty_output_count}")
    print(f"有效数据: {len(data)}")
    print(f"格式错误/缺失字段: {invalid_count}")
    print(f"Output为空 (将被清洗): {empty_output_count}")
    print("-" * 30)

    # 随机预览 3 条数据
    print("\n=== 数据预览 (请确认代码是否完整，是否有Markdown标记) ===")
    samples = random.sample(data, min(3, len(data)))
    for idx, sample in enumerate(samples):
        print(f"\n[Sample {idx + 1}]")
        print(f"指令: {sample['instruction'][:100]}...")
        print(f"代码片段 (前200字符):\n{sample['output'][:200]}")
        print("-" * 20)

        # 检查是否包含 markdown 代码块标记
        if "```" not in sample['output']:
            print("⚠️ 警告: 这条数据的 Output 中没有发现 markdown (```) 标记。")
            print("如果这是纯代码，建议在微调时加上 prompt template，或者统一加上 markdown 标记。")


if __name__ == "__main__":
    inspect_dataset()