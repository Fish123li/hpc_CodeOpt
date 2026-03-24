import json
import os
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# 配置
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_finetune_dataset.jsonl"
OUTPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_filtered.jsonl"
DISCARD_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_discarded.jsonl"
MAX_WORKERS = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_score(record):
    code_snippet = record.get('output', '')[:4000]
    instruction = record.get('instruction', '')

    # 动态提示词：覆盖 CUDA, MPI, OpenMP, HIP, Fortran
    prompt = f"""
    你是一位精通高性能计算（HPC）的顶级专家，熟练掌握 CUDA, MPI, OpenMP, HIP 以及 Fortran。
    请评估以下代码片段作为训练数据的质量（1-5分）。

    任务指令: {instruction}
    代码内容: 
    {code_snippet}
    ...

    评分标准：
    1分：垃圾数据。代码不完整、无法编译，或仅包含 "TODO"/"FIXME"。
    2分：严重依赖缺失。严重依赖未知的私有库（如 "my_mpi_utils.h"）导致无法理解核心逻辑，或存在严重的并发错误（如死锁）。
    3分：合格。包含核心 HPC 逻辑（如并行计算、消息传递），但可能缺少标准头文件引用或缺少 main 函数。
    4分：良好。逻辑清晰，使用了标准的 HPC 范式（如正确的 MPI 通信模式、安全的 OpenMP 线程管理），代码风格较好。
    5分：完美。完全自包含，使用标准库（<mpi.h>, <omp.h>, <cuda_runtime.h>），包含错误检查和良好的注释。

    注意：
    - 如果是 Fortran 代码，请依据 Fortran 标准评分，不要因为不是 C++ 而打低分。
    - 重点关注并行逻辑的正确性。

    请仅以JSON格式返回结果：
    {{"score": int, "reason": "简短评价"}}
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"score": 0, "reason": str(e)}


def process_line(line):
    try:
        record = json.loads(line)
        if len(record.get('output', '')) < 50:
            return None, record, {"score": 0, "reason": "Code too short"}

        eval_result = get_score(record)
        return record, eval_result
    except json.JSONDecodeError:
        return None, None, None


def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"一共加载了 {total_lines} 条 HPC 数据。")

    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            processed_count += sum(1 for _ in f)
    if os.path.exists(DISCARD_FILE):
        with open(DISCARD_FILE, 'r', encoding='utf-8') as f:
            processed_count += sum(1 for _ in f)

    print(f"已处理 {processed_count} 条，剩余 {total_lines - processed_count} 条。")
    lines_to_process = lines[processed_count:]

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out, \
            open(DISCARD_FILE, 'a', encoding='utf-8') as f_discard:

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_line, line): line for line in lines_to_process}

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines_to_process),
                               desc="HPC Scoring"):
                record, eval_result = future.result()

                if record:
                    score = eval_result.get('score', 0)
                    if 'meta' not in record: record['meta'] = {}
                    record['meta']['quality_score'] = score
                    record['meta']['quality_reason'] = eval_result.get('reason', '')

                    # 保持 3 分及以上
                    if score >= 3:
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    else:
                        f_discard.write(json.dumps(record, ensure_ascii=False) + "\n")

                f_out.flush()
                f_discard.flush()


if __name__ == "__main__":
    main()