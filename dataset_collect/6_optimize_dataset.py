import json
import os
import re
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"  # 替换你的 Key
BASE_URL = "https://api.deepseek.com"
INPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_filtered.jsonl"
OUTPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_final.jsonl"
MAX_WORKERS = 5  # 如果还是报错多，可以尝试降到 3
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_content(text, tag):
    """
    使用正则表达式提取 <TAG>...</TAG> 之间的内容
    re.DOTALL (re.S) 让 . 匹配包括换行符在内的所有字符
    """
    pattern = f"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def optimize_code(record):
    instruction = record['instruction']
    original_code = record['output']

    # 修改后的 System Prompt：不再要求 JSON，而是要求 XML 标签格式
    system_prompt = """你是一个精通高性能计算（HPC）的资深架构师。你的任务是清洗和优化用户提供的 HPC 代码。

    请严格按照以下步骤思考并重构代码：
    1. 【语言保持】：识别代码语言（C++, C, Fortran 等），保持原语言，严禁跨语言翻译。
    2. 【去依赖化】：移除对私有头文件的引用，展开非标准宏，确保代码自包含。
    3. 【安全性】：修复内存泄漏、竞态条件、死锁等并发错误。
    4. 【标准化】：CUDA 用标准 API，MPI/OpenMP 包含标准头文件。

    *** 输出格式要求 ***
    请不要返回 JSON，请直接使用以下标签包裹你的回答：

    <THOUGHT>
    在这里写下你的分析：语言是什么？修复了哪些 bug？去除了哪些依赖？
    </THOUGHT>

    <CODE>
    // 在这里写下完整的、优化后的代码
    // 包含必要的头文件
    ...
    </CODE>
    """

    user_prompt = f"指令：{instruction}\n\n原始代码：\n{original_code}"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # 移除 response_format json_object，改为普通文本模式
            temperature=0.2,
            max_tokens=4096  # 确保生成的代码完整
        )
        content = response.choices[0].message.content

        # 使用正则提取内容
        thought = extract_content(content, "THOUGHT")
        code = extract_content(content, "CODE")

        # 如果正则没提取到，可能模型忘记写标签，尝试备用策略（直接把整个内容当代码，但这很罕见）
        if not code:
            # 简单的 fallback：尝试找 markdown 代码块
            code_match = re.search(r"```\w*\n(.*?)```", content, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            else:
                # 如果实在解析不了，返回 None，丢弃这条数据
                return None

        return {
            "thought": thought if thought else "No thought provided",
            "optimized_code": code
        }

    except Exception as e:
        print(f"API Error: {e}")
        return None


def process_line(line):
    try:
        record = json.loads(line)
        # 跳过已优化的
        if record.get('meta', {}).get('is_optimized'):
            return record

        optimization = optimize_code(record)

        if optimization and optimization.get('optimized_code'):
            new_record = record.copy()
            new_record['output'] = optimization['optimized_code']

            if 'meta' not in new_record: new_record['meta'] = {}
            new_record['meta']['is_optimized'] = True
            new_record['meta']['optimization_thought'] = optimization['thought']

            return new_record
        else:
            # 优化失败保留原样，或者返回 None 丢弃（这里选择保留原样但标记未优化）
            # print(f"Skipping failed optimization for instruction: {record['instruction'][:30]}...")
            return record
    except Exception:
        return None


def main():
    # 读取输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"文件不存在: {INPUT_FILE}")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"开始优化 {total_lines} 条数据 (使用 XML 标签解析模式)...")

    # 断点续传逻辑
    processed_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for _ in f: processed_count += 1

    print(f"已完成 {processed_count} 条，从第 {processed_count + 1} 条继续...")
    lines_to_process = lines[processed_count:]

    # 打开输出文件（追加模式）
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_line, line): line for line in lines_to_process}

            # 进度条
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(lines_to_process),
                               desc="HPC Optimizing"):
                result_record = future.result()
                if result_record:
                    f_out.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                    f_out.flush()


if __name__ == "__main__":
    main()