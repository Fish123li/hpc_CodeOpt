import json
import time
import re
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置部分 =================
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"
BASE_URL = "https://api.deepseek.com"
# 使用原始字符串 r"" 解决路径报错问题
INPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_raw_dataset.jsonl"
OUTPUT_FILE =r"E:\about_task\HPC_Model_FT\middle_collect\hpc_finetune_dataset.jsonl"
MAX_WORKERS = 5  # 根据你的 API 限速调整，建议 5-10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# ================= 智能分类与 Prompt 工程 =================

def detect_hpc_type(code, file_path):
    """
    根据代码内容和文件路径，判断任务类型
    """
    code_lower = code.lower()
    path_lower = file_path.lower()

    if "cuda" in code_lower or "__global__" in code_lower or ".cu" in path_lower:
        return "cuda_hpc"
    elif "mpi_" in code_lower or "mpi.h" in code_lower:
        return "mpi_hpc"
    elif "omp_" in code_lower or "#pragma omp" in code_lower:
        return "openmp_hpc"
    elif ".h" in path_lower or ".hpp" in path_lower:
        return "header_definition"
    else:
        return "general_cpp"


def get_system_prompt(task_type):
    """
    为不同类型的代码定制 System Prompt，确保专业性
    """
    base_prompt = "你是一个世界顶级的HPC（高性能计算）架构师和代码专家。你的任务是构建用于微调LLM的高质量指令数据集。"

    if task_type == "cuda_hpc":
        return base_prompt + "当前任务是针对 CUDA 并行编程。请生成一个极其专业的需求描述，涉及 Kernel 优化、Grid/Block 维度配置或显存管理。"
    elif task_type == "mpi_hpc":
        return base_prompt + "当前任务是针对 MPI 分布式计算。生成的指令应关注通信模式（如广播、规约）、进程排名（Rank）控制或死锁避免。"
    elif task_type == "header_definition":
        return base_prompt + "当前代码是头文件接口定义。请生成一个请求设计类结构或 API 接口的指令。"
    else:
        return base_prompt + "请生成一个清晰、具体的 C++ 编程需求。"


def generate_prompt_messages(code, file_path, task_type):
    """
    构建发送给 DeepSeek 的最终消息
    """
    # 提取文件名作为上下文
    filename = os.path.basename(file_path)
    dir_context = os.path.dirname(file_path).replace("\\", "/")[-30:]  # 取最后层级目录

    user_content = f"""
    我有一段位于 `{dir_context}/{filename}` 的高质量 HPC 代码。
    请你扮演一个用户，根据这段代码的逻辑，**反推**出一个想让 AI 写出这段代码的【Prompt指令】。

    **严格要求：**
    1. **具体性**：指令必须明确提及代码的核心功能（例如："{filename} 中的内存混合管理类"）。
    2. **技术栈**：如果代码用了 CUDA/MPI，指令中必须包含这些关键词。
    3. **纯净输出**：直接输出指令内容，**绝对不要**输出 "好的"、"这是指令"、引号或其他废话。
    4. **逻辑对应**：不要凭空捏造代码里没有的功能。

    **代码片段（前 3500 字符）：**
    ```cpp
    {code[:3500]} 
    ```
    """
    return [
        {"role": "system", "content": get_system_prompt(task_type)},
        {"role": "user", "content": user_content}
    ]


# ================= 辅助清洗函数 =================

def clean_instruction(text):
    """
    清洗模型返回的文本，去除由于模型“太有礼貌”产生的噪音
    """
    # 去除首尾空白
    text = text.strip()
    # 去除可能存在的首尾引号
    text = re.sub(r'^["\']|["\']$', '', text)
    # 去除常见的对话前缀
    text = re.sub(r'^(Here is the instruction|Instruction:|Prompt:|User:|指令：|请写一个).*?[:：]\s*', '', text,
                  flags=re.IGNORECASE)
    return text


# ================= 核心处理逻辑 =================

def process_single_line(line_str):
    try:
        data = json.loads(line_str)
        code = data.get("code", "")
        file_path = data.get("file_path", "")

        # === 质量过滤器 1: 长度过滤 ===
        # 代码太短通常意味着信息量不足，或者是简单的宏定义，不适合做指令微调
        if len(code) < 80:
            return None

        # === 质量过滤器 2: 排除纯注释或空行比例过高的文件 ===
        # 简单判断：如果有效代码行太少，跳过（此处仅做简单长度判断，可扩展）

        # 智能分类
        task_type = detect_hpc_type(code, file_path)
        messages = generate_prompt_messages(code, file_path, task_type)

        # 重试机制
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.6,  # 降低温度，让指令更精准，减少幻觉
                    max_tokens=512
                )

                raw_instruction = response.choices[0].message.content
                clean_inst = clean_instruction(raw_instruction)

                # === 质量过滤器 3: 结果校验 ===
                # 如果生成的指令太短（比如模型拒绝回答），丢弃
                if len(clean_inst) < 10:
                    continue

                new_record = {
                    "instruction": clean_inst,
                    "input": "",  # Code Generation 任务通常 Input 为空
                    "output": code,
                    "meta": {
                        "source": "hpc_repo",
                        "file_name": os.path.basename(file_path),
                        "task_tag": task_type,  # 保留标签，后续可以分析数据分布
                        "length": len(code)
                    }
                }
                return json.dumps(new_record, ensure_ascii=False)

            except Exception as e:
                if attempt == 2:
                    # print(f"Failed processing {file_path}: {e}") # 调试时可打开
                    pass
                time.sleep(1)
        return None

    except Exception as e:
        return None


def main():
    # 检查输入文件
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到文件 {INPUT_FILE}")
        return

    print(f"开始读取文件: {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"读取到 {len(lines)} 行数据。开始生成高质量指令...")

    results = []
    # 使用 Tqdm 显示进度条
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_single_line, line) for line in lines]

        for future in tqdm(as_completed(futures), total=len(lines), desc="Generating"):
            res = future.result()
            if res:
                results.append(res)

    # 写入结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for line in results:
            f.write(line + '\n')

    print(f"处理完成！")
    print(f"原始数据: {len(lines)}")
    print(f"有效生成: {len(results)} (过滤掉了劣质或失败的数据)")
    print(f"结果已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()