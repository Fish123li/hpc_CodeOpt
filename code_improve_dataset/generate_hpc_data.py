import os
import json
import random
import time
import hashlib
import re
from tqdm import tqdm
from openai import OpenAI

# ================= 配置区域 =================
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"
BASE_URL = "https://api.deepseek.com"  # DeepSeek API地址
OUTPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_optimization_dataset_v2.jsonl"
TARGET_COUNT = 2000
MODEL_NAME = "deepseek-chat"

# 质量控制参数
MIN_CODE_LENGTH = 100  # 字符数，防止代码太短
MAX_RETRIES = 3  # API调用重试次数

# ================= 场景分类矩阵 (用于保证丰富性) =================
DOMAINS = [
    "Computational Fluid Dynamics (CFD)", "Molecular Dynamics (MD)",
    "Weather Prediction", "Quantum Chemistry", "Astrophysics N-body",
    "Seismic Processing", "Financial Monte Carlo", "Deep Learning Kernels",
    "Image Processing", "Graph Algorithms (BFS/SSSP)"
]

LANGUAGES = [
    "C", "C++", "Fortran", "CUDA C++", "Python (w/ Numba/CuPy)"
]

PARADIGMS = [
    "MPI (Message Passing Interface)", "OpenMP (Multi-threading)",
    "OpenACC (Offloading)", "CUDA (GPU Programming)", "SIMD Intrinsics (AVX-512/NEON)",
    "Hybrid MPI + OpenMP", "Hybrid MPI + CUDA"
]

OPTIMIZATION_TYPES = [
    "Loop Tiling/Blocking for Cache Locality",
    "Vectorization/SIMD alignment",
    "Memory Coalescing (GPU)",
    "Avoiding Warp Divergence (GPU)",
    "Non-blocking Communication (MPI Isend/Irecv)",
    "False Sharing Prevention",
    "Loop Unrolling & Pipelining",
    "AoS to SoA conversion",
    "Reducing Branch Misprediction"
]

# ================= 客户端初始化 =================
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_random_scenario():
    """随机组合生成一个具体的HPC场景"""
    return {
        "domain": random.choice(DOMAINS),
        "lang": random.choice(LANGUAGES),
        "paradigm": random.choice(PARADIGMS),
        "opt_type": random.choice(OPTIMIZATION_TYPES)
    }


def construct_prompt(scenario):
    """构建Prompt，强制输出JSON格式"""

    prompt = f"""
你是一位拥有20年经验的高性能计算(HPC)专家。请生成一个高质量的代码优化案例。

**场景要求**:
- 应用领域: {scenario['domain']}
- 编程语言: {scenario['lang']}
- 并行/编程模型: {scenario['paradigm']}
- 核心优化点: {scenario['opt_type']}
- 代码难度: 中级到高级 (避免简单的Hello World或基础教程代码)

**输出格式**:
必须严格输出且仅输出一个JSON对象，不要包含markdown代码块标记（如 ```json），JSON包含以下四个字段：
1. "instruction": 描述用户的需求。措辞要多样化，例如“请优化这段代码以减少缓存未命中”、“请提供一个更高效的实现，减少这段代码的计算复杂度。”、“如何改进这段Fortran代码的内存访问模式，以提升HPC集群上的性能？”、“找出这段代码的性能瓶颈，并提供至少两种优化方案。”等。
2. "input": 优化前的原始代码（性能较低，或有明显缺陷）。代码应当完整或包含核心计算核。
3. "output": 优化后的代码（应用了{scenario['opt_type']}）。
4. "suggestion": 详细的优化建议和原理分析（解释为什么这样做能提升性能）。

**质量控制**:
- input和output的代码必须逻辑对应。
- input代码长度不能太短，要有实际计算逻辑。
- 确保代码符合HPC领域的最佳实践。
    """
    return prompt


def validate_data(entry, existing_hashes):
    """数据校验与去重"""
    try:
        # 1. 字段完整性检查
        required_keys = ["instruction", "input", "output", "suggestion"]
        if not all(key in entry for key in required_keys):
            return False, "Missing keys"

        # 2. 长度检查
        if len(entry["input"]) < MIN_CODE_LENGTH:
            return False, "Input code too short"

        # 3. 去重 (基于input代码的哈希)
        input_hash = hashlib.md5(entry["input"].strip().encode()).hexdigest()
        if input_hash in existing_hashes:
            return False, "Duplicate input"

        # 4. 简单内容检查 (防止模型输出空代码)
        if entry["input"] == entry["output"]:
            return False, "No changes made in output"

        return True, input_hash
    except Exception as e:
        return False, str(e)


def extract_json(content):
    """尝试从模型返回的文本中提取JSON"""
    try:
        # 尝试直接解析
        return json.loads(content)
    except json.JSONDecodeError:
        # 尝试查找首尾的大括号
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
    return None


def main():
    # 1. 初始化：读取已存在的数据以支持断点续传
    existing_hashes = set()
    current_count = 0

    if os.path.exists(OUTPUT_FILE):
        print(f"检测到现有文件 {OUTPUT_FILE}，正在加载以进行去重...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "input" in data:
                        h = hashlib.md5(data["input"].strip().encode()).hexdigest()
                        existing_hashes.add(h)
                        current_count += 1
                except:
                    continue
        print(f"已加载 {current_count} 条数据。目标: {TARGET_COUNT}")
    else:
        print("未检测到现有文件，将创建新文件。")

    # 2. 主循环
    pbar = tqdm(total=TARGET_COUNT, initial=current_count, desc="Generating HPC Data")

    while current_count < TARGET_COUNT:
        scenario = get_random_scenario()
        prompt = construct_prompt(scenario)

        success = False
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "你是一个严谨的数据集合成助手。只输出JSON格式。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # 稍微增加随机性以丰富措辞
                    max_tokens=2500  # 预留足够的生成长度
                )

                content = response.choices[0].message.content
                json_data = extract_json(content)

                if json_data:
                    is_valid, msg = validate_data(json_data, existing_hashes)
                    if is_v:
                        # 写入文件 (append mode)
                        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

                        existing_hashes.add(msg)  # msg is the hash
                        current_count += 1
                        pbar.update(1)
                        success = True
                        break  # 成功则跳出重试循环
                    else:
                        # print(f"数据校验失败: {msg}") # 调试时可打开
                        pass
                else:
                    # print("JSON解析失败")
                    pass

            except Exception as e:
                print(f"\nAPI Error: {e}, Retrying...")
                time.sleep(2)

        if not success:
            # 如果多次重试失败，稍微暂停一下，避免请求过于频繁
            time.sleep(1)

    pbar.close()
    print(f"完成！共生成 {current_count} 条数据，保存在 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()