import os
import json
import random
import time
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"  # 替换你的 DeepSeek API Key
BASE_URL = "https://api.deepseek.com"  # DeepSeek API 地址
OUTPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_optimization_dataset.jsonl"
TARGET_COUNT = 800  # 目标生成数量
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 1. 定义 HPC 知识图谱 (确保覆盖面)
# 格式: {Domain: [List of specific optimization topics]}
TOPICS = {
    "OpenMP (CPU Parallelism)": [
        "False Sharing avoidance (padding or local accumulation)",
        "Loop Scheduling (static vs dynamic for unbalanced loads)",
        "Reduction clause usage to avoid race conditions",
        "SIMD Vectorization (#pragma omp simd)",
        "Thread Affinity / Pinning (OMP_PROC_BIND)",
        "Critical section minimization (using atomic instead)",
        "Loop Collapse for nested loops",
        "False sharing in array access patterns"
    ],
    "MPI (Distributed Computing)": [
        "Replacing blocking MPI_Send/Recv with Non-blocking MPI_Isend/Irecv",
        "Overlapping Communication and Computation",
        "Using MPI_Type_vector for non-contiguous data (vs manual packing)",
        "Using MPI_Collective ops (Reduce/Allreduce) instead of point-to-point loops",
        "MPI_Graph_create/Cart_create for topology optimization",
        "One-sided communication (MPI_Put/Get) optimization",
        "Reducing message latency by aggregating small messages"
    ],
    "CUDA (GPU Acceleration)": [
        "Global Memory Coalescing (Stride access)",
        "Shared Memory Tiling (Matrix Mult/Stencil)",
        "Warp Divergence avoidance",
        "Bank Conflict resolution in Shared Memory",
        "Using CUDA Streams for concurrent Kernel/Memcpy",
        "Optimizing Block/Grid dimensions (Occupancy)",
        "Using constant memory for read-only parameters",
        "Warp Shuffle instructions instead of Shared Memory reduction"
    ],
    "General C++/HPC Optimization": [
        "Loop Interchange for Cache Locality (Row-major access)",
        "Loop Unrolling (Compiler hints vs Manual)",
        "Branch Prediction optimization (likely/unlikely)",
        "Data Structure optimization (SoA vs AoS)",
        "Memory Alignment (posix_memalign / alignas)",
        "Prefetching (Software prefetch)"
    ]
}

# 2. 定义应用场景 (确保代码不枯燥，不全是矩阵乘法)
SCENARIOS = [
    "Matrix Multiplication",
    "Sparse Matrix Vector Multiplication (SpMV)",
    "1D/2D/3D Stencil Computation (Heat Equation)",
    "N-Body Particle Simulation",
    "Monte Carlo Simulation (Finance/Physics)",
    "Image Convolution/Filtering",
    "Fluid Dynamics (CFD) Solver step",
    "Molecular Dynamics force calculation",
    "Vector Dot Product / Saxpy",
    "Histogram calculation"
]

# 系统提示词：强制输出 JSON 且扮演专家
SYSTEM_PROMPT = """
You are a Principal HPC Performance Engineer and C++ Expert.
Your task is to generate a high-quality dataset for fine-tuning an LLM on code optimization.
You must output a strictly valid JSON object.
"""


def generate_one_sample(domain, topic, scenario):
    """
    构造 Prompt 并调用 API 生成单条数据
    """
    prompt = f"""
    Please generate a rigorous C++ code optimization example based on the following constraints:

    1. **Domain**: {domain}
    2. **Optimization Topic**: {topic}
    3. **Application Scenario**: {scenario}

    The output must be a single JSON object containing exactly these fields:
    - "instruction": A user request asking to optimize the code (e.g., "Optimize this MPI code to reduce latency...").
    - "input": The "Bad Code" (Slow, buggy, or naive implementation). It must be syntactically correct C++ but have the specific performance issue.
    - "output": The "Good Code" (Optimized version) followed by a concise "Analysis" section explaining *why* it is faster.

    **Requirements**:
    - The code should be self-contained enough to be understood.
    - Do NOT include generic conversational filler.
    - The "input" code must clearly demonstrate the inefficiency mentioned in the topic.
    - The "output" code must correctly apply the optimization.
    - **Response Format**: JSON only. No markdown fencing outside the JSON.
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 或者 deepseek-coder
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # 增加一点创造性，避免雷同
            response_format={"type": "json_object"}  # 强制 JSON 模式 (关键)
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"\n[Error] Generation failed: {e}")
        return None


def main():
    # ================= 修改开始：断点续传逻辑 =================
    generated_count = 0
    if os.path.exists(OUTPUT_FILE):
        print(f"📂 检测到文件 {OUTPUT_FILE} 已存在，正在统计已生成数量...")
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f_read:
                # 统计行数，每一行是一条数据
                generated_count = sum(1 for _ in f_read)
            print(f"✅ 已找到 {generated_count} 条历史数据，将从这里继续生成。")
        except Exception as e:
            print(f"⚠️ 读取文件出错，将从头开始: {e}")
            generated_count = 0
    else:
        print("🆕 创建新文件，开始生成。")
    # ================= 修改结束 =================

    print(f"🚀 开始生成 HPC 优化数据集，目标总数: {TARGET_COUNT} 条")
    print(f"💾 输出文件: {OUTPUT_FILE}")

    # 如果已经达到了目标，直接结束
    if generated_count >= TARGET_COUNT:
        print("🎉 目标数量已达成，无需继续生成！")
        return

    # 使用追加模式打开文件
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        # initial参数可以让进度条显示正确的总进度
        pbar = tqdm(total=TARGET_COUNT, initial=generated_count)

        while generated_count < TARGET_COUNT:
            # 1. 随机组合 Prompt
            domain = random.choice(list(TOPICS.keys()))
            topic = random.choice(TOPICS[domain])
            scenario = random.choice(SCENARIOS)

            # 2. 生成数据
            data = generate_one_sample(domain, topic, scenario)

            if data and "instruction" in data and "input" in data and "output" in data:
                # 3. 简单的质量检查
                if len(data["input"]) < 50 or len(data["output"]) < 50:
                    continue

                # 4. 写入文件
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()  # 确保立即保存

                generated_count += 1
                pbar.update(1)
                pbar.set_description(f"Last: {domain[:10]}...")

                time.sleep(0.5)
            else:
                pass

    print("\n✅ 数据集生成完毕！")


if __name__ == "__main__":
    main()
