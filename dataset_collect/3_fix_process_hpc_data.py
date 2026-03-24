import os
import re
import json
import hashlib

# ================= 配置区域 =================
SOURCE_DIR = r"E:\about_task\HPC_Model_FT\middle_collect\clone_repository"  # 你的400个仓库所在的根目录
OUTPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_raw_dataset.jsonl"
MIN_LINES = 10  # 代码过短通常没有训练价值
MAX_LINES = 500  # 代码过长可能截断
MAX_Total_SAMPLES = 15000  # 预留一些空间清洗

# HPC 关键词（必须包含至少一个才能入选）
HPC_KEYWORDS = [
    "__global__", "__device__", "cudaMalloc", "cudaMemcpy",  # CUDA
    "MPI_Init", "MPI_Comm_rank", "MPI_Send",  # MPI
    "#pragma omp", "omp_get_thread_num",  # OpenMP
    "#pragma acc",  # OpenACC
    "hipMalloc", "hipLaunchKernel"  # HIP
]

# 垃圾代码特征（只要包含这些，直接丢弃）
BAD_SIGNALS = [
    "TODO your code here",
    "TODO:",
    "FIXME",
    "YOUR CODE GOES HERE",
    "return;",  # 空函数体常见特征
    "return 0;"  # 只有返回的空main
]

# 允许的文件后缀
ALLOWED_EXTS = {'.cu', '.cuh', '.cpp', '.c', '.h', '.hpp', '.f90', '.F90'}


# ===========================================

def calculate_hash(content):
    """计算内容哈希用于去重"""
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def is_hpc_code(content):
    """检查是否包含 HPC 关键词"""
    return any(kw in content for kw in HPC_KEYWORDS)


def is_garbage(content):
    """检查是否是作业填空题或垃圾代码"""
    # 1. 检查是否有显式的 TODO
    for signal in BAD_SIGNALS:
        if signal.lower() in content.lower():
            # 如果有 TODO，且代码行数很少，极大概率是空函数
            if len(content.split('\n')) < 20:
                return True

    # 2. 检查大括号内是否为空 (简易正则)
    # 匹配类似 void foo() { } 的结构
    if re.search(r'\{\s*\}', content):
        return True

    return False


def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 1. 长度过滤
        lines = content.split('\n')
        if len(lines) < MIN_LINES or len(lines) > MAX_LINES:
            return None

        # 2. HPC 相关性过滤
        if not is_hpc_code(content):
            return None

        # 3. 质量过滤 (这是关键，过滤掉你之前遇到的那种空作业)
        if is_garbage(content):
            return None

        return content
    except Exception as e:
        return None


def main():
    unique_hashes = set()
    samples_count = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for root, dirs, files in os.walk(SOURCE_DIR):

            # 过滤掉 .git, build 等无关目录
            dirs[:] = [d for d in dirs if d not in ['.git', 'build', 'bin', 'Debug', 'Release']]

            for file in files:
                ext = os.path.splitext(file)[1]
                if ext not in ALLOWED_EXTS:
                    continue

                file_path = os.path.join(root, file)
                content = process_file(file_path)

                if content:
                    content_hash = calculate_hash(content)
                    if content_hash in unique_hashes:
                        continue  # 跳过重复文件

                    unique_hashes.add(content_hash)

                    # 构建基础数据条目
                    data_entry = {
                        "file_path": file_path,
                        "language": ext[1:],  # cu, cpp etc.
                        "code": content
                    }

                    out_f.write(json.dumps(data_entry) + '\n')
                    samples_count += 1

                    if samples_count % 100 == 0:
                        print(f"已收集 {samples_count} 条数据...")

                    if samples_count >= MAX_Total_SAMPLES:
                        break
            if samples_count >= MAX_Total_SAMPLES:
                break

    print(f"处理完成！共提取 {samples_count} 个有效代码文件。结果保存在 {OUTPUT_FILE}")


if __name__ == "__main__":
    main()