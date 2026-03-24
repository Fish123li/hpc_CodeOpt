import json
import os
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================
# 替换为你的 DeepSeek API Key
API_KEY = "sk-4459681d5d46443393b0b6b50675b7dc"
BASE_URL = "https://api.deepseek.com"

# 输入和输出文件路径
INPUT_FILE = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_final.jsonl"
OUTPUT_FILE =r"E:\about_task\HPC_Model_FT\middle_collect\hpc_dataset_final_clean.jsonl"

# 线程数 (根据你的 API Rate Limit 调整，DeepSeek通常支持较高的并发)
MAX_WORKERS = 8
# ===========================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_system_prompt():
    """
    构造 System Prompt，核心逻辑：
    1. 识别代码语言。
    2. 加上 markdown 标记 (如 ```cpp)。
    3. 保留非代码的解释性文本。
    4. 不要改变代码原始内容。
    """
    return """
你是一个专业的数据集清洗助手。你的任务是格式化代码生成模型的输出数据。
输入的文本包含代码，但可能缺少 Markdown 标记 (```)，或者混合了文本和裸代码。

请遵循以下规则处理输入：
1. **自动识别语言**：分析代码属于什么语言（Python, C++, Java, Bash, Fortran, Go 等）。
2. **添加标记**：使用标准的 Markdown 代码块包裹代码部分，并在 ``` 后加上正确的语言标识符。
3. **混合内容处理**：如果输入包含“好的，代码如下：”等解释性文字，请保留文字在代码块外部。
4. **幂等性**：如果输入已经是正确的 Markdown 格式，请原样返回，不要重复包裹。
5. **严禁修改**：除了添加 Markdown 标记外，绝对不要修改代码的内容、逻辑或缩进。

只返回处理后的完整字符串，不要包含其他无关的寒暄。
"""


def process_single_output(text):
    """调用 DeepSeek API 处理单条 output"""
    if not text or not isinstance(text, str):
        return text

    # 简单预判：如果已经包含 ```，可能不需要处理，节省 token
    # 但为了保险（比如语言标记错误），你也可以去掉这个判断强制走一遍
    if "```" in text and len(text) > 10:
        return text

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": generate_system_prompt()},
                {"role": "user", "content": text}
            ],
            temperature=0.1,  # 低温度保证输出稳定，不做发散生成
            max_tokens=4096  # 根据你的代码长度调整
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing text snippet: {e}")
        return text  # 出错时返回原始内容，避免数据丢失


def process_dataset():
    # 1. 读取数据
    data = []
    is_jsonl = False

    print(f"正在读取 {INPUT_FILE} ...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
        else:
            is_jsonl = True
            for line in f:
                if line.strip():
                    data.append(json.loads(line))

    print(f"共加载 {len(data)} 条数据，开始处理...")

    # 2. 并发处理
    processed_data = [None] * len(data)  # 预分配位置保证顺序

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 创建任务映射
        future_to_idx = {
            executor.submit(process_single_output, item.get('output', '')): i
            for i, item in enumerate(data)
        }

        # 使用 tqdm 显示进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(data)):
            idx = future_to_idx[future]
            try:
                new_output = future.result()
                # 复制原始数据并更新 output
                item = data[idx].copy()
                item['output'] = new_output
                processed_data[idx] = item
            except Exception as exc:
                print(f"Data index {idx} generated an exception: {exc}")
                processed_data[idx] = data[idx]  # 出错保留原样

    # 3. 保存结果
    print(f"处理完成，正在保存到 {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        if is_jsonl:
            for item in processed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        else:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

    print("完成！")


if __name__ == "__main__":
    process_dataset()