import pandas as pd
import os
from pathlib import Path
from git import Repo
from tqdm import tqdm  # 进度条库
from colorama import init, Fore, Style  # 颜色库

# 初始化颜色库 (适配 Windows 终端)
init(autoreset=True)

# ================= 配置 =================
CSV_PATH = r"E:\about_task\HPC_Model_FT\middle_collect\hpc_repos_clean.csv"
TARGET_DIR = r"E:\about_task\HPC_Model_FT\middle_collect\clone_repository"


# =======================================

def main():
    # 1. 检查 CSV
    if not os.path.exists(CSV_PATH):
        print(f"{Fore.RED}❌ 错误: 找不到文件 {CSV_PATH}，请先运行 Step 1。")
        return

    # 2. 读取数据
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"{Fore.RED}❌ 读取 CSV 失败: {e}")
        return

    tasks = list(zip(df['full_name'], df['clone_url']))
    total_tasks = len(tasks)

    print(f"{Fore.CYAN}🚀 准备下载 {total_tasks} 个 HPC 仓库到: {TARGET_DIR}\n")

    # 统计计数器
    stats = {
        "success": 0,
        "skipped": 0,
        "failed": 0
    }

    # 3. 进度条循环
    # ncols=100 限制进度条宽度，防止换行
    with tqdm(total=total_tasks, unit="repo", ncols=100) as pbar:
        for full_name, url in tasks:
            # 动态设置进度条右侧描述
            pbar.set_description(f"Processing {full_name[:20]}...")

            # 构建本地路径 E:\...\owner\repo
            repo_path = Path(TARGET_DIR) / full_name.replace("/", os.sep)

            try:
                # --- 情况 A: 断点续传 (已存在) ---
                if (repo_path / ".git").exists():
                    # 使用 tqdm.write 而不是 print，防止打乱进度条
                    tqdm.write(f"{Fore.YELLOW}[SKIP]    {full_name} (已存在)")
                    stats["skipped"] += 1

                # --- 情况 B: 开始下载 ---
                else:
                    # depth=1 浅克隆，节省时间
                    Repo.clone_from(url, repo_path, depth=1)
                    tqdm.write(f"{Fore.GREEN}[SUCCESS] {full_name}")
                    stats["success"] += 1

            except Exception as e:
                # --- 情况 C: 下载失败 ---
                # 获取简短的错误信息
                error_msg = str(e).split('\n')[0][:50]
                tqdm.write(f"{Fore.RED}[FAILED]  {full_name} -> {error_msg}...")
                stats["failed"] += 1

                # 失败时尝试清理空文件夹，以免下次误判为已存在
                if repo_path.exists() and not any(repo_path.iterdir()):
                    try:
                        os.rmdir(repo_path)
                    except:
                        pass

            # 更新进度条
            pbar.update(1)

    # 4. 最终总结
    print("\n" + "=" * 40)
    print(f"{Fore.CYAN}🎉 下载任务结束")
    print("=" * 40)
    print(f"{Fore.GREEN}✅ 成功下载: {stats['success']}")
    print(f"{Fore.YELLOW}⏭️  跳过重复: {stats['skipped']}")
    print(f"{Fore.RED}❌ 下载失败: {stats['failed']}")
    print("=" * 40)


if __name__ == "__main__":
    main()