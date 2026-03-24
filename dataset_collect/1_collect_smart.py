import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# ================= 配置区域 =================
GITHUB_TOKEN = "YOUR_TOKEN_HERE"
OUTPUT_DIR = r"E:\about_task\HPC_Model_FT\middle_collect"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "hpc_repos_clean.csv")
TARGET_COUNT = 500
YEARS_BACK = 5
MIN_STARS = 20


# ===========================================

def fetch_repos():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    headers = {"Authorization": f"token {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    date_cutoff = (datetime.now() - timedelta(days=YEARS_BACK * 365)).strftime("%Y-%m-%d")

    # 构建多维度 Query，涵盖核心 HPC 领域
    queries = [
        f"topic:hpc language:C++ stars:>{MIN_STARS} pushed:>{date_cutoff} NOT homework",
        f"topic:cuda stars:>{MIN_STARS} pushed:>{date_cutoff} NOT tutorial",
        f"topic:mpi language:C stars:>{MIN_STARS} pushed:>{date_cutoff}",
        f"topic:openmp stars:>{MIN_STARS} pushed:>{date_cutoff}",
        f"topic:fortran topic:hpc stars:>{MIN_STARS} pushed:>{date_cutoff}",
        f"topic:simd OR topic:avx stars:>{MIN_STARS} pushed:>{date_cutoff}"
    ]

    unique_repos = {}
    print(f"策略: Star>={MIN_STARS}, Post-{date_cutoff}, 目标: {TARGET_COUNT}")

    for q in queries:
        if len(unique_repos) >= TARGET_COUNT: break
        print(f"--> 搜索: {q}")
        page = 1
        while len(unique_repos) < TARGET_COUNT:
            try:
                res = requests.get("https://api.github.com/search/repositories",
                                   params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
                                   headers=headers)
                if res.status_code != 200:
                    print(f"API 休眠 (Status {res.status_code})...")
                    time.sleep(60)
                    continue

                items = res.json().get("items", [])
                if not items: break

                for item in items:
                    # ID去重
                    if item['id'] in unique_repos: continue
                    # 描述过滤
                    desc = (item.get('description') or "").lower()
                    if any(x in desc for x in ['homework', 'assignment', 'course', 'learn', 'sample']): continue

                    unique_repos[item['id']] = {
                        "full_name": item['full_name'],
                        "clone_url": item['clone_url'],
                        "stars": item['stargazers_count'],
                        "pushed_at": item['pushed_at'],
                        "description": item.get('description', "")
                    }
                print(f"   当前收集: {len(unique_repos)}")
                page += 1
                time.sleep(1.5)
            except Exception as e:
                print(f"Error: {e}")
                break

    df = pd.DataFrame(list(unique_repos.values()))
    if not df.empty:
        df = df.sort_values(by="stars", ascending=False).head(TARGET_COUNT)
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 元数据搜集完成: {len(df)} 条 -> {OUTPUT_FILE}")
    else:
        print("❌ 未搜集到数据，请检查Token。")


if __name__ == "__main__":
    fetch_repos()