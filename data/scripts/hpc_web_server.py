# -*- coding: utf-8 -*-
import torch
import uvicorn
import re
import os
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ================= 配置区域 =================
MODEL_PATH = "/data/hpc_ft/output/hpc_coder_v3_epoch1_merged"
PORT = 25004
# ===========================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("HPC-Web")

app = FastAPI(title="HPC Intelligent System", version="4.0 Strict")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

os.makedirs("templates", exist_ok=True)


class OptimizationRequest(BaseModel):
    code: str
    mode: str
    selected_suggestions: Optional[List[str]] = []


class OptimizationResponse(BaseModel):
    original_code: str
    optimized_code: Optional[str] = None
    suggestions: List[Dict[str, str]] = []
    report: str
    domain: str
    execution_time: float


class HPCAgent:
    def __init__(self):
        logger.info(f"Loading Model: {MODEL_PATH} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model Loaded!")

    def detect_domain(self, code: str) -> str:
        code_lower = code.lower()
        if "mpi" in code_lower: return "MPI Parallel"
        if "cuda" in code_lower: return "CUDA GPU"
        if "omp" in code_lower: return "OpenMP"
        if "immintrin" in code_lower: return "SIMD/AVX"
        if "program" in code_lower: return "Fortran"
        return "General HPC"

    def _extract_suggestions_strict(self, text: str) -> List[Dict[str, str]]:
        """
        强校验提取：只提取 <SUG> 标签内的内容，忽略所有标签外的废话和代码
        """
        suggestions = []
        # 正则匹配 <SUG>...<TITLE>...</TITLE>...<DESC>...</DESC>...</SUG>
        # 使用 DOTALL 模式 (.) 匹配换行符
        pattern = r'<SUG>.*?<TITLE>(.*?)</TITLE>.*?<DESC>(.*?)</DESC>.*?</SUG>'
        matches = re.findall(pattern, text, re.DOTALL)

        for title, desc in matches:
            t = title.strip()
            d = desc.strip()
            # 再次清洗：如果标题里依然混入了换行符或其他标签，进行清理
            if t and d:
                suggestions.append({"title": t, "desc": d})

        return suggestions

    def _parse_optimize_response(self, text: str) -> dict:
        """
        强校验提取：只提取 <CODE> 和 <REPORT> 标签
        """
        result = {"optimized_code": None, "clean_report": ""}

        # 1. 提取代码
        code_match = re.search(r'<CODE>(.*?)</CODE>', text, re.DOTALL)
        if code_match:
            raw_code = code_match.group(1).strip()
            # 去除可能存在的 markdown 包裹
            raw_code = re.sub(r'^```[a-zA-Z]*\n', '', raw_code)
            raw_code = re.sub(r'\n```$', '', raw_code)
            result["optimized_code"] = raw_code.strip()
        else:
            # 兜底：如果没有检测到标签，尝试找 Markdown
            code_match_md = re.search(r'```(?:c|cpp|cuda|fortran)?\s*\n(.*?)```', text, re.DOTALL)
            if code_match_md:
                result["optimized_code"] = code_match_md.group(1).strip()
            else:
                result["optimized_code"] = "// Error: Code generation failed or format invalid."

        # 2. 提取报告
        report_match = re.search(r'<REPORT>(.*?)</REPORT>', text, re.DOTALL)
        if report_match:
            result["clean_report"] = report_match.group(1).strip()
        else:
            # 如果没找到报告标签，取代码之后的所有内容
            if result["optimized_code"] and result["optimized_code"] in text:
                result["clean_report"] = text.split("</CODE>")[-1].strip()
            else:
                result["clean_report"] = "暂无分析报告"

        return result

    def generate(self, code: str, mode: str, selected_suggestions: List[str] = None) -> dict:
        input_ids = None
        base_instruction = ""

        # === 模式 A: 纯建议分析 ===
        if mode == "analyze":
            base_instruction = (
                "作为HPC专家，请分析以下代码的性能瓶颈。\n"
                "【任务】仅列出优化建议，不要生成代码，不要写总结。\n"
                "【严格输出格式】请务必为每一条建议使用 XML 标签包裹，格式如下：\n"
                "<SUG>\n"
                "  <TITLE>优化策略名称</TITLE>\n"
                "  <DESC>策略的详细原理说明</DESC>\n"
                "</SUG>\n"
                "<SUG>\n"
                "  ...\n"
                "</SUG>\n"
                "请忽略任何与格式无关的内容。"
            )

        # === 模式 B: 代码生成 (自动 或 选中) ===
        else:
            strategies = ""
            if mode == "optimize_auto":
                strategies = "所有适用的最佳HPC优化策略"
            else:
                strategies = "、".join(selected_suggestions)

            base_instruction = (
                f"作为HPC专家，请根据以下指定的策略对代码进行优化：[{strategies}]。\n"
                "【严格输出格式】\n"
                "1. 请将优化后的完整代码放在 <CODE> 和 </CODE> 标签之间。\n"
                "2. 请将优化原理分析报告放在 <REPORT> 和 </REPORT> 标签之间。\n"
                "3. 不要输出标签之外的任何内容。\n"
                "示例：\n"
                "<CODE>\n...\n</CODE>\n"
                "<REPORT>\n...\n</REPORT>"
            )

        final_content = f"{base_instruction}\n\nInput Code:\n{code}"

        messages = [{"role": "user", "content": final_content}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=3000,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][len(input_ids[0]):]
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 根据模式调用不同的解析器
        if mode == "analyze":
            suggestions = self._extract_suggestions_strict(response_text)
            if not suggestions:
                # 极少数情况模型没听话，返回一个默认提示
                suggestions = [{"title": "解析失败", "desc": "模型未按规定格式输出建议，请重试。"}]
            return {"suggestions": suggestions, "optimized_code": None, "clean_report": ""}
        else:
            return self._parse_optimize_response(response_text)


agent = None


@app.on_event("startup")
async def startup_event():
    global agent
    agent = HPCAgent()


@app.get("/")
async def read_root():
    return FileResponse('templates/index.html')


@app.get("/api/status")
async def check_status():
    return {"status": "connected" if agent else "loading"}


@app.post("/api/process", response_model=OptimizationResponse)
async def process_code(req: OptimizationRequest):
    if not agent: raise HTTPException(status_code=503, detail="Model loading")
    start_time = time.time()
    domain = agent.detect_domain(req.code)

    result = agent.generate(req.code, req.mode, req.selected_suggestions)

    return OptimizationResponse(
        original_code=req.code,
        optimized_code=result.get("optimized_code"),
        suggestions=result.get("suggestions", []),
        report=result.get("clean_report"),
        domain=domain,
        execution_time=time.time() - start_time
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)