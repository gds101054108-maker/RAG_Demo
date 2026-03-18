"""
测试 PaddleOCR API 返回格式
"""

import requests
import base64
import json
from pathlib import Path
import glob

API_URL = "https://feubu8f2jcses3d8.aistudio-app.com/layout-parsing"
TOKEN = "5e362c25a7b46a97605fd3e7d113cde2672992af"

# 查找测试 PDF 文件
pdf_files = list(Path(r"D:\openclaw_demo\RAG_demo\data\documents").glob("*.pdf"))
if not pdf_files:
    print("未找到 PDF 文件")
    exit(1)

# 选择较小的文件测试
test_pdf = min(pdf_files, key=lambda p: p.stat().st_size)
print(f"测试文件: {test_pdf.name}")
print(f"文件大小: {test_pdf.stat().st_size / 1024:.2f} KB")

# 读取文件
with open(test_pdf, 'rb') as f:
    file_data = f.read()
file_base64 = base64.b64encode(file_data).decode('utf-8')

# 构建请求
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

payload = {
    "file": file_base64,
    "fileName": test_pdf.name
}

print("\n正在调用 API...")
response = requests.post(API_URL, headers=headers, json=payload, timeout=300)

print(f"状态码: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    
    # 保存完整结果到文件
    with open(r"D:\openclaw_demo\RAG_demo\paddleocr_response.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print("\n完整结果已保存到 paddleocr_response.json")
    
    # 分析结构
    print("\n=== 结果结构 ===")
    print(f"顶层键: {list(result.keys())}")
    
    if "result" in result:
        res = result["result"]
        print(f"result 键: {list(res.keys())}")
        
        if "layoutParsingResults" in res:
            layouts = res["layoutParsingResults"]
            print(f"layoutParsingResults 数量: {len(layouts)}")
            
            all_texts = []
            
            for i, layout in enumerate(layouts):
                print(f"\n--- Layout {i+1} ---")
                print(f"键: {list(layout.keys())}")
                
                # 检查各种可能的文本字段
                for key in ["markdown", "text", "content", "html"]:
                    if key in layout:
                        val = layout[key]
                        if isinstance(val, str) and len(val) > 10:
                            print(f"\n=== {key} (前 300 字符) ===")
                            print(val[:300])
                            all_texts.append(val)
                        elif isinstance(val, dict):
                            print(f"{key} 是字典，键: {list(val.keys())}")
                        elif isinstance(val, list):
                            print(f"{key} 是列表，长度: {len(val)}")
                
                # 检查 prunedResult
                if "prunedResult" in layout:
                    pruned = layout["prunedResult"]
                    if isinstance(pruned, dict):
                        print(f"\nprunedResult 键: {list(pruned.keys())}")
                        
                        # 检查 boxes 或 blocks
                        for block_key in ["boxes", "blocks", "layout_blocks", "words"]:
                            if block_key in pruned:
                                blocks = pruned[block_key]
                                if isinstance(blocks, list):
                                    print(f"\n{block_key} 数量: {len(blocks)}")
                                    if blocks:
                                        print(f"第一个 block 键: {list(blocks[0].keys()) if isinstance(blocks[0], dict) else type(blocks[0])}")
                                        
                                        # 提取文本
                                        block_texts = []
                                        for block in blocks:
                                            if isinstance(block, dict):
                                                for text_key in ["text", "content", "ocr_text", "rec_text"]:
                                                    if text_key in block and block[text_key]:
                                                        block_texts.append(str(block[text_key]))
                                        
                                        if block_texts:
                                            print(f"\n从 {block_key} 提取的文本片段数量: {len(block_texts)}")
                                            print("前 5 个片段:")
                                            for t in block_texts[:5]:
                                                print(f"  - {t[:50]}..." if len(t) > 50 else f"  - {t}")
                                            all_texts.extend(block_texts)
            
            print(f"\n=== 总共提取文本片段: {len(all_texts)} ===")
            combined = "\n\n".join(all_texts)
            print(f"合并后总字符数: {len(combined)}")
            
else:
    print(f"错误: {response.text}")