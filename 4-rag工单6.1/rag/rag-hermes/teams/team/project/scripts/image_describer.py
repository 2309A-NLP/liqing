import json, base64, httpx, sys, os
from pathlib import Path

PROJECT_ROOT = Path("/mnt/d/Desktop/5-6-rag-hermes工单5.28/rag-hermes/teams/team/project")
MINERU_OUT = Path("/mnt/d/Desktop/5-6-rag-hermes工单5.28/rag-hermes/teams/MinerU_out")
OUTPUT_DIR = PROJECT_ROOT / "data" / "image_descriptions"
MIMO_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2-omni"
MIMO_API_KEY = os.environ.get("XIAOMI_API_KEY") or "tp-c1xyyzdqrq79asl216ihh43po2wkewspod12r0w3bvl26dxp"

def _get_caption(block):
    raw = block.get("image_caption", "")
    if isinstance(raw, list): return " ".join(raw)
    return raw or ""

def should_describe(block):
    """有标题或有来源注释的都描述"""
    caption = _get_caption(block)
    footnote = block.get("image_footnote", "") or ""
    if isinstance(footnote, list): footnote = " ".join(footnote)
    return bool(caption.strip()) or bool(footnote.strip())

def get_image_path(img_path):
    candidates = list(MINERU_OUT.rglob(img_path))
    return candidates[0] if candidates else None

def describe_image(img_path, block):
    actual_path = get_image_path(img_path)
    if not actual_path: return None
    with open(actual_path, "rb") as f: img_b64 = base64.b64encode(f.read()).decode()
    ext = actual_path.suffix.lower()
    mime = {"jpg":"image/jpeg","jpeg":"image/jpeg","png":"image/png","gif":"image/gif","webp":"image/webp"}.get(ext.lstrip("."),"image/jpeg")
    caption = _get_caption(block)
    prompt = f"这是一张招股说明书中的图片。请详细描述这张图片的内容，包括所有文字、数据、图表结构、层级关系等关键信息。"
    try:
        resp = httpx.post(f"{MIMO_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {MIMO_API_KEY}"},
            json={"model": MIMO_MODEL, "messages":[{
                "role":"user","content":[
                    {"type":"text","text":prompt},
                    {"type":"image_url","image_url":{"url":f"data:{mime};base64,{img_b64}"}}
            ]}], "max_tokens":2048, "temperature":0.1}, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  API失败: {e}")
        return None

def main():
    preview = "--preview" in sys.argv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_desc = []
    for sf in sorted((PROJECT_ROOT/"data"/"source_docs").glob("*_content_list.json")):
        doc_name = sf.stem.replace("_content_list","")
        print(f"\n{doc_name}")
        with open(sf) as f: blocks = json.load(f)
        for b in [x for x in blocks if x["type"]=="image"]:
            img_path = b.get("img_path","")
            page = b.get("page_idx",0)+1
            if not should_describe(b): continue
            print(f"  p{page} {_get_caption(b)[:50]}")
            desc = "(预览)" if preview else describe_image(img_path, b)
            if desc:
                all_desc.append({"source_file":doc_name,"page_no":page,
                    "chunk_type":"image_description","img_path":img_path,
                    "image_caption":_get_caption(b),"description":desc})
    out = OUTPUT_DIR/"descriptions.json"
    with open(out,"w",encoding="utf-8") as f: json.dump(all_desc,f,ensure_ascii=False,indent=2)
    tag = "预览" if preview else "完成"
    print(f"\n{tag}: {len(all_desc)} 张图")

if __name__ == "__main__":
    main()
