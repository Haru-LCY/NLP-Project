# -*- coding: utf-8 -*-
"""
ShirohaPet API 服务
提供聊天、问答和视觉理解接口
"""

from fastapi import FastAPI, Request
from datetime import datetime
from threading import Lock
from typing import List, Optional, Tuple
import uvicorn
import requests
import json
import torch
import platform
import sys
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from Shiroha.utils import get_config
import argparse

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# 确保标准输出使用 UTF-8 编码，防止中文乱码
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


print("🖥️ 初始化 PyTorch 引擎...")
ENGINE = "torch"
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ PyTorch 引擎加载成功 (使用 CUDA 加速)")
else:
    DEVICE = "cpu"
    print("⚠️ 未检测到可用的 CUDA 设备，使用 CPU 进行推理，性能可能较差")

api = FastAPI()

adapter_path = "./models/Shiroha"
max_seq_length = 2048

DEFAULT_RAG_SOURCE = os.environ.get("RAG_SOURCE_FILE", "./rag_sources.txt")
DEFAULT_RAG_CHUNK_SIZE = int(os.environ.get("RAG_CHUNK_SIZE", "500"))
DEFAULT_RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "100"))
DEFAULT_RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
_rag_store = None
_rag_lock = Lock()


def _chunk_text(content: str, chunk_size: int, overlap: int) -> List[str]:
    """Split raw text into overlapping chunks."""
    normalized = content.replace("\r\n", "\n")
    paragraphs = [p.strip() for p in normalized.split("\n\n") if p.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            segment = paragraph[start:end].strip()
            if segment:
                chunks.append(segment)
            if end >= len(paragraph):
                break
            start = max(0, end - overlap)
    return chunks


class SimpleTextRAG:
    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        if SKLEARN_AVAILABLE and chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.matrix = self.vectorizer.fit_transform(chunks)
        else:
            self.vectorizer = None
            self.matrix = None

    def _fallback_retrieve(self, query: str, top_k: int) -> List[str]:
        # Simple token overlap when sklearn is unavailable
        query_terms = set(query.lower().split())
        scored = []
        for idx, chunk in enumerate(self.chunks):
            tokens = set(chunk.lower().split())
            overlap = len(query_terms & tokens)
            if overlap:
                scored.append((overlap, idx))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [self.chunks[idx] for _, idx in scored[:top_k]]

    def retrieve(self, query: str, top_k: int) -> List[str]:
        if not query or not self.chunks:
            return []
        if self.vectorizer is None or self.matrix is None:
            return self._fallback_retrieve(query, top_k)
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1]
        results: List[str] = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                continue
            results.append(self.chunks[idx])
        return results


def _load_rag_store(path: Optional[str]) -> Optional[SimpleTextRAG]:
    if not path or not os.path.exists(path):
        print(f"ℹ️ RAG 知识库文件 {path} 不存在，跳过加载")
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
    except Exception as exc:
        print(f"⚠️ 读取 RAG 文件失败: {exc}")
        return None
    if not raw_text.strip():
        print(f"ℹ️ RAG 文件 {path} 内容为空，跳过加载")
        return None
    chunk_size = max(128, DEFAULT_RAG_CHUNK_SIZE)
    overlap = min(DEFAULT_RAG_CHUNK_OVERLAP, chunk_size - 1)
    chunks = _chunk_text(raw_text, chunk_size=chunk_size, overlap=overlap)
    print(f"📚 RAG 知识库加载完成，共 {len(chunks)} 段")
    return SimpleTextRAG(chunks)


def _resolve_rag_path() -> str:
    rag_path = DEFAULT_RAG_SOURCE
    try:
        cfg = get_config()
        rag_path = cfg.get("rag", {}).get("source_file") or rag_path
    except Exception as exc:
        print(f"⚠️ 加载配置时无法获取 RAG 文件路径: {exc}")
    return rag_path


def _get_rag_store() -> Optional[SimpleTextRAG]:
    global _rag_store
    if _rag_store is not None:
        return _rag_store
    with _rag_lock:
        if _rag_store is not None:
            return _rag_store
        rag_path = _resolve_rag_path()
        _rag_store = _load_rag_store(rag_path)
    return _rag_store


def append_conversation_to_rag(prompt: str, reply: str) -> None:
    if not prompt and not reply:
        return
    rag_path = _resolve_rag_path()
    if not rag_path:
        return
    timestamp = get_current_time()
    entry = (
        "\n\n"
        f"[{timestamp}] USER INPUT:\n{prompt or '(empty)'}\n"
        f"[{timestamp}] ASSISTANT OUTPUT:\n{reply or '(empty)'}\n"
    )
    directory = os.path.dirname(rag_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as exc:
            print(f"⚠️ 无法创建 RAG 目录 {directory}: {exc}")
            return
    global _rag_store
    with _rag_lock:
        try:
            with open(rag_path, "a", encoding="utf-8") as rag_file:
                rag_file.write(entry)
        except Exception as exc:
            print(f"⚠️ 无法写入 RAG 文件 {rag_path}: {exc}")
            return
        _rag_store = _load_rag_store(rag_path)


def augment_prompt_with_rag(prompt: str) -> Tuple[str, List[str]]:
    store = _get_rag_store()
    if not prompt or store is None:
        return prompt, []
    top_k = max(1, DEFAULT_RAG_TOP_K)
    snippets = store.retrieve(prompt, top_k=top_k)
    if not snippets:
        return prompt, []
    context = "\n\n".join(f"[{idx + 1}] {snippet.strip()}" for idx, snippet in enumerate(snippets))
    augmented = (
        "You have access to the following reference notes pulled from a trusted knowledge file. "
        "Use them when helpful and cite only what is supported.\n"
        f"{context}\n\n"
        f"User question: {prompt}"
    )
    return augmented, snippets


def load_model_and_tokenizer():
    print(f"📂 模型加载路径: {adapter_path}")
    print(f"⚙️ 推理引擎: {ENGINE} | 计算设备: {DEVICE}")


    print("🔧 正在加载 PyTorch LoRA 模型...")

    try:
        print("🔄 正在准备基础模型和 LoRA 适配器...")
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            print(f"❌ 严重错误：未找到适配器配置文件 {adapter_config_path}")
            print("💡 请运行 download.py 以下载基础模型与 LoRA 适配器")
            exit(1)

        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        base_model_path = adapter_config.get("base_model_name_or_path")
        if not base_model_path:
            print("❌ 严重错误：适配器配置缺少 base_model_name_or_path")
            exit(1)
        if not os.path.exists(base_model_path):
            print(f"❌ 严重错误：基础模型路径不存在: {base_model_path}")
            print("💡 请确认 Qwen3-14B 模型是否已下载并与 adapter_config.json 中的路径一致")
            exit(1)

        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        device_map = "cuda"
        print("在cuda上加载模型，用nvidia-smi监控显存状态")

        print(f"📦 正在加载基础模型: {base_model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        print(f"🎯 正在应用 LoRA 适配器: {adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            device_map=device_map,
        )

        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        print("✅ LoRA 模型加载成功！")
        print(f"   📍 基础模型: {base_model_path}")
        print(f"   📍 适配器: {adapter_path}")
        print(f"   🏷️ 推理设备: {DEVICE}")
    except Exception as e:
        print(f"❌ 严重错误：无法加载 PyTorch LoRA 模型！")
        print(f"错误详情: {e}")
        print()
        print("🔍 可能的原因：")
        print("1. LoRA 文件损坏或不完整")
        print("2. 缺少必需的 PyTorch 依赖")
        print()
        print("💡 解决方案：")
        print("重新运行 download.py 确保 LoRA 文件正确下载")
        print()
        print("🚨 程序退出：应用需要 LoRA 模型才能运行")
        exit(1)

    return model, tokenizer


# 辅助函数：获取当前时间
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 辅助函数：记录请求日志
def log_request(prompt):
    print(f'📥 [{get_current_time()}] 收到用户请求: {prompt}')


# 辅助函数：记录响应日志
def log_response(response):
    print(f'📤 [{get_current_time()}] 生成最终回复: {response}')


# 辅助函数：解析请求
def parse_request(json_post_list):
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    return prompt, history


# 辅助函数：创建标准响应
def create_response(response_text, history, status=200):
    time = get_current_time()
    return {
        "response": response_text,
        "history": history,
        "status": status,
        "time": time
    }



def call_openrouter_api(config, api_key, model, messages, image_url=None, max_tokens=2048):
    """调用 OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://Shiroha-pet.local",
        "X-Title": "ShirohaPet"
    }

    # 处理图像输入 - 按照 OpenRouter 官方文档格式
    if image_url:
        # 如果有图像，将最后一个用户消息修改为包含图像
        for message in reversed(messages):
            if message['role'] == 'user':
                if isinstance(message['content'], str):
                    # 将字符串转换为官方文档要求的数组格式
                    message['content'] = [
                        {"type": "text", "text": message['content']},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                elif isinstance(message['content'], list):
                    # 如果已经是数组格式，直接添加图像
                    message['content'].append({"type": "image_url", "image_url": {"url": image_url}})
                break

    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }

    # 从配置中获取 OpenRouter 地址，如果不存在则使用默认值
    endpoint_url = config.get('endpoints', {}).get('openrouter', "https://openrouter.ai/api/v1/chat/completions")
    response = requests.post(endpoint_url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()


@api.post("/chat")
async def create_chat(request: Request):
    print("[DEBUG] Create chat is called")
    json_post_list = await request.json()
    prompt, history = parse_request(json_post_list)
    prompt = prompt or ""
    log_request(prompt)
    history = history or []
    history_with_user = history + [{'role': 'user', 'content': prompt}]

    augmented_prompt, rag_chunks = augment_prompt_with_rag(prompt)
    if rag_chunks:
        print(f"📚 RAG 命中 {len(rag_chunks)} 段上下文参与回答")

    model_history = history + [{'role': 'user', 'content': augmented_prompt}]

    print(f"💬 使用 {ENGINE.upper()} 引擎进行推理...")
    print(f"📊 最大生成长度: {json_post_list.get('max_new_tokens', 2048)} tokens")
    
    text = tokenizer.apply_chat_template(
        model_history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    print("✅ 聊天模板应用完成")

    max_new_tokens = int(json_post_list.get('max_new_tokens', 2048))
    max_new_tokens = max(1, max_new_tokens)
    temperature = float(json_post_list.get('temperature', 0.7))
    top_p = float(json_post_list.get('top_p', 0.9))
    top_p = max(0.01, min(top_p, 1.0))

    # 推理
    print("🤖 正在生成回复...")
    encoded = tokenizer(
        text,
        return_tensors="pt",
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": max(0.01, temperature),
        "top_p": top_p,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            **generation_kwargs,
        )
    generated_tokens = generated[0, encoded["input_ids"].shape[-1]:]
    reply = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"✅ 回复生成完成 (长度: {len(reply)} 字符)")

    history_with_user.append({"role": "assistant", "content": reply})
    append_conversation_to_rag(prompt, reply)

    log_response(reply)
    return create_response(reply, history_with_user)

@api.post("/qwen3")
async def create_qwen3_chat(request: Request):
    """
    Robust qwen3 endpoint:
    - If enable_qwen3 is False: return the latest assistant reply from history (if any).
      Try multiple fallbacks: json field 'reply', 'message', 'content', or last assistant in history.
    - Add lots of debug printing so we can see what the client actually sent.
    """
    config = get_config()
    json_post_list = await request.json()

    # Debug: dump what client sent (important to inspect)
    try:
        print("---- /qwen3 REQUEST BODY DUMP ----")
        print(json.dumps(json_post_list, ensure_ascii=False, indent=2))
        print("---- end request dump ----")
    except Exception:
        print("(/qwen3) cannot pretty-print request body")

    # Parse request with your parser if available
    try:
        prompt, history = parse_request(json_post_list)
    except Exception as e:
        print(f"/qwen3 parse_request failed: {e}")
        # best effort: try to extract history from common keys
        history = json_post_list.get("history") or json_post_list.get("messages") or []
        prompt = json_post_list.get("prompt") or json_post_list.get("text") or ""

    # Debug print of parsed history
    print("/qwen3 parsed prompt:", repr(prompt))
    print(f"/qwen3 parsed history length: {len(history)}")
    for i, msg in enumerate(history[-8:], start=max(0, len(history)-8)):
        print(f"  history[{i}] = ({msg.get('role')}) {repr(msg.get('content'))}")

    if prompt != "":
        history = history + [{'role': 'assistant', 'content': prompt}]

    # If Qwen disabled -> short-circuit and return a safe fallback
    if not config.get("enable_qwen3", True):
        print("🚫 Qwen3 已禁用：准备返回 LoRA 的最后有效回复（执行多种 fallback）")

        # 1) Try explicit 'reply' field from client body
        reply_candidates = []
        if isinstance(json_post_list, dict):
            for key in ("reply", "message", "content", "text"):
                val = json_post_list.get(key)
                if isinstance(val, str) and val.strip():
                    reply_candidates.append(val.strip())

        # 2) Last non-empty assistant in parsed history
        for msg in reversed(history):
            if msg.get("role") == "assistant":
                c = (msg.get("content") or "").strip()
                if c:
                    reply_candidates.append(c)
                    break

        # 3) Last non-empty user message as ultimate fallback
        if not reply_candidates:
            for msg in reversed(history):
                if msg.get("role") == "user":
                    c = (msg.get("content") or "").strip()
                    if c:
                        reply_candidates.append(c)
                        break

        # 4) final fallback -> empty string (do not use "……")
        final_reply = reply_candidates[0] if reply_candidates else ""

        print(f"/qwen3 final_reply chosen (len={len(final_reply)}): {repr(final_reply)}")

        # Return in the same response shape your front-end expects (use create_response)
        return create_response(final_reply, history)

    # If Qwen3 enabled -> regular flow (kept from your original code)
    api_key = config.get('openrouter_api_key', '')
    endpoint_url = config.get('server', {}).get('qwen3', '')

    if "openrouter.ai" in endpoint_url and api_key.strip():
        print("🌐 使用 OpenRouter 调用 Qwen3...")
        try:
            result = call_openrouter_api(
                config,
                api_key,
                "qwen/qwen3-235b-a22b",
                history,
                max_tokens=4096
            )
            final_response = result['choices'][0]['message']['content']
        except Exception as e:
            error_msg = f"OpenRouter API 错误: {str(e)}"
            log_response(error_msg)
            return create_response(error_msg, history, status=500)
    else:
        print(f"🏠 使用本地端点 ({endpoint_url}) 调用 Qwen3...")
        try:
            response = requests.post(
                f"{endpoint_url}/api/chat",
                json={
                    "model": "qwen3:14b",
                    "messages": history,
                    "stream": False,
                    "options": {"keep_alive": -1}
                },
                timeout=60
            )
            response.raise_for_status()
            final_response = response.json()['message']['content']
        except Exception as e:
            print(f"❌ 本地 API 调用失败: {e}")
            raise

    history.append({'role': 'assistant', 'content': final_response})
    log_response(final_response)
    return create_response(final_response, history)

@api.post("/qwenvl")
async def create_qwenvl_chat(request: Request):
    json_post_list = await request.json()
    prompt, history = parse_request(json_post_list)
    log_request(prompt)

    if "image" in json_post_list:
        image_url = json_post_list.get('image')
        print(f"🖼️ 检测到图像输入: {image_url[:100]}...")
        history = history + [{'role': 'user', 'content': prompt, 'images': [image_url]}]
    else:
        print("📝 纯文本模式（无图像输入）")
        history = history + [{'role': 'user', 'content': prompt}]

    config = get_config()
    api_key = config.get('openrouter_api_key', '')
    endpoint_url = config.get('server', {}).get('qwenvl', '')
    image_url_for_api = json_post_list.get('image') if "image" in json_post_list else None

    # 仅当 endpoint 指向 openrouter 且 API key 存在时，才使用 OpenRouter
    if "openrouter.ai" in endpoint_url and api_key.strip():
        print(f"🌐 检测到 qwenvl endpoint 指向 OpenRouter，使用 API Key 进行调用...")
        try:
            result = call_openrouter_api(
                config,
                api_key,
                "qwen/qwen-2.5-vl-7b-instruct",
                history,
                image_url=image_url_for_api
            )
            final_response = result['choices'][0]['message']['content']
            print("✅ OpenRouter 视觉 API 调用成功")
        except Exception as e:
            error_msg = f"OpenRouter API 错误: {str(e)}"
            print(f"❌ {error_msg}")
            log_response(error_msg)
            return create_response(error_msg, history, status=500)
    else:
        # 使用本地端点 (Ollama 或其他)
        print(f"🏠 使用本地端点 ({endpoint_url}) 进行调用...")
        try:
            response = requests.post(
                f"{endpoint_url}/api/chat",
                json={"model": "qwen2.5vl:7b", "messages": history,
                      "stream": False, "options": {"keep_alive": -1}},
            )
            response.raise_for_status()
            final_response = response.json()['message']['content']
            print("✅ 本地视觉 API 调用成功")
        except requests.exceptions.RequestException as e:
            print(f"❌ 调用本地视觉 API 时出错: {e}")
            raise

    history = history + [{'role': 'assistant', 'content': final_response}]
    log_response(final_response)
    return create_response(final_response, history)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=28565)
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 ShirohaPet API 服务启动中...")
    print("=" * 60)
    
    model, tokenizer = load_model_and_tokenizer()
    
    print("=" * 60)
    print("✅ 模型加载完成，启动 FastAPI 服务器...")
    print(f"🌐 服务地址: http://0.0.0.0:{args.port}")
    print("=" * 60)

    uvicorn.run(api, host='0.0.0.0', port=args.port, workers=1)