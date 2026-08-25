#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMind-3 upstream JSONL 스키마에 맞는 한국어 학습 데이터셋을 생성합니다.

검증 기준:
  dataset/lm_dataset.py
  trainer/train_agent.py

설치:
  uv pip install -U datasets huggingface_hub tqdm

실행 예:
  uv run python build_minimind_ko.py --preset tiny
  uv run python build_minimind_ko.py --preset mini
  uv run python build_minimind_ko.py --preset full --output ./dataset

출력 파일:
  pretrain_t2t_mini.jsonl
  pretrain_t2t.jsonl
  sft_t2t_mini.jsonl
  sft_t2t.jsonl
  dpo.jsonl
  rlaif.jsonl
  agent_rl.jsonl
  agent_rl_math.jsonl
  manifest.json

주의:
- MiniMind의 SFTDataset은 `tools`, `tool_calls`를 문자열로 읽기 때문에
  해당 필드는 JSON 문자열로 직렬화합니다.
- RLAIFDataset은 conversations[:-1]을 rollout prompt로 쓰므로 마지막
  assistant placeholder를 유지합니다.
- AgentRLDataset은 top-level `gt`를 요구합니다.
- Agent RL 샘플은 trainer/train_agent.py의 여섯 가지 도구만 사용합니다.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

SEED = 42

# -----------------------------------------------------------------------------
# Public Korean source datasets
# -----------------------------------------------------------------------------
PRETRAIN_SOURCES = [
    {
        "repo": "AdaMLLab/KorMix",
        "config": "minhash_deduped",
        "weight": 1.0
    },
]

SFT_SOURCES = [
    {"repo": "channelcorp/KoMagpie-raw", "weight": 0.55, "kind": "generic"},
    {"repo": "beomi/KoAlpaca-RealQA", "weight": 0.20, "kind": "generic"},
    {"repo": "llami-team/Korean-OpenThoughts-114k-Normalized", "weight": 0.25, "kind": "reasoning"},
]

DPO_SOURCES = [
    {"repo": "maywell/ko_Ultrafeedback_binarized", "weight": 1.0},
]

PRESETS = {
    "tiny": {
        "pretrain_mini": 10_000,
        "pretrain_full": 30_000,
        "sft_mini": 5_000,
        "sft_full": 15_000,
        "dpo": 3_000,
        "rlaif": 3_000,
        "agent": 2_000,
        "agent_math": 3_000,
        "tool_sft_ratio": 0.08,
    },
    "mini": {
        "pretrain_mini": 100_000,
        "pretrain_full": 500_000,
        "sft_mini": 50_000,
        "sft_full": 200_000,
        "dpo": 20_000,
        "rlaif": 30_000,
        "agent": 20_000,
        "agent_math": 30_000,
        "tool_sft_ratio": 0.10,
    },
    "full": {
        "pretrain_mini": 200_000,
        "pretrain_full": 2_000_000,
        "sft_mini": 100_000,
        "sft_full": 600_000,
        "dpo": 60_000,
        "rlaif": 100_000,
        "agent": 80_000,
        "agent_math": 120_000,
        "tool_sft_ratio": 0.12,
    },
}

# -----------------------------------------------------------------------------
# Exact tool set from current MiniMind trainer/train_agent.py
# -----------------------------------------------------------------------------
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "수식을 계산합니다.", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "단위를 변환합니다.", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "현재 날씨를 조회합니다.", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "현재 시각을 조회합니다.", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Seoul"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "환율을 조회합니다.", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "텍스트를 번역합니다.", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# Must mirror current train_agent.py's mock environment.
WEATHER_DATA = {
    "서울": ("28°C", "맑음"), "부산": ("25°C", "흐림"), "제주": ("30°C", "비"),
    "Tokyo": ("12°C", "맑음"), "New York": ("8°C", "흐림"),
    "London": ("5°C", "비"), "Paris": ("10°C", "맑음"), "Sydney": ("25°C", "맑음"),
}
TIME_DATA = {
    "Asia/Seoul": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00",
    "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00",
    "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00",
}
EXCHANGE_DATA = {
    ("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12,
    ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79,
    ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72,
}
TRANSLATE_DATA = {
    ("안녕하세요", "english"): "Hello",
    ("오늘 날씨가 좋습니다", "english"): "The weather is nice today",
    ("기계 학습은 흥미롭습니다", "english"): "Machine learning is interesting",
    ("Good morning", "korean"): "좋은 아침입니다",
    ("I love programming", "korean"): "저는 프로그래밍을 좋아합니다",
    ("Happy birthday", "korean"): "생일 축하합니다",
}
UNIT_DATA = {
    "km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462,
    "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048,
    "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556,
}

SYSTEM_TOOL_TEXT = "# Tools\n필요한 경우 아래 도구를 호출하세요. 도구 호출 결과를 바탕으로 최종 답변을 작성하세요."

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        try:
            x = json.dumps(x, ensure_ascii=False)
        except Exception:
            x = str(x)
    x = x.replace("\x00", "")
    x = re.sub(r"\r\n?", "\n", x)
    x = re.sub(r"[ \t]+", " ", x)
    x = re.sub(r"\n{4,}", "\n\n\n", x)
    return x.strip()


def repair_korean_mojibake(s: str) -> str:
    """CP949/EUC-KR로 잘못 디코딩된 UTF-8 한국어 텍스트를 복구합니다."""
    try:
        repaired = s.encode("cp949", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return s
    if not repaired:
        return s

    def score(text: str) -> tuple[int, int]:
        hangul = len(re.findall(r"[가-힣]", text))
        suspicious = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff�]", text))
        return hangul, -suspicious

    return repaired if score(repaired) > score(s) else s


def has_korean(s: str) -> bool:
    return bool(re.search(r"[가-힣]", s))


def digest(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    n = 0
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp_path.replace(path)
    return n


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_jsonl_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def complete_marker(path: Path) -> Path:
    return path.with_name(path.name + ".complete.json")


def read_complete_marker(path: Path) -> Optional[Dict[str, Any]]:
    marker = complete_marker(path)
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("size_bytes") != path.stat().st_size:
        return None
    return data


def write_complete_marker(path: Path, rows: int, kind: str, expected: int) -> None:
    marker = complete_marker(path)
    data = {
        "rows": rows,
        "size_bytes": path.stat().st_size,
        "kind": kind,
        "expected": expected,
        "source_exhausted": rows < expected,
    }
    marker.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_load(repo: str, config: str = None):
    print(f"\n[LOAD] {repo}")
    try:
        return load_dataset(repo, config, split="train", streaming=True)
    except Exception as e1:
        print(f"[WARN] train split 로드 실패: {e1}")
    try:
        ds = load_dataset(repo, config, streaming=True)
        split = next(iter(ds.keys()))
        print(f"[INFO] split={split} 사용")
        return ds[split]
    except Exception as e2:
        print(f"[SKIP] {repo}: {e2}")
        return None


def shuffled_take(ds, n: int, seed: int) -> Iterator[Dict[str, Any]]:
    if ds is None:
        return iter(())
    try:
        ds = ds.shuffle(seed=seed, buffer_size=min(max(10_000, n), 100_000))
    except Exception:
        pass
    def gen():
        for i, row in enumerate(ds):
            if i >= n:
                break
            yield row
    return gen()

# -----------------------------------------------------------------------------
# Message conversion
# -----------------------------------------------------------------------------
ROLE_MAP = {"human": "user", "user": "user", "prompt": "user", "gpt": "assistant", "assistant": "assistant", "bot": "assistant", "system": "system", "tool": "tool", "function": "tool"}


def normalize_message(m: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(m, dict):
        return None
    role = ROLE_MAP.get(str(m.get("role") or m.get("from") or m.get("speaker") or "").lower(), str(m.get("role") or "").lower())
    content = normalize_text(m.get("content") if m.get("content") is not None else m.get("value") if m.get("value") is not None else m.get("text"))
    if role not in {"system", "user", "assistant", "tool"}:
        return None
    out: Dict[str, Any] = {"role": role, "content": content}

    # MiniMind SFTDataset declares these as strings.
    if m.get("reasoning_content") is not None:
        out["reasoning_content"] = normalize_text(m.get("reasoning_content"))
    if m.get("tools") is not None:
        out["tools"] = m["tools"] if isinstance(m["tools"], str) else json_dumps(m["tools"])
    tc = m.get("tool_calls") if m.get("tool_calls") is not None else m.get("function_call")
    if tc is not None:
        if isinstance(tc, str):
            out["tool_calls"] = tc
        else:
            # OpenAI nested function call -> MiniMind {name, arguments}
            calls = tc if isinstance(tc, list) else [tc]
            converted = []
            for c in calls:
                if isinstance(c, dict) and isinstance(c.get("function"), dict):
                    fn = c["function"]
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except Exception: pass
                    converted.append({"name": fn.get("name", ""), "arguments": args})
                elif isinstance(c, dict):
                    converted.append(c)
            out["tool_calls"] = json_dumps(converted)
    return out


def normalize_conversation(v: Any) -> List[Dict[str, Any]]:
    if isinstance(v, str):
        try: v = json.loads(v)
        except Exception: return []
    if isinstance(v, dict):
        for k in ("messages", "conversations", "conversation"):
            if k in v:
                return normalize_conversation(v[k])
        return []
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        m = normalize_message(x)
        if m:
            out.append(m)
    return out


def extract_generic_sft(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    for k in ("conversations", "messages", "conversation", "chat"):
        if k in row:
            conv = normalize_conversation(row[k])
            if len(conv) >= 2:
                return conv
    q = normalize_text(row.get("instruction") or row.get("question") or row.get("prompt") or row.get("query"))
    inp = normalize_text(row.get("input"))
    a = normalize_text(row.get("output") or row.get("response") or row.get("answer") or row.get("completion"))
    if q and inp:
        q += "\n\n" + inp
    if q and a:
        return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    return []


def extract_reasoning_sft(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    q = normalize_text(row.get("question") or row.get("prompt") or row.get("problem"))
    reasoning = normalize_text(row.get("reasoning") or row.get("rationale") or row.get("solution"))
    answer = normalize_text(row.get("response") or row.get("answer") or row.get("final_answer"))
    if not q or not answer:
        return []
    # Current template supports reasoning_content explicitly.
    msg: Dict[str, Any] = {"role": "assistant", "content": answer}
    if reasoning:
        msg["reasoning_content"] = reasoning
    return [{"role": "user", "content": q}, msg]


def valid_sft(conv: List[Dict[str, Any]]) -> bool:
    if len(conv) < 2:
        return False
    roles = [m.get("role") for m in conv]
    if "user" not in roles or "assistant" not in roles:
        return False
    total = sum(len(normalize_text(m.get("content"))) + len(normalize_text(m.get("reasoning_content"))) for m in conv)
    return 20 <= total <= 60_000

# -----------------------------------------------------------------------------
# Pretrain
# -----------------------------------------------------------------------------
def extract_pretrain(row: Dict[str, Any]) -> str:
    for k in ("text", "content", "document", "body", "article", "raw_text"):
        v = row.get(k)
        if isinstance(v, str):
            v = normalize_text(v)
            v = repair_korean_mojibake(v)
            if 50 <= len(v) <= 100_000 and has_korean(v):
                return v
    return ""


def collect_pretrain(target: int, seed: int) -> Iterator[Dict[str, Any]]:
    seen = set()
    total_weight = sum(s["weight"] for s in PRETRAIN_SOURCES)
    produced = 0
    for idx, src in enumerate(PRETRAIN_SOURCES):
        print(src)
        quota = max(1, round(target * src["weight"] / total_weight))
        ds = safe_load(src["repo"], src.get("config"))
        got = 0
        for row in tqdm(shuffled_take(ds, quota * 4, seed + idx), desc=f"pretrain:{src['repo']}"):
            text = extract_pretrain(row)
            if not text:
                continue
            h = digest(text)
            if h in seen:
                continue
            seen.add(h)
            yield {"text": text}
            got += 1; produced += 1
            if got >= quota or produced >= target:
                break
        if produced >= target:
            break

# -----------------------------------------------------------------------------
# Tool SFT generator (exact MiniMind SFT message shape)
# -----------------------------------------------------------------------------
def safe_eval_expr(expr: str) -> float:
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Mod, ast.FloorDiv)
    if not all(isinstance(n, allowed) for n in ast.walk(tree)):
        raise ValueError("unsafe expression")
    return eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}})


def make_math_case(rng: random.Random) -> Tuple[str, Dict[str, Any], Dict[str, Any], str, str]:
    a, b = rng.randint(2, 999), rng.randint(2, 999)
    op = rng.choice(["+", "-", "*", "/"])
    if op == "/":
        a = b * rng.randint(2, 200)
    expr = f"{a}{op}{b}"
    result = safe_eval_expr(expr)
    if isinstance(result, float) and result.is_integer(): result = int(result)
    q = rng.choice([f"{expr} 계산해줘.", f"계산기로 {expr}의 값을 구해줘.", f"{expr}은 얼마야?"])
    call = {"name": "calculate_math", "arguments": {"expression": expr}}
    tool = {"result": str(result)}
    ans = f"계산 결과는 {result}입니다."
    return q, call, tool, ans, str(result)


def make_unit_case(rng: random.Random):
    key = rng.choice(list(UNIT_DATA.keys()))
    fr, to = key.split("_", 1)
    value = rng.choice([1, 2, 5, 10, 25, 100])
    result = round(value * UNIT_DATA[key], 4)
    q = f"{value} {fr}를 {to}로 변환해줘."
    call = {"name": "unit_converter", "arguments": {"value": value, "from_unit": fr, "to_unit": to}}
    tool = {"result": result}
    ans = f"{value} {fr}는 약 {result} {to}입니다."
    return q, call, tool, ans, str(result)


def make_weather_case(rng: random.Random):
    loc = rng.choice(list(WEATHER_DATA.keys()))
    temp, cond = WEATHER_DATA[loc]
    q = f"{loc} 날씨를 조회해줘."
    call = {"name": "get_current_weather", "arguments": {"location": loc}}
    tool = {"city": loc, "temperature": temp, "humidity": "65%", "condition": cond}
    ans = f"{loc}의 현재 기온은 {temp}이고 날씨는 {cond}입니다."
    # one gt per tool call, because current reward's tool_gap compares call count vs len(gt)
    return q, call, tool, ans, temp


def make_time_case(rng: random.Random):
    tz = rng.choice(list(TIME_DATA.keys()))
    dt = TIME_DATA[tz]
    q = f"{tz} 시간대의 현재 시각을 알려줘."
    call = {"name": "get_current_time", "arguments": {"timezone": tz}}
    tool = {"datetime": dt, "timezone": tz}
    ans = f"{tz}의 현재 시각은 {dt}입니다."
    return q, call, tool, ans, dt


def make_fx_case(rng: random.Random):
    pair = rng.choice(list(EXCHANGE_DATA.keys()))
    fr, to = pair; rate = EXCHANGE_DATA[pair]
    q = f"{fr}에서 {to}로 가는 환율을 조회해줘."
    call = {"name": "get_exchange_rate", "arguments": {"from_currency": fr, "to_currency": to}}
    tool = {"from": fr, "to": to, "rate": rate}
    ans = f"{fr}/{to} 환율은 {rate}입니다."
    return q, call, tool, ans, str(rate)


def make_translate_case(rng: random.Random):
    text, target = rng.choice(list(TRANSLATE_DATA.keys()))
    translated = TRANSLATE_DATA[(text, target)]
    q = f"'{text}'를 {target}로 번역해줘."
    call = {"name": "translate_text", "arguments": {"text": text, "target_language": target}}
    tool = {"translated_text": translated}
    ans = f"번역 결과: {translated}"
    return q, call, tool, ans, translated

TOOL_CASE_MAKERS = [make_math_case, make_unit_case, make_weather_case, make_time_case, make_fx_case, make_translate_case]


def make_tool_sft_sample(rng: random.Random) -> Dict[str, Any]:
    q, call, tool_result, final_answer, _gt = rng.choice(TOOL_CASE_MAKERS)(rng)
    return {"conversations": [
        {"role": "system", "content": SYSTEM_TOOL_TEXT, "tools": json_dumps(TOOLS)},
        {"role": "user", "content": q},
        {"role": "assistant", "content": "", "tool_calls": json_dumps([call])},
        {"role": "tool", "content": json_dumps(tool_result)},
        {"role": "assistant", "content": final_answer},
    ]}

# -----------------------------------------------------------------------------
# SFT
# -----------------------------------------------------------------------------
def collect_sft(target: int, seed: int, tool_ratio: float) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    tool_target = round(target * tool_ratio)
    normal_target = target - tool_target
    seen = set()
    total_weight = sum(s["weight"] for s in SFT_SOURCES)
    produced = 0

    for idx, src in enumerate(SFT_SOURCES):
        quota = max(1, round(normal_target * src["weight"] / total_weight))
        ds = safe_load(src["repo"])
        got = 0
        for row in tqdm(shuffled_take(ds, quota * 4, seed + idx), desc=f"sft:{src['repo']}"):
            conv = extract_reasoning_sft(row) if src["kind"] == "reasoning" else extract_generic_sft(row)
            if not valid_sft(conv):
                continue
            sig = json_dumps(conv)
            h = digest(sig)
            if h in seen:
                continue
            seen.add(h)
            yield {"conversations": conv}
            got += 1; produced += 1
            if got >= quota or produced >= normal_target:
                break
        if produced >= normal_target:
            break

    for _ in range(tool_target):
        yield make_tool_sft_sample(rng)

# -----------------------------------------------------------------------------
# DPO
# -----------------------------------------------------------------------------
def list_or_text(v: Any) -> List[Dict[str, Any]]:
    conv = normalize_conversation(v)
    if conv:
        return [{"content": m.get("content", ""), "role": m["role"]} for m in conv if m["role"] in {"system", "user", "assistant"}]
    txt = normalize_text(v)
    return [{"content": txt, "role": "assistant"}] if txt else []


def extract_dpo(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chosen = list_or_text(row.get("chosen") or row.get("preferred") or row.get("chosen_response"))
    rejected = list_or_text(row.get("rejected") or row.get("non_preferred") or row.get("rejected_response"))
    prompt = normalize_text(row.get("prompt") or row.get("instruction") or row.get("question"))
    if not chosen or not rejected:
        return None
    if prompt and chosen[0]["role"] == "assistant":
        chosen.insert(0, {"content": prompt, "role": "user"})
    if prompt and rejected[0]["role"] == "assistant":
        rejected.insert(0, {"content": prompt, "role": "user"})
    return {"chosen": chosen, "rejected": rejected}


def collect_dpo(target: int, seed: int) -> Iterator[Dict[str, Any]]:
    seen = set(); produced = 0
    for idx, src in enumerate(DPO_SOURCES):
        ds = safe_load(src["repo"])
        for row in tqdm(shuffled_take(ds, target * 3, seed + idx), desc=f"dpo:{src['repo']}"):
            item = extract_dpo(row)
            if not item:
                continue
            h = digest(json_dumps(item))
            if h in seen:
                continue
            seen.add(h)
            yield item
            produced += 1
            if produced >= target:
                return

# -----------------------------------------------------------------------------
# RLAIF: exact behavior expected by RLAIFDataset (conversations[:-1])
# -----------------------------------------------------------------------------
def collect_rlaif(sft_path: Path, target: int, seed: int) -> Iterator[Dict[str, Any]]:
    candidates = []
    for obj in iter_jsonl(sft_path):
        conv = obj.get("conversations")
        if not isinstance(conv, list) or len(conv) < 2:
            continue
        # Avoid tool SFT here; generic RLAIF can be reward-model judged.
        if any(m.get("tools") or m.get("role") == "tool" for m in conv if isinstance(m, dict)):
            continue
        if conv[-1].get("role") != "assistant":
            continue
        prompt_conv = [dict(m) for m in conv[:-1]] + [{"role": "assistant", "content": ""}]
        candidates.append({"conversations": prompt_conv})
        if len(candidates) >= target * 3:
            break
    rng = random.Random(seed)
    rng.shuffle(candidates)
    yield from candidates[:target]

# -----------------------------------------------------------------------------
# Agent RL: exact AgentRLDataset contract = conversations + top-level gt
# -----------------------------------------------------------------------------
def make_agent_rl_sample(rng: random.Random, math_only: bool = False) -> Dict[str, Any]:
    maker = make_math_case if math_only else rng.choice(TOOL_CASE_MAKERS)
    q, _call, _tool_result, _final_answer, gt = maker(rng)
    # AgentRLDataset does messages[:-1], so the final assistant placeholder is mandatory.
    # The tool list is carried on system.tools, exactly as SFTDataset/AgentRLDataset parse it.
    return {
        "conversations": [
            {"role": "system", "content": SYSTEM_TOOL_TEXT, "tools": json_dumps(TOOLS)},
            {"role": "user", "content": q},
            {"role": "assistant", "content": ""},
        ],
        "gt": [gt],
    }


def collect_agent(target: int, seed: int, math_only: bool = False) -> Iterator[Dict[str, Any]]:
    rng = random.Random(seed)
    seen = set()
    produced = 0
    duplicate_retries = 0
    max_duplicate_retries = max(10_000, target)
    while produced < target:
        item = make_agent_rl_sample(rng, math_only=math_only)
        h = digest(json_dumps(item))
        if h in seen:
            duplicate_retries += 1
            if math_only and duplicate_retries < max_duplicate_retries:
                continue
        else:
            seen.add(h)
            duplicate_retries = 0
        produced += 1
        yield item

# -----------------------------------------------------------------------------
# Validation against current MiniMind reader contracts
# -----------------------------------------------------------------------------
def validate_pretrain(obj: Dict[str, Any]) -> None:
    assert isinstance(obj.get("text"), str)


def validate_sft(obj: Dict[str, Any]) -> None:
    conv = obj.get("conversations")
    assert isinstance(conv, list) and conv
    for m in conv:
        assert m.get("role") in {"system", "user", "assistant", "tool"}
        assert isinstance(m.get("content", ""), str)
        if "tools" in m:
            assert isinstance(m["tools"], str); json.loads(m["tools"])
        if "tool_calls" in m:
            assert isinstance(m["tool_calls"], str); json.loads(m["tool_calls"])
        if "reasoning_content" in m:
            assert isinstance(m["reasoning_content"], str)


def validate_dpo(obj: Dict[str, Any]) -> None:
    for k in ("chosen", "rejected"):
        assert isinstance(obj.get(k), list) and obj[k]
        for m in obj[k]:
            assert m.get("role") in {"system", "user", "assistant"}
            assert isinstance(m.get("content"), str)


def validate_rlaif(obj: Dict[str, Any]) -> None:
    validate_sft(obj)
    conv = obj["conversations"]
    assert conv[-1].get("role") == "assistant"


def validate_agent(obj: Dict[str, Any]) -> None:
    validate_sft(obj)
    assert isinstance(obj.get("gt"), list)
    assert obj["gt"]
    conv = obj["conversations"]
    assert conv[-1].get("role") == "assistant"
    systems = [m for m in conv if m.get("role") == "system" and m.get("tools")]
    assert systems, "Agent RL requires system.tools"
    parsed = json.loads(systems[0]["tools"])
    valid_names = {t["function"]["name"] for t in parsed}
    upstream_names = {t["function"]["name"] for t in TOOLS}
    assert valid_names <= upstream_names


def validate_file(path: Path, kind: str, limit: int = 1000) -> int:
    fn = {"pretrain": validate_pretrain, "sft": validate_sft, "dpo": validate_dpo, "rlaif": validate_rlaif, "agent": validate_agent}[kind]
    n = 0
    for obj in iter_jsonl(path):
        fn(obj); n += 1
        if n >= limit:
            break
    if n == 0:
        raise RuntimeError(f"{path} is empty")
    return n

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="MiniMind-3 호환 한국어 데이터셋 생성기")
    p.add_argument("--preset", choices=PRESETS.keys(), default="mini", help="생성 규모: tiny, mini, full")
    p.add_argument("--output", default="dataset_ko", help="출력 디렉터리")
    p.add_argument("--seed", type=int, default=SEED, help="랜덤 시드")
    p.add_argument("--skip-full", action="store_true", help="mini 파일과 RL 파일만 생성")
    p.add_argument("--only", default="", help="쉼표로 구분한 파일명만 생성합니다. 예: agent_rl_math.jsonl,sft_t2t.jsonl")
    args = p.parse_args()

    cfg = PRESETS[args.preset]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {"preset": args.preset, "files": {}, "sources": {"pretrain": PRETRAIN_SOURCES, "sft": SFT_SOURCES, "dpo": DPO_SOURCES}}
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    jobs = []

    def wants(name: str) -> bool:
        return not only or name in only or Path(name).stem in only

    def build(name: str, rows: Iterable[Dict[str, Any]], kind: str, expected: int):
        path = out / name
        print(f"\n=== BUILD {name} ===")
        if path.exists():
            marked = read_complete_marker(path)
            if marked:
                checked = validate_file(path, kind)
                mb = path.stat().st_size / 1024 / 1024
                existing_count = int(marked.get("rows", 0))
                stats["files"][name] = {"rows": existing_count, "size_mb": round(mb, 2), "validated_first_rows": checked, "kind": kind, "skipped_existing": True}
                print(f"[SKIP] {name}: complete marker {existing_count:,} rows / {mb:.1f} MB")
                return path
            existing_count = count_jsonl_lines(path)
            if existing_count >= expected:
                checked = validate_file(path, kind)
                mb = path.stat().st_size / 1024 / 1024
                stats["files"][name] = {"rows": existing_count, "size_mb": round(mb, 2), "validated_first_rows": checked, "kind": kind, "skipped_existing": True}
                write_complete_marker(path, existing_count, kind, expected)
                print(f"[SKIP] {name}: existing {existing_count:,} rows / {mb:.1f} MB")
                return path
            print(f"[REBUILD] {name}: existing {existing_count:,} rows < expected {expected:,}")
        count = write_jsonl(path, rows)
        checked = validate_file(path, kind)
        mb = path.stat().st_size / 1024 / 1024
        stats["files"][name] = {"rows": count, "size_mb": round(mb, 2), "validated_first_rows": checked, "kind": kind}
        write_complete_marker(path, count, kind, expected)
        print(f"[OK] {name}: {count:,} rows / {mb:.1f} MB")
        return path

    pre_mini = out / "pretrain_t2t_mini.jsonl"
    if wants("pretrain_t2t_mini.jsonl"):
        pre_mini = build("pretrain_t2t_mini.jsonl", collect_pretrain(cfg["pretrain_mini"], args.seed), "pretrain", cfg["pretrain_mini"])
    if not args.skip_full and wants("pretrain_t2t.jsonl"):
        build("pretrain_t2t.jsonl", collect_pretrain(cfg["pretrain_full"], args.seed + 10), "pretrain", cfg["pretrain_full"])

    sft_mini = out / "sft_t2t_mini.jsonl"
    if wants("sft_t2t_mini.jsonl"):
        sft_mini = build("sft_t2t_mini.jsonl", collect_sft(cfg["sft_mini"], args.seed + 20, cfg["tool_sft_ratio"]), "sft", cfg["sft_mini"])
    if args.skip_full:
        sft_full = sft_mini
    else:
        sft_full = out / "sft_t2t.jsonl"
        if wants("sft_t2t.jsonl"):
            sft_full = build("sft_t2t.jsonl", collect_sft(cfg["sft_full"], args.seed + 30, cfg["tool_sft_ratio"]), "sft", cfg["sft_full"])

    if wants("dpo.jsonl"):
        build("dpo.jsonl", collect_dpo(cfg["dpo"], args.seed + 40), "dpo", cfg["dpo"])
    if wants("rlaif.jsonl"):
        build("rlaif.jsonl", collect_rlaif(sft_full, cfg["rlaif"], args.seed + 50), "rlaif", cfg["rlaif"])
    if wants("agent_rl.jsonl"):
        build("agent_rl.jsonl", collect_agent(cfg["agent"], args.seed + 60, math_only=False), "agent", cfg["agent"])
    if wants("agent_rl_math.jsonl"):
        build("agent_rl_math.jsonl", collect_agent(cfg["agent_math"], args.seed + 70, math_only=True), "agent", cfg["agent_math"])

    manifest = out / "manifest.json"
    manifest.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== DONE ===")
    print(out.resolve())
    print("생성된 JSONL 파일을 MiniMind 저장소의 ./dataset/ 디렉터리에 복사해서 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
