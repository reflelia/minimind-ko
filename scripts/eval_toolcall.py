import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "수학 표현식의 결과를 계산합니다. 사칙연산, 거듭제곱, 제곱근 등을 지원합니다.", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "수학 표현식. 예: 123+456, 2**10, sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "현재 날짜와 시간을 조회합니다. 시간대 지정이 가능합니다.", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "시간대 이름. 예: Asia/Seoul, America/New_York", "default": "Asia/Seoul"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "지정 범위의 난수를 생성합니다.", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "최솟값", "default": 0}, "max": {"type": "integer", "description": "최댓값", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "텍스트의 문자 수와 단어 수를 계산합니다.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "통계를 낼 텍스트"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "길이, 무게, 온도 등의 단위 변환을 수행합니다.", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "변환할 값"}, "from_unit": {"type": "string", "description": "원본 단위. 예: km, miles, kg, pounds, celsius, fahrenheit"}, "to_unit": {"type": "string", "description": "대상 단위"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "지정 도시의 현재 날씨 정보를 조회합니다. 온도, 습도, 날씨 상태를 포함합니다.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "도시 이름. 예: 서울, 부산, New York"}, "unit": {"type": "string", "description": "온도 단위. celsius 또는 fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "두 통화 사이의 실시간 환율을 조회합니다.", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "원본 통화 코드. 예: USD, KRW, EUR"}, "to_currency": {"type": "string", "description": "대상 통화 코드. 예: USD, KRW, EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "텍스트를 대상 언어로 번역합니다.", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "번역할 텍스트"}, "target_language": {"type": "string", "description": "대상 언어. 예: english, korean, japanese, french"}}, "required": ["text", "target_language"]}}},
]

MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Seoul")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "맑음"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    return [TOOL_MAP[n] for n in names]

TEST_CASES = [
    {"prompt": "256 곱하기 37이 얼마인지 계산해줘", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "지금 몇 시야?", "tools": ["get_current_time", "random_number"]},
    {"prompt": "100킬로미터를 마일로 변환해줘", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "1부터 1000까지 난수를 하나 만들고 그 제곱을 계산해줘", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "서울 오늘 날씨 어때?", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "달러 원 환율을 조회해줘", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "'안녕하세요 세계'를 영어로 번역해줘", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe)))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def parse_tool_call_from_text(content):
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {"name": data.get("name", ""), "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)}
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"알 수 없는 도구: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"도구 실행 실패: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    st = time.time()
    print('🧠: ', end='')
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s') if args.show_speed else print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    if not stream:
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None:
                tool_calls = []
            for tc_chunk in delta.tool_calls:
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({
                        "id": "",
                        "function": {"name": "", "arguments": ""}
                    })
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    messages = [{"role": "user", "content": prompt}]
    while True:
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
        if not tool_calls:
            break
        tool_calls = [{
            "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
            "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
            "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
        } for tc in tool_calls] if args.backend == 'api' else tool_calls
        messages.append({"role": "assistant", "content": content} if args.backend == 'local' else {"role": "assistant", "content": content, "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls]})
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)} if args.backend == 'local' else {"role": "tool", "content": json.dumps(result, ensure_ascii=False), "tool_call_id": tc["id"]})


def main():
    parser = argparse.ArgumentParser(description="MiniMind ToolCall 평가")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="추론 backend(local=로컬 모델, api=OpenAI 호환 API)")
    parser.add_argument('--load_from', default='model', type=str, help="모델 로드 경로(model=원본 torch 가중치, 그 외=transformers 형식)")
    parser.add_argument('--save_dir', default='out', type=str, help="모델 저장 디렉터리")
    parser.add_argument('--weight', default='full_sft', type=str, help="가중치 파일 접두사")
    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="hidden layer 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 구조 사용 여부(0=아니오, 1=예)")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="최대 생성 길이")
    parser.add_argument('--temperature', default=0.9, type=float, help="생성 또는 distillation temperature")
    parser.add_argument('--top_p', default=0.9, type=float, help="nucleus sampling 임계값(0-1)")
    parser.add_argument('--show_speed', default=0, type=int, help="decode 속도 표시(tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="OpenAI 호환 API base_url")
    parser.add_argument('--api_key', default='sk-123', type=str, help="OpenAI 호환 API key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API 요청에 사용할 모델 이름")
    parser.add_argument('--stream', default=1, type=int, help="API 모드 streaming 출력 여부(0=아니오, 1=예)")
    args = parser.parse_args()

    model = tokenizer = client = None
    if args.backend == 'local': model, tokenizer = init_model(args)
    else: client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    input_mode = int(input('[0] 자동 테스트\n[1] 직접 입력\n'))

    cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES] if input_mode == 0 else iter(lambda: {"prompt": input('💬: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": "", "tools": TOOLS, "tool_names": []})
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 사용 가능한 도구: {case["tool_names"]}\n')
            print(f'💬: {case["prompt"]}')
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
