# 참고: tokenizer를 다시 학습하는 것은 권장하지 않습니다. MiniMind에는 이미 tokenizer가 포함되어 있으며, 이 스크립트는 학습과 참고용입니다. 서로 다른 tokenizer로 학습한 모델은 출력 호환성이 떨어집니다.
# Note: It is not recommended to re-train the tokenizer. MiniMind already includes one. This script is for learning and reference only. Training models with different tokenizers will lead to inconsistent outputs and reduce model reusability in the community.
import os
import json
from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

DATA_PATH = '../dataset/sft_t2t_mini.jsonl'
TOKENIZER_DIR = '../model_learn_tokenizer/'
VOCAB_SIZE = 6400
SPECIAL_TOKENS_NUM = 36

def get_texts(data_path):
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= 10000: break # 테스트용으로 10000줄만 사용합니다.
            try:
                data = json.loads(line)
                contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                if contents:
                    yield "\n".join(contents)
            except json.JSONDecodeError:
                continue

def train_tokenizer(data_path, tokenizer_dir, vocab_size, special_tokens_num=SPECIAL_TOKENS_NUM):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    special_tokens_list = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", 
        "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", 
        "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>", 
        "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>", "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"
    ]
    
    additional_tokens_list = [
        "<tool_call>", "</tool_call>",
        "<tool_response>", "</tool_response>",
        "<think>", "</think>"
    ]
    num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)
    buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)] # 일정 수의 buffer token 자리를 남깁니다.
    all_special_tokens = special_tokens_list + additional_tokens_list + buffer_tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=all_special_tokens
    )
    texts = get_texts(data_path)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(special_tokens_list)

    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save(tokenizer_dir)
    tokenizer_json_path = os.path.join(tokenizer_dir, "tokenizer.json")
    with open(tokenizer_json_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)
    for token_info in tokenizer_data.get('added_tokens', []):
        if token_info['content'] not in special_tokens_list:
            token_info['special'] = False
    with open(tokenizer_json_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    
    added_tokens_decoder = {}
    for i, token in enumerate(all_special_tokens):
        idx = tokenizer.token_to_id(token)
        added_tokens_decoder[str(idx)] = {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True if token in special_tokens_list else False
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [t for t in special_tokens_list if t not in ["<|endoftext|>"]],
        "bos_token": "<|im_start|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>",
        "legacy": True,
        "model_max_length": 131072,
        "pad_token": "<|endoftext|>",
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "unk_token": "<|endoftext|>",
        "image_token": "<|image_pad|>",
        "audio_token": "<|audio_pad|>",
        "video_token": "<|video_pad|>",
        "vision_bos_token": "<|vision_start|>",
        "vision_eos_token": "<|vision_end|>",
        "audio_bos_token": "<|audio_start|>",
        "audio_eos_token": "<|audio_end|>",
        "chat_template": "{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0].role == 'system' %}\n        {{- messages[0].content + '\\n\\n' }}\n    {%- endif %}\n    {{- \"# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0].role == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}\n{%- for message in messages[::-1] %}\n    {%- set index = (messages|length - 1) - loop.index0 %}\n    {%- if ns.multi_step_tool and message.role == \"user\" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}\n        {%- set ns.multi_step_tool = false %}\n        {%- set ns.last_query_index = index %}\n    {%- endif %}\n{%- endfor %}\n{%- for message in messages %}\n    {%- if message.content is string %}\n        {%- set content = message.content %}\n    {%- else %}\n        {%- set content = '' %}\n    {%- endif %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}\n        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {%- set reasoning_content = '' %}\n        {%- if message.reasoning_content is string %}\n            {%- set reasoning_content = message.reasoning_content %}\n        {%- else %}\n            {%- if '</think>' in content %}\n                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n            {%- endif %}\n        {%- endif %}\n        {%- if true %}\n            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}\n        {%- endif %}\n        {%- if message.tool_calls %}\n            {%- for tool_call in message.tool_calls %}\n                {%- if (loop.first and content) or (not loop.first) %}\n                    {{- '\\n' }}\n                {%- endif %}\n                {%- if tool_call.function %}\n                    {%- set tool_call = tool_call.function %}\n                {%- endif %}\n                {{- '<tool_call>\\n{\"name\": \"' }}\n                {{- tool_call.name }}\n                {{- '\", \"arguments\": ' }}\n                {%- if tool_call.arguments is string %}\n                    {{- tool_call.arguments }}\n                {%- else %}\n                    {{- tool_call.arguments | tojson }}\n                {%- endif %}\n                {{- '}\\n</tool_call>' }}\n            {%- endfor %}\n        {%- endif %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if loop.first or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n    {%- if open_thinking is defined and open_thinking is true %}\n        {{- '<think>\\n' }}\n    {%- else %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}",
        "tokenizer_class": "PreTrainedTokenizerFast"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print("Tokenizer training completed.")

def eval_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    messages = [
        {"role": "system", "content": "당신은 훌륭한 챗봇이며 항상 올바르게 응답합니다."},
        {"role": "user", "content": '너는 어디에서 왔니?'},
        {"role": "assistant", "content": '저는 달에서 왔습니다.'},
        {"role": "user", "content": '정확히 어디에서 왔니?'},
        {"role": "assistant", "content": '저는 지구에서 왔습니다.'}
    ]
    new_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False
    )
    print('-'*100)
    print(new_prompt)
    print('-'*100)
    print('tokenizer vocab 크기:', len(tokenizer))
    model_inputs = tokenizer(new_prompt)
    print('encoder 길이:', len(model_inputs['input_ids']))
    response = tokenizer.decode(model_inputs['input_ids'], skip_special_tokens=False)
    print('decoder 일치 여부:', response == new_prompt, "\n")
    print('-'*100)
    print('압축률 테스트(Chars/Tokens):')
    test_texts = [
        # 한국어 샘플
        "인공지능은 컴퓨터 과학의 한 분야로, 지능의 본질을 이해하고 사람처럼 반응할 수 있는 시스템을 만드는 것을 목표로 합니다. 이 분야에는 자연어 처리, 음성 인식, 이미지 인식, 로봇, 전문가 시스템 등이 포함됩니다.",
        "우주 항행은 행성 사이 또는 더 먼 우주 공간을 이동하는 기술을 뜻합니다. 화학 로켓만으로는 먼 거리를 이동하기 어렵기 때문에 이온 추진, 핵열 로켓, 핵융합 추진 같은 여러 아이디어가 연구되고 있습니다.",
        # 영어 샘플
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information, and the need for significant computational resources.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply. International cooperation and policy frameworks are also necessary to drive the global shift towards a greener economy and reduce carbon emissions.",
        # 혼합 샘플
        "Python은 간결한 문법과 풍부한 생태계로 잘 알려진 고급 프로그래밍 언어입니다. It is widely used in data science, machine learning, and web development. 개발자는 NumPy, Pandas, PyTorch 같은 라이브러리를 활용해 복잡한 애플리케이션을 빠르게 만들 수 있습니다.",
    ]
    
    total_compression = 0
    for i, text in enumerate(test_texts):
        encoded = tokenizer.encode(text)
        token_count = len(encoded)
        char_count = len(text)
        compression_ratio = char_count / token_count
        total_compression += compression_ratio
        print(f"샘플 {i+1} | 문자 수: {char_count:4} | Tokens: {token_count:3} | 압축률: {compression_ratio:.2f}")
    
    print(f"평균 압축률: {total_compression / len(test_texts):.2f}")
    print('-'*100)
    print('스트리밍 디코딩(바이트 버퍼) 테스트:')
    input_ids = model_inputs['input_ids']
    token_cache = []
    for tid in input_ids:
        token_cache.append(tid)
        current_decode = tokenizer.decode(token_cache)
        if current_decode and '\ufffd' not in current_decode:
            display_ids = token_cache[0] if len(token_cache) == 1 else token_cache
            raw_tokens = [tokenizer.convert_ids_to_tokens(int(t)) for t in (token_cache if isinstance(token_cache, list) else [token_cache])]
            print(f'Token ID: {str(display_ids):15} -> Raw: {str(raw_tokens):20} -> Decode Str: {current_decode}')
            token_cache = []

if __name__ == '__main__':
    train_tokenizer(DATA_PATH, TOKENIZER_DIR, VOCAB_SIZE)
    eval_tokenizer(TOKENIZER_DIR)
