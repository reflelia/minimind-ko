import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params
warnings.filterwarnings('ignore')

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind 모델 추론 및 대화")
    parser.add_argument('--load_from', default='model', type=str, help="모델 로드 경로(model=원본 torch 가중치, 그 외=transformers 형식)")
    parser.add_argument('--save_dir', default='out', type=str, help="모델 저장 디렉터리")
    parser.add_argument('--weight', default='full_sft', type=str, help="가중치 파일 접두사")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA 가중치 이름(None이면 사용 안 함)")
    parser.add_argument('--hidden_size', default=768, type=int, help="hidden size")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="hidden layer 수")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE 구조 사용 여부(0=아니오, 1=예)")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="RoPE 위치 인코딩 외삽 활성화")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="최대 생성 길이")
    parser.add_argument('--temperature', default=0.85, type=float, help="생성 또는 distillation temperature")
    parser.add_argument('--top_p', default=0.95, type=float, help="nucleus sampling 임계값(0-1)")
    parser.add_argument('--open_thinking', default=0, type=int, help="adaptive thinking 사용 여부(0=아니오, 1=예)")
    parser.add_argument('--historys', default=0, type=int, help="포함할 이전 대화 수(0이면 미사용)")
    parser.add_argument('--show_speed', default=1, type=int, help="decode 속도 표시(tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="실행 장치")
    args = parser.parse_args()
    
    prompts = [
        '너는 어떤 일을 잘하니?',
        '왜 하늘은 파란색이야?',
        'Python으로 피보나치 수열을 계산하는 함수를 작성해줘',
        '광합성의 기본 과정을 설명해줘',
        '내일 비가 오면 어떻게 외출하는 게 좋을까?',
        '반려동물로 고양이와 강아지의 장단점을 비교해줘',
        '머신러닝이 무엇인지 설명해줘',
        '한국 음식 몇 가지를 추천해줘'
    ]
    
    conversation = []
    model, tokenizer = init_model(args)
    input_mode = int(input('[0] 자동 테스트\n[1] 직접 입력\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    prompt_iter = prompts if input_mode == 0 else iter(lambda: input('💬: '), '')
    for prompt in prompt_iter:
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0: print(f'💬: {prompt}')
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        if 'pretrain' in args.weight:
            inputs = tokenizer.bos_token + prompt
        else:
            inputs = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True, open_thinking=bool(args.open_thinking))
        
        inputs = tokenizer(inputs, return_tensors="pt", truncation=True).to(args.device)

        print('🧠: ', end='')
        st = time.time()
        generated_ids = model.generate(
            inputs=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, temperature=args.temperature, repetition_penalty=1
        )
        response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response})
        gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s\n\n') if args.show_speed else print('\n\n')

if __name__ == "__main__":
    main()
