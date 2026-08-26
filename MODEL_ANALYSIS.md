# MiniMind 모델 코드 분석

이 문서는 `model/model_minimind.py`, `model/model_lora.py`, `trainer/trainer_utils.py`를 기준으로 현재 프로젝트의 모델 구조와 학습 연결 방식을 정리합니다.

## 요약

현재 모델은 외부 LLaMA/Qwen 가중치를 가져오는 구조가 아니라, `MiniMindForCausalLM`을 직접 생성하는 자체 decoder-only Transformer입니다. `train_pretrain.py`의 기본값은 `--from_weight none`이므로 pretraining은 랜덤 초기화 가중치에서 scratch로 시작합니다.

단, tokenizer는 `model/` 디렉터리의 기존 MiniMind tokenizer를 사용합니다. 따라서 “가중치 scratch pretraining”이며, tokenizer까지 새로 학습하는 완전 scratch 구성은 아닙니다.

```text
모델 구조: MiniMind 자체 LLaMA-like decoder-only Transformer
가중치 시작점: 기본 scratch
tokenizer: model/ 내부 MiniMind tokenizer
학습 결과 저장: out/<weight>_<hidden_size>.pth
MoE 학습 결과 저장: out/<weight>_<hidden_size>_moe.pth
```

## 기본 설정

`MiniMindConfig`의 주요 기본값은 다음과 같습니다.

| 항목 | 기본값 | 의미 |
| --- | ---: | --- |
| `hidden_size` | `768` | hidden dimension |
| `num_hidden_layers` | `8` | Transformer block 수 |
| `vocab_size` | `6400` | tokenizer vocab 크기 |
| `num_attention_heads` | `8` | query head 수 |
| `num_key_value_heads` | `4` | key/value head 수 |
| `head_dim` | `96` | head당 dimension |
| `hidden_act` | `silu` | FFN 활성화 함수 |
| `intermediate_size` | `2432` | dense FFN 내부 dimension |
| `max_position_embeddings` | `32768` | RoPE buffer 길이 |
| `rope_theta` | `1e6` | RoPE base |
| `tie_word_embeddings` | `True` | embedding과 lm_head weight 공유 |
| `dropout` | `0.0` | dropout |

기본 dense 모델의 실제 파라미터 수는 약 `63.9M`입니다.

```text
dense total params: 63,912,192
```

## 전체 구조

`MiniMindForCausalLM`은 Hugging Face의 `PreTrainedModel`, `GenerationMixin`을 상속합니다. 내부에는 `MiniMindModel`과 `lm_head`가 있습니다.

```text
input_ids
  -> token embedding
  -> dropout
  -> MiniMindBlock x num_hidden_layers
      -> RMSNorm
      -> causal self-attention
      -> residual add
      -> RMSNorm
      -> FeedForward 또는 MOEFeedForward
      -> residual add
  -> final RMSNorm
  -> lm_head
  -> logits
```

`tie_word_embeddings=True`이면 `model.embed_tokens.weight`와 `lm_head.weight`가 같은 파라미터를 공유합니다. 작은 모델에서 embedding/output layer의 파라미터 부담을 줄이는 데 의미가 있습니다.

## Attention

Attention은 causal self-attention입니다.

구성:

```text
q_proj: hidden_size -> num_attention_heads * head_dim
k_proj: hidden_size -> num_key_value_heads * head_dim
v_proj: hidden_size -> num_key_value_heads * head_dim
o_proj: num_attention_heads * head_dim -> hidden_size
```

기본값에서는 query head가 `8`, key/value head가 `4`입니다. 즉 GQA(Grouped Query Attention) 형태입니다.

```text
num_attention_heads = 8
num_key_value_heads = 4
n_rep = 8 / 4 = 2
```

K/V head를 `repeat_kv()`로 query head 수에 맞춰 반복합니다. 이 방식은 MHA보다 KV cache 크기를 줄이면서 MQA보다 표현력을 더 확보하는 절충입니다.

추가 특징:

```text
Q/K에 head_dim 단위 RMSNorm 적용
RoPE 위치 인코딩 적용
PyTorch scaled_dot_product_attention 사용 가능
fallback attention 구현 포함
past_key_value 기반 KV cache 지원
```

## RoPE와 YaRN

위치 정보는 절대 위치 embedding이 아니라 RoPE(Rotary Position Embedding)를 사용합니다.

`precompute_freqs_cis()`에서 `freqs_cos`, `freqs_sin` buffer를 미리 만들고, forward 시 현재 sequence 구간만 잘라 attention에 넣습니다.

```text
freqs_cos[start_pos:start_pos + seq_length]
freqs_sin[start_pos:start_pos + seq_length]
```

`inference_rope_scaling=True`이면 YaRN 방식의 RoPE scaling 설정이 켜집니다.

```text
factor = 16
original_max_position_embeddings = 2048
beta_fast = 32
beta_slow = 1
type = yarn
```

이 옵션은 긴 컨텍스트 추론 시 위치 인코딩 범위를 확장하기 위한 설정입니다. 코드상 기본 `max_position_embeddings`는 `32768`입니다.

## RMSNorm

정규화는 LayerNorm이 아니라 RMSNorm입니다.

```python
x * rsqrt(mean(x^2) + eps)
```

각 block에서 attention 전, FFN 전 두 번 사용하고, 모든 block이 끝난 뒤 final norm을 한 번 더 적용합니다. LLaMA 계열 모델과 같은 pre-norm 구조입니다.

## Dense FFN

`use_moe=False`일 때 MLP는 `FeedForward`입니다. 구조는 LLaMA류 SwiGLU입니다.

```text
down_proj(SiLU(gate_proj(x)) * up_proj(x))
```

기본 `hidden_size=768`에서 `intermediate_size`는 다음 식으로 계산됩니다.

```python
math.ceil(hidden_size * math.pi / 64) * 64
```

결과는 `2432`입니다.

## MoE FFN

`--use_moe 1` 또는 `MiniMindConfig(use_moe=True)`를 사용하면 각 block의 FFN이 `MOEFeedForward`로 바뀝니다. Attention 구조는 동일하고, FFN 자리만 MoE로 교체됩니다.

기본 MoE 설정:

| 항목 | 기본값 |
| --- | ---: |
| `num_experts` | `4` |
| `num_experts_per_tok` | `1` |
| `moe_intermediate_size` | `intermediate_size` |
| `norm_topk_prob` | `True` |
| `router_aux_loss_coef` | `5e-4` |

동작 흐름:

```text
x_flat
  -> gate(hidden_size -> num_experts)
  -> softmax
  -> top-k expert 선택
  -> 선택된 expert FFN 실행
  -> top-k weight로 가중합
```

기본값에서는 토큰마다 4개 expert 중 1개만 사용합니다.

MoE 모델의 실제 파라미터 수:

```text
moe total params: 198,416,640
moe active params per token: 63,936,768
```

즉 저장되는 전체 파라미터는 약 `198.4M`이지만, 토큰 하나가 forward에서 활성화하는 파라미터 규모는 약 `63.9M`입니다.

## Aux Loss

MoE에서는 router가 특정 expert만 과하게 쓰는 것을 막기 위해 aux loss를 계산합니다.

```python
load = one_hot(topk_idx).float().mean(0)
aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
```

`MiniMindModel.forward()`는 각 layer의 MoE aux loss를 합산해서 반환합니다.

```text
return hidden_states, presents, aux_loss
```

`MiniMindForCausalLM.forward()`는 이 값을 `MoeCausalLMOutputWithPast`의 `aux_loss`로 넘깁니다.

일반 dense 모델에서는 MoE layer가 없으므로 `aux_loss`는 `0`입니다. 따라서 pretraining 로그에서 `aux_loss=0`이면 보통 `--use_moe 0`으로 dense 모델을 학습 중이라는 뜻입니다.

## Causal LM Loss

`labels`가 주어지면 다음 토큰 예측 cross entropy를 계산합니다.

```text
logits[..., :-1, :] vs labels[..., 1:]
ignore_index = -100
```

학습 스크립트에서는 대체로 다음처럼 총 loss를 구성합니다.

```python
loss = res.loss + res.aux_loss
```

따라서 dense 모델에서는 사실상 CE loss만 학습하고, MoE 모델에서는 CE loss에 router balancing loss가 더해집니다.

## Generate

`MiniMindForCausalLM.generate()`는 직접 구현되어 있습니다.

지원 기능:

```text
KV cache
temperature
top-k sampling
top-p sampling
repetition penalty
eos 종료
streamer 출력
num_return_sequences
```

generation 중에는 `past_key_values`가 있으면 새 token 구간만 forward에 넣어 계산량을 줄입니다.

## Checkpoint 로딩 방식

`trainer/trainer_utils.py`의 `init_model()`은 항상 먼저 새 모델 객체를 만듭니다.

```python
model = MiniMindForCausalLM(lm_config)
```

그 다음 `from_weight != 'none'`이면 `out/` 아래 가중치를 로드합니다.

```text
out/<from_weight>_<hidden_size>.pth
out/<from_weight>_<hidden_size>_moe.pth
```

예:

```text
out/pretrain_768.pth
out/full_sft_768.pth
out/pretrain_moe_768_moe.pth
out/full_sft_moe_768_moe.pth
```

Pretraining 기본값은 `from_weight='none'`입니다. SFT/DPO/Agent는 이전 단계의 weight를 로드해서 이어 학습합니다.

```text
pretrain: scratch -> pretrain
full_sft: pretrain -> full_sft
dpo: full_sft -> dpo
agent: full_sft -> agent
```

## LoRA 코드

`model/model_lora.py`는 간단한 LoRA monkey patch 방식입니다.

`LoRA` 모듈:

```text
A: in_features -> rank
B: rank -> out_features
output: B(A(x))
```

초기화:

```text
A: normal(mean=0, std=0.02)
B: zero
```

`apply_lora()`는 `nn.Linear` 중 `in_features == out_features`인 layer에만 LoRA를 붙입니다. 원래 forward를 감싸서 다음처럼 동작하게 만듭니다.

```python
original_linear(x) + lora(x)
```

주의할 점은 이 방식이 PEFT 스타일 adapter 주입이 아니라 Python monkey patch 방식이라는 것입니다. 그래서 `train_lora.py`에서는 `torch.compile`과의 호환 문제를 피하기 위해 자동으로 compile을 끄는 로직이 있습니다.

## Dense와 MoE 사용 시 주의

Dense와 MoE는 FFN 파라미터 구조가 다릅니다. 따라서 같은 학습 흐름 안에서 `--use_moe` 값을 섞으면 안 됩니다.

올바른 dense 흐름:

```powershell
uv run python trainer/train_pretrain.py --save_weight pretrain --use_moe 0
uv run python trainer/train_full_sft.py --from_weight pretrain --save_weight full_sft --use_moe 0
```

올바른 MoE 흐름:

```powershell
uv run python trainer/train_pretrain.py --save_weight pretrain_moe --use_moe 1
uv run python trainer/train_full_sft.py --from_weight pretrain_moe --save_weight full_sft_moe --use_moe 1
```

MoE 가중치를 추론할 때도 `--use_moe 1`을 붙여야 합니다.

```powershell
uv run python eval_llm.py --weight full_sft_moe --save_dir out --load_from model --use_moe 1
```

## 코드상 특징과 한계

- 구조는 작고 읽기 쉬운 LLaMA-like decoder-only Transformer입니다.
- GQA, RoPE, RMSNorm, SwiGLU, KV cache, tied embedding 등 현대 LLM의 핵심 요소를 최소 구현으로 담고 있습니다.
- `transformers` 생태계와 연결되도록 `PreTrainedModel`, `GenerationMixin`, `PretrainedConfig`를 사용합니다.
- MoE는 FFN 위치에만 적용되며, shared expert는 없습니다.
- LoRA는 간단한 monkey patch 구현이라 범용 PEFT 기능과는 다릅니다.
- 기본 학습은 외부 pretrained weight 없이 scratch에서 시작합니다.
