# minimind-ko

MiniMind-3 원본 프로젝트를 한국어 학습에 맞게 정리한 작업 저장소입니다.

이 저장소에는 MiniMind 모델 코드, 학습 코드, 실행 스크립트와 한국어 JSONL 데이터셋 생성기가 포함되어 있습니다.

## 구성

```text
build_minimind_ko.py   한국어 MiniMind 학습 데이터셋 생성기
MODEL_ANALYSIS.md      MiniMind 모델 구조와 코드 분석 문서
dataset/               MiniMind JSONL Dataset reader
dataset_ko/            생성된 한국어 학습 데이터셋(깃 추적 제외)
model/                 MiniMind 모델 정의와 tokenizer
trainer/               pretrain, SFT, DPO, PPO, GRPO 학습 코드
scripts/               API 서버, 웹 데모, tool-call 평가, 모델 변환 스크립트
eval_llm.py            로컬 추론 및 대화 실행
out/                   학습된 모델 가중치 저장 위치(깃 추적 제외)
checkpoints/           학습 checkpoint 저장 위치(깃 추적 제외)
upstream_minimind/     원본 MiniMind 클론 보관용(깃 추적 제외)
```

모델 백본, dense/MoE 차이, aux loss, checkpoint 로딩 방식은 [MODEL_ANALYSIS.md](MODEL_ANALYSIS.md)에 정리되어 있습니다.

## 준비

```powershell
uv sync
uv pip install -r requirements.txt
```

PyTorch가 설치되어 있는지 확인합니다.

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CUDA 환경이면 본인 CUDA 버전에 맞는 PyTorch를 설치하세요.

## 데이터셋 생성

빠른 테스트용:

```powershell
uv run python build_minimind_ko.py --preset tiny
```

중간 규모:

```powershell
uv run python build_minimind_ko.py --preset mini
```

전체 규모:

```powershell
uv run python build_minimind_ko.py --preset full
```

특정 파일만 다시 생성할 수도 있습니다.

```powershell
uv run python build_minimind_ko.py --preset full --only sft_t2t.jsonl
```

생성 결과는 기본적으로 `dataset_ko/`에 저장됩니다.

```text
pretrain_t2t_mini.jsonl
pretrain_t2t.jsonl
sft_t2t_mini.jsonl
sft_t2t.jsonl
dpo.jsonl
rlaif.jsonl
```

현재 full 생성 기준으로 `sft_t2t.jsonl`은 공개 소스에서 확보 가능한 유효 샘플 수만큼 생성됩니다. 이 환경에서는 약 49만 줄이 생성되었습니다.

## 사용 데이터셋

`build_minimind_ko.py`는 아래 공개 데이터셋을 Hugging Face에서 가져와 MiniMind JSONL 형식으로 변환합니다.

| 용도 | 데이터셋 | 설정 | 비중 | 설명 |
| --- | --- | --- | --- | --- |
| Pretrain | `AdaMLLab/KorMix` | `minhash_deduped` | `1.0` | 한국어 일반 텍스트 pretraining 데이터 |
| SFT | `channelcorp/KoMagpie-raw` | 기본 train split | `0.55` | 한국어 지시/대화 SFT 데이터 |
| SFT | `beomi/KoAlpaca-RealQA` | 기본 train split | `0.20` | 한국어 질의응답 SFT 데이터 |
| SFT | `llami-team/Korean-OpenThoughts-114k-Normalized` | 기본 train split | `0.25` | 한국어 reasoning SFT 데이터 |
| DPO | `maywell/ko_Ultrafeedback_binarized` | 기본 train split | `1.0` | 한국어 선호학습 DPO 데이터 |

추가로 생성되는 데이터는 외부에서 직접 다운로드하지 않고 스크립트 내부에서 만듭니다.

```text
rlaif.jsonl          sft_t2t.jsonl에서 rollout prompt 형태로 샘플링
```

## 기본 학습 순서

먼저 pretraining을 수행합니다.

```powershell
uv run python trainer/train_pretrain.py --data_path dataset_ko/pretrain_t2t.jsonl --save_weight pretrain
```

빠른 동작 확인은 mini 파일로 시작할 수 있습니다.

```powershell
uv run python trainer/train_pretrain.py --data_path dataset_ko/pretrain_t2t_mini.jsonl --save_weight pretrain
```

Pretrain 이후 full SFT를 수행합니다.

```powershell
uv run python trainer/train_full_sft.py --data_path dataset_ko/sft_t2t.jsonl --from_weight pretrain --save_weight full_sft
```

DPO 학습:

```powershell
uv run python trainer/train_dpo.py --data_path dataset_ko/dpo.jsonl --from_weight full_sft --save_weight dpo
```

## MoE 학습

기본값은 dense 모델입니다. MoE 모델로 학습하려면 모든 단계에서 `--use_moe 1`을 붙여야 합니다.

```powershell
uv run python trainer/train_pretrain.py --data_path dataset_ko/pretrain_t2t.jsonl --save_weight pretrain_moe --use_moe 1
uv run python trainer/train_full_sft.py --data_path dataset_ko/sft_t2t.jsonl --from_weight pretrain_moe --save_weight full_sft_moe --use_moe 1
```

MoE로 학습한 가중치는 추론할 때도 `--use_moe 1`이 필요합니다.

```powershell
uv run python eval_llm.py --weight full_sft_moe --save_dir out --load_from model --use_moe 1
```

Dense와 MoE는 가중치 구조가 다르므로 `pretrain`, `full_sft`, `dpo`, 추론 단계에서 같은 구조를 계속 사용해야 합니다.

## 추론

SFT 모델:

```powershell
uv run python eval_llm.py --weight full_sft --save_dir out --load_from model
```

OpenAI 호환 API 서버:

```powershell
uv run python scripts/serve_openai_api.py --weight full_sft --save_dir out --load_from model
```

Streamlit 웹 데모:

```powershell
uv run streamlit run scripts/web_demo.py
```

## 자주 쓰는 옵션

```text
--batch_size             배치 크기
--accumulation_steps     그래디언트 누적 step 수
--epochs                 학습 epoch 수
--learning_rate          학습률
--save_interval          모델 저장 간격
--from_weight            이어서 시작할 가중치 이름
--from_resume 1          checkpoint에서 이어 학습
--use_moe 1              MoE 모델 사용
--use_compile 1          torch.compile 사용
```

GPU 메모리가 부족하면 `--batch_size`를 줄이고 `--accumulation_steps`를 늘리는 방식으로 조정합니다.

```powershell
uv run python trainer/train_pretrain.py --data_path dataset_ko/pretrain_t2t.jsonl --batch_size 8 --accumulation_steps 16
```

## 의존성 주의

`transformers==4.57.6`은 `huggingface-hub < 1.0`이 필요합니다. 이 프로젝트의 `pyproject.toml`은 다음 범위로 맞춰져 있습니다.

```toml
datasets==3.6.0
huggingface-hub>=0.34.0,<1.0
```

버전 충돌이 나면 아래 명령으로 다시 맞출 수 있습니다.

```powershell
uv pip install "huggingface-hub>=0.34.0,<1.0" "datasets==3.6.0" "transformers==4.57.6"
```

## Git 관리

대용량 데이터와 학습 산출물은 `.gitignore`에 포함되어 있습니다.

```text
dataset_ko/
out/
checkpoints/
*.pt
*.pth
*.safetensors
upstream_minimind/
```

코드와 설정 파일만 커밋하는 것을 권장합니다.
