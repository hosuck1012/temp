# Travel Trend LLM Fine-Tuning

DeepSeek-R1-Distill-Qwen-1.5B를 기반으로 한국어 트렌드 키워드와 문맥을 입력받아 여행 콘텐츠 활용 가능성을 `HIGH / MEDIUM / LOW`로 분류하도록 QLoRA 파인튜닝한 실험 프로젝트입니다.

## 목표

트렌드 후보가 실제 여행 콘텐츠로 이어질 수 있는지 로컬 LLM이 1차 판정하도록 만드는 것이 목표입니다.

- `HIGH`: 특정 지역 방문, 관광지, 축제, 명소, 여행 행동과 직접 연결
- `MEDIUM`: 음식·문화·취미·라이프스타일 등을 통해 여행으로 자연스럽게 확장 가능
- `LOW`: 여행과의 자연스러운 연결 근거가 약함

## 환경

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- GPU: NVIDIA RTX A1000 Laptop GPU 4GB
- Method: 4-bit QLoRA
- Quantization: NF4 + double quantization
- LoRA: `r=8`, `alpha=16`, `dropout=0.05`
- Target modules: `q_proj`, `v_proj`
- Epochs: 3
- Batch size: 1
- Gradient accumulation: 4
- Learning rate: `5e-5`
- Max length: 256

## 데이터

현재 데이터셋은 실제 최신 트렌드의 정답 데이터가 아니라 분류 구조와 파인튜닝 과정을 검증하기 위해 만든 합성 데이터입니다.

- Train: 300개
- Validation: 60개
- Test: 60개
- 각 분할은 HIGH / MEDIUM / LOW가 균형을 이루도록 구성

따라서 아래 성능은 합성 테스트셋에 대한 controlled benchmark이며, 실제 최신 트렌드에서 동일한 정확도를 보장하지 않습니다.

## 시행착오

### V1
15개 소규모 데이터로 QLoRA를 적용했지만 성능 개선이 확인되지 않았습니다.

### V2
`label + reason + travel_angle`을 JSON으로 함께 생성하도록 학습했습니다. 그러나 분류보다 긴 설명 생성이 loss의 대부분을 차지했고, 예측이 `MEDIUM`으로 편향되는 문제가 발생했습니다.

### V3
출력을 `HIGH / MEDIUM / LOW` 단일 라벨로 줄였습니다. 이후 DeepSeek-R1의 chat template이 generation prompt에만 `<think>` prefix를 추가하면서 TRL의 completion 경계가 어긋나는 문제를 발견했습니다.

### V4
최종적으로 다음을 적용했습니다.

1. DeepSeek-R1 chat template에서 generation-only `<think>` prefix 제거
2. `completion_only_loss=True`로 정답 토큰만 학습
3. 라벨을 단일 토큰 `A / B / C`로 변경
   - A = HIGH
   - B = MEDIUM
   - C = LOW
4. 평가에서도 다음 토큰의 A/B/C logits만 직접 비교

## 최종 결과

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Original DeepSeek | 0.3333 | 0.1667 |
| Fine-tuned QLoRA V4 | **0.9000** | **0.9008** |

V4 confusion matrix:

```text
              Predicted
             H   M   L
Actual H    16   4   0
Actual M     0  19   1
Actual L     0   1  19
```

60개 중 54개를 맞혔고, `HIGH → LOW`, `LOW → HIGH`처럼 극단적으로 반대되는 오분류는 없었습니다.

## 주요 파일

- `train_qlora_v4.py`: 최종 QLoRA 학습 코드
- `evaluate_v4.py`: Original vs V4 평가
- `demo_llm.py`: 입력한 키워드/문맥을 Original과 V4로 비교하는 데모
- `data/`: 합성 train/validation/test 데이터셋
- `docs/experiment_history.md`: V1~V4 실험 과정과 디버깅 기록

## 실행 개요

```powershell
python train_qlora_v4.py
python evaluate_v4.py
python demo_llm.py compare
```

원본 모델 가중치는 저장소에 포함하지 않습니다. Hugging Face에서 다시 내려받도록 코드에 모델 이름을 명시했습니다.

## 한계 및 다음 단계

- 합성 데이터 기반 평가이므로 실제 트렌드 일반화 성능은 아직 검증되지 않음
- 1.5B 모델과 4GB VRAM 환경 때문에 더 큰 모델 및 Full Fine-tuning을 사용하지 못함
- 실제 뉴스/YouTube 기반 트렌드 샘플을 사람이 라벨링한 독립 테스트셋으로 검증할 필요가 있음
- 향후 Trend Catch Module과 연동하여 로컬 여행 연관성 판정기로 활용 예정
