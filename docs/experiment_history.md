# Experiment History

## 프로젝트 목표

`트렌드 키워드 + 문맥`을 입력받아 여행 콘텐츠 활용 가능성을 `HIGH / MEDIUM / LOW`로 분류하는 로컬 LLM을 만드는 실험입니다.

## V1 — 15개 샘플 QLoRA

- Base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Train: 15개
- 4-bit NF4 QLoRA
- LoRA target: `q_proj`, `v_proj`
- 결과: 의미 있는 성능 개선을 확인하지 못함

작은 데이터셋만으로는 원하는 분류 기준을 충분히 학습시키기 어렵다는 점을 확인했습니다.

## V2 — 300개 + JSON 생성

출력을 다음처럼 설계했습니다.

```json
{
  "label": "HIGH",
  "reason": "...",
  "travel_angle": "..."
}
```

Train 300 / Validation 60으로 늘렸지만, 분류 라벨보다 reason/travel_angle의 긴 문장 생성이 loss의 큰 비중을 차지했습니다. 평가 결과 예측이 `MEDIUM` 쪽으로 심하게 편향되었습니다.

## V3 — 분류 전용 출력

출력을 `HIGH / MEDIUM / LOW` 하나로 줄여 분류 자체에 집중하도록 변경했습니다.

이 과정에서 DeepSeek-R1 계열 chat template의 generation prompt가 `<think>` prefix를 추가한다는 문제를 발견했습니다. prompt-completion 전체 렌더링과 generation prompt의 경계가 달라져 TRL의 `completion_only_loss=True`에서 completion label이 모두 `-100`으로 masking되고, Trainer가 학습 샘플을 제거하는 문제가 발생했습니다.

또한 라벨 토큰화 결과가 서로 달랐습니다.

```text
HIGH   -> 1 token
MEDIUM -> 2 tokens
LOW    -> 1 token
```

따라서 문자열 전체 likelihood 비교가 라벨 자체의 토큰화 구조에 영향을 받을 수 있음을 확인했습니다.

## V4 — 최종 실험

최종적으로 다음을 적용했습니다.

### 1. A/B/C 단일 토큰 분류

```text
A = HIGH
B = MEDIUM
C = LOW
```

A/B/C가 각각 단일 토큰인지 검증한 뒤 학습했습니다.

### 2. DeepSeek `<think>` generation prefix 수정

기본 chat template의 generation-only reasoning prefix를 제거하여 prompt와 completion 경계를 일치시켰습니다.

### 3. Completion-only loss

모델이 프롬프트 문장을 외우는 것이 아니라 정답 A/B/C 토큰에만 loss를 계산하도록 구성했습니다.

### 4. QLoRA 설정

- 4-bit NF4
- double quantization
- LoRA r=8
- alpha=16
- dropout=0.05
- target: q_proj, v_proj
- epochs=3
- batch size=1
- gradient accumulation=4
- learning rate=5e-5
- max length=256

실제 데이터 최대 길이는 Train 186 / Validation 187 token으로 확인했습니다.

## 최종 평가

동일한 60개 합성 Test Set 기준:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Original DeepSeek | 0.3333 | 0.1667 |
| QLoRA V4 | **0.9000** | **0.9008** |

V4 confusion matrix:

```text
              Predicted
             H   M   L
Actual H    16   4   0
Actual M     0  19   1
Actual L     0   1  19
```

## 해석 시 주의점

이 결과는 실제 최신 트렌드 데이터가 아니라 규칙 기반으로 만든 합성 데이터셋에 대한 controlled benchmark입니다. 따라서 `실전 정확도 90%`라고 일반화하면 안 됩니다. 현재 결과가 보여주는 것은 작은 1.5B 기반 모델도 올바른 출력 설계, loss masking, 라벨 토큰 설계와 QLoRA를 통해 정의된 분류 규칙을 학습할 수 있다는 점입니다.

다음 검증 단계는 학습 데이터 생성 방식과 독립적인 실제 한국 트렌드 샘플을 사람이 라벨링하여 generalization 성능을 측정하는 것입니다.
