import json
import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

TRAIN_PATH = "data/train_dataset.jsonl"
VAL_PATH = "data/validation_dataset.jsonl"

OUTPUT_DIR = "output/trend-travel-deepseek-qlora-v4"

MAX_LENGTH = 256


# =========================================================
# 1. GPU 확인
# =========================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU를 사용할 수 없습니다.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2,
    ),
    "GB",
)


# =========================================================
# 2. Tokenizer
# =========================================================

print("\nTokenizer 로딩...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# DeepSeek-R1의 기본 chat template은 생성 prompt 끝에만 <think>를 붙인다.
# prompt-completion 전체를 렌더링했을 때는 이 토큰이 없어 TRL이 completion
# 경계를 찾지 못하므로, V4의 단일 라벨 학습에 맞게 reasoning prefix를 제거한다.
reasoning_generation_prefix = "<｜Assistant｜><think>\\n"

if reasoning_generation_prefix not in tokenizer.chat_template:
    raise RuntimeError("DeepSeek-R1 chat template의 <think> 생성 prefix를 찾지 못했습니다.")

tokenizer.chat_template = tokenizer.chat_template.replace(
    reasoning_generation_prefix,
    "<｜Assistant｜>",
)


# =========================================================
# 3. Dataset
# =========================================================

print("\nDataset 로딩...")

train_dataset = load_dataset(
    "json",
    data_files=TRAIN_PATH,
    split="train",
)

val_dataset = load_dataset(
    "json",
    data_files=VAL_PATH,
    split="train",
)

print("변환 전 Train:", len(train_dataset))
print("변환 전 Validation:", len(val_dataset))


# =========================================================
# 4. Prompt / Completion 형태로 변환
# =========================================================


def convert_example(example):

    label_map = {
        "HIGH": "A",
        "MEDIUM": "B",
        "LOW": "C",
    }

    prompt_text = f"""다음 트렌드의 여행 콘텐츠 활용 가능성을 분류하라.

분류 기준:
A: 특정 지역 방문, 관광, 축제, 명소, 여행 행동과 직접 연결된다.
B: 여행 자체는 아니지만 문화, 음식, 취미, 콘텐츠 등을 통해 자연스럽게 여행으로 확장할 수 있다.
C: 여행과의 자연스러운 연결 근거가 부족하다.

키워드:
{example["keyword"]}

문맥:
{example["context"]}

A, B, C 중 하나만 답하라.
"""

    return {
        "prompt": [
            {
                "role": "user",
                "content": prompt_text,
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": label_map[example["label"]],
            }
        ],
    }

remove_columns = train_dataset.column_names

train_dataset = train_dataset.map(
    convert_example,
    remove_columns=remove_columns,
)

val_dataset = val_dataset.map(
    convert_example,
    remove_columns=val_dataset.column_names,
)

print("\n변환 후 Train:", len(train_dataset))
print("변환 후 Validation:", len(val_dataset))
print("변환 후 Train columns:", train_dataset.column_names)
print("변환 후 Validation columns:", val_dataset.column_names)

print("\n첫 번째 Train 데이터:")
print(train_dataset[0])


def validate_converted_dataset(dataset, name):

    if dataset.column_names != ["prompt", "completion"]:
        raise ValueError(
            f"{name} columns가 prompt/completion이 아닙니다: {dataset.column_names}"
        )

    invalid_completions = [
        example["completion"]
        for example in dataset
        if (
            len(example["completion"]) != 1
            or example["completion"][0].get("role") != "assistant"
            or example["completion"][0].get("content") not in {"A", "B", "C"}
        )
    ]

    if invalid_completions:
        raise ValueError(
            f"{name}에 A/B/C가 아닌 completion이 있습니다: "
            f"{invalid_completions[:3]}"
        )

    completion_labels = sorted(
        {example["completion"][0]["content"] for example in dataset}
    )
    print(f"{name} completion labels:", completion_labels)

    sequence_lengths = []
    completion_token_lengths = []

    for example in dataset:
        prompt_ids = tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        full_ids = tokenizer.apply_chat_template(
            example["prompt"] + example["completion"],
            tokenize=True,
            return_dict=True,
        )["input_ids"]

        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(f"{name}의 prompt/completion 토큰 경계가 일치하지 않습니다.")

        completion_token_length = len(full_ids) - len(prompt_ids)

        if completion_token_length <= 0:
            raise ValueError(f"{name}의 completion 토큰이 비어 있습니다.")

        sequence_lengths.append(len(full_ids))
        completion_token_lengths.append(completion_token_length)

    print(
        f"{name} token length: min={min(sequence_lengths)}, "
        f"max={max(sequence_lengths)}"
    )
    print(
        f"{name} completion token length: "
        f"min={min(completion_token_lengths)}, max={max(completion_token_lengths)}"
    )

    if max(sequence_lengths) > MAX_LENGTH:
        raise ValueError(
            f"{name}의 최대 길이 {max(sequence_lengths)}가 "
            f"max_length={MAX_LENGTH}를 초과합니다."
        )


validate_converted_dataset(train_dataset, "Train")
validate_converted_dataset(val_dataset, "Validation")

# =========================================================
# 5. 4-bit QLoRA
# =========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,
)


# =========================================================
# 6. LoRA 설정
# =========================================================

lora_config = LoraConfig(
    r=8,

    lora_alpha=16,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM",

    # 4GB VRAM이므로 우선 Q/V만
    target_modules=[
        "q_proj",
        "v_proj",
    ],
)


# =========================================================
# 7. Training 설정
# =========================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    # V2 2e-4보다 낮춤
    learning_rate=5e-5,

    gradient_checkpointing=True,

    fp16=True,
    bf16=False,

    model_init_kwargs={
        "dtype": torch.float16,
    },

    # 출력은 매우 짧지만 prompt 때문에 어느 정도 확보
    max_length=MAX_LENGTH,

    # 정답 A/B/C 부분만 loss 계산
    completion_only_loss=True,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",
    greater_is_better=False,

    save_total_limit=2,

    logging_steps=10,
    report_to="none",
)

# =========================================================
# 8. Trainer
# =========================================================

print("\nQLoRA Trainer 생성...")

trainer = SFTTrainer(
    model=MODEL_NAME,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    # 위에서 단일 라벨용으로 수정한 chat template을 Trainer에도 전달
    processing_class=tokenizer,

    quantization_config=bnb_config,

    peft_config=lora_config,
)


# 이전에 발생한 BF16 Gradient 문제 방지
for name, param in trainer.model.named_parameters():

    if param.requires_grad:
        param.data = param.data.float()


trainer.model.print_trainable_parameters()

print("Trainer Train dataset:", len(trainer.train_dataset))
print("Trainer Eval dataset:", len(trainer.eval_dataset))

if len(trainer.train_dataset) == 0:
    raise RuntimeError("Trainer 전처리 후 train dataset이 0개입니다.")

if len(trainer.eval_dataset) == 0:
    raise RuntimeError("Trainer 전처리 후 validation dataset이 0개입니다.")


# =========================================================
# 9. 학습
# =========================================================

print("\n==============================")
print("QLoRA V4 학습 시작")
print("==============================")

trainer.train()


# =========================================================
# 10. Best Adapter 저장
# =========================================================

print("\nBest Adapter 저장...")

trainer.model.save_pretrained(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)


print("\n==============================")
print("학습 완료")
print("==============================")

print("저장 위치:", OUTPUT_DIR)
print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best validation metric:", trainer.state.best_metric)
