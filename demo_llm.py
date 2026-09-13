import sys
import gc
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
V4_ADAPTER = "output/trend-travel-deepseek-qlora-v4"

LABEL_MAP = {
    "HIGH": "A",
    "MEDIUM": "B",
    "LOW": "C",
}


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# V4 학습과 동일하게 DeepSeek-R1의 generation-only <think> prefix를 제거한다.
reasoning_generation_prefix = "<｜Assistant｜><think>\\n"

if reasoning_generation_prefix not in tokenizer.chat_template:
    raise RuntimeError("DeepSeek-R1 chat template의 <think> 생성 prefix를 찾지 못했습니다.")

tokenizer.chat_template = tokenizer.chat_template.replace(
    reasoning_generation_prefix,
    "<｜Assistant｜>",
)


LABEL_TOKEN_IDS = {}

for actual_label, abc_label in LABEL_MAP.items():
    ids = tokenizer.encode(abc_label, add_special_tokens=False)

    if len(ids) != 1:
        raise RuntimeError(
            f"{abc_label}가 단일 token이 아닙니다: {ids}"
        )

    LABEL_TOKEN_IDS[actual_label] = ids[0]


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def make_prompt(keyword, context):
    text = f"""다음 트렌드의 여행 콘텐츠 활용 가능성을 분류하라.

분류 기준:
A: 특정 지역 방문, 관광, 축제, 명소, 여행 행동과 직접 연결된다.
B: 여행 자체는 아니지만 문화, 음식, 취미, 콘텐츠 등을 통해 자연스럽게 여행으로 확장할 수 있다.
C: 여행과의 자연스러운 연결 근거가 부족하다.

키워드:
{keyword}

문맥:
{context}

A, B, C 중 하나만 답하라.
"""

    messages = [{"role": "user", "content": text}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )


def predict(model, keyword, context):
    prompt = make_prompt(keyword, context)

    inputs = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    labels = ["HIGH", "MEDIUM", "LOW"]

    raw_scores = torch.tensor(
        [
            logits[LABEL_TOKEN_IDS[label]].item()
            for label in labels
        ]
    )

    probs = torch.softmax(raw_scores, dim=0)

    scores = {
        label: probs[i].item()
        for i, label in enumerate(labels)
    }

    prediction = max(scores, key=scores.get)
    return prediction, scores


def print_result(model_name, prediction, scores):
    print("\n====================================")
    print(model_name)
    print("====================================")
    print("판정:", prediction)
    print("\n분류 확률")
    print(f"HIGH   : {scores['HIGH'] * 100:.2f}%")
    print(f"MEDIUM : {scores['MEDIUM'] * 100:.2f}%")
    print(f"LOW    : {scores['LOW'] * 100:.2f}%")


def cleanup():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_original(keyword, context):
    print("\nOriginal DeepSeek 로딩...")

    model = load_base()
    model.eval()

    prediction, scores = predict(model, keyword, context)

    print_result(
        "Original DeepSeek",
        prediction,
        scores,
    )

    del model
    cleanup()


def run_v4(keyword, context):
    print("\nQLoRA V4 로딩...")

    base_model = load_base()

    model = PeftModel.from_pretrained(
        base_model,
        V4_ADAPTER,
    )

    model.eval()

    prediction, scores = predict(model, keyword, context)

    print_result(
        "Fine-tuned QLoRA V4",
        prediction,
        scores,
    )

    del model
    del base_model
    cleanup()


def main():
    if len(sys.argv) < 2:
        print(
            """
사용법:

Original 모델:
python demo_llm.py base

Fine-tuned V4:
python demo_llm.py v4

두 모델 비교:
python demo_llm.py compare
"""
        )
        return

    mode = sys.argv[1].lower()

    print("\n====================================")
    print("Travel Trend LLM Demo")
    print("====================================")

    keyword = input("\n트렌드 키워드 입력: ").strip()
    context = input("문맥 입력: ").strip()

    if not keyword:
        print("키워드를 입력해야 합니다.")
        return

    if mode == "base":
        run_original(keyword, context)
    elif mode == "v4":
        run_v4(keyword, context)
    elif mode == "compare":
        run_original(keyword, context)
        run_v4(keyword, context)
    else:
        print("base / v4 / compare 중 하나를 입력하세요.")


if __name__ == "__main__":
    main()
