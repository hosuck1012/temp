import gc
import json
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel


MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
TEST_PATH = "data/test_dataset.jsonl"
V4_ADAPTER = "output/trend-travel-deepseek-qlora-v4"
LABELS = ["HIGH", "MEDIUM", "LOW"]


print("\nTokenizer 로딩...")

tokenizer = AutoTokenizer.from_pretrained(V4_ADAPTER)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("\n===== Label Tokenization =====")

ABC_TOKEN_IDS = {}

for label in ["A", "B", "C"]:
    ids = tokenizer.encode(label, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(label, "->", ids, tokens)

    if len(ids) != 1:
        raise RuntimeError(
            f"{label}가 단일 토큰이 아닙니다. token ids = {ids}"
        )

    ABC_TOKEN_IDS[label] = ids[0]


LABEL_TOKEN_IDS = {
    "HIGH": ABC_TOKEN_IDS["A"],
    "MEDIUM": ABC_TOKEN_IDS["B"],
    "LOW": ABC_TOKEN_IDS["C"],
}

print("\n===== Classification Mapping =====")
print("HIGH   -> A ->", LABEL_TOKEN_IDS["HIGH"])
print("MEDIUM -> B ->", LABEL_TOKEN_IDS["MEDIUM"])
print("LOW    -> C ->", LABEL_TOKEN_IDS["LOW"])


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)


def load_test_data():
    data = []

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    return data


def make_prompt(item):
    text = f"""다음 트렌드의 여행 콘텐츠 활용 가능성을 분류하라.

분류 기준:
A: 특정 지역 방문, 관광, 축제, 명소, 여행 행동과 직접 연결된다.
B: 여행 자체는 아니지만 문화, 음식, 취미, 콘텐츠 등을 통해 자연스럽게 여행으로 확장할 수 있다.
C: 여행과의 자연스러운 연결 근거가 부족하다.

키워드:
{item["keyword"]}

문맥:
{item["context"]}

A, B, C 중 하나만 답하라.
"""

    messages = [{"role": "user", "content": text}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def check_prompt(dataset):
    if not dataset:
        raise RuntimeError("Test dataset이 비어 있습니다.")

    sample_prompt = make_prompt(dataset[0])

    print("\n===== Prompt Check =====")
    print(sample_prompt)

    if sample_prompt.rstrip().endswith("<think>"):
        raise RuntimeError(
            "V4 tokenizer의 generation prompt 끝에 <think>가 남아 있습니다."
        )

    print("\nPrompt 검증 완료")


def predict(model, item):
    prompt = make_prompt(item)

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

    next_token_logits = outputs.logits[0, -1, :]

    scores = {
        label: next_token_logits[token_id].item()
        for label, token_id in LABEL_TOKEN_IDS.items()
    }

    prediction = max(scores, key=scores.get)
    return prediction, scores


def evaluate(model, dataset, name):
    model.eval()

    y_true = []
    y_pred = []

    print("\n========================================")
    print(name)
    print("========================================")

    for i, item in enumerate(dataset, start=1):
        prediction, scores = predict(model, item)
        expected = item["label"]

        y_true.append(expected)
        y_pred.append(prediction)

        print(
            f"[{i:02}/{len(dataset)}] "
            f"{item['keyword']} | "
            f"정답={expected} | "
            f"예측={prediction}"
        )

        print({key: round(value, 4) for key, value in scores.items()})

    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )

    print("\nAccuracy:")
    print(round(accuracy, 4))

    print("\nMacro F1:")
    print(round(macro_f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            digits=4,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix")
    print("행 = 실제 / 열 = 예측")
    print("순서 =", LABELS)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print(cm)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )


def cleanup_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


test_data = load_test_data()

print("\nTest 데이터:", len(test_data))

if len(test_data) == 0:
    raise RuntimeError("Test dataset이 비어 있습니다.")

check_prompt(test_data)


print("\nOriginal 모델 로딩...")

base_model = load_base()

original_result = evaluate(
    base_model,
    test_data,
    "Original DeepSeek - A/B/C Classification",
)

del base_model
cleanup_gpu()

print("\nOriginal 모델 메모리 해제 완료")


print("\nV4 모델 로딩...")

base_model = load_base()

v4_model = PeftModel.from_pretrained(
    base_model,
    V4_ADAPTER,
)

v4_result = evaluate(
    v4_model,
    test_data,
    "QLoRA V4 - A/B/C Classification",
)

del v4_model
del base_model
cleanup_gpu()

print("\nV4 모델 메모리 해제 완료")


print("\n========================================")
print("최종 성능 비교")
print("========================================")
print(f"Original Accuracy : {original_result['accuracy']:.4f}")
print(f"V4 Accuracy       : {v4_result['accuracy']:.4f}")
print()
print(f"Original Macro F1 : {original_result['macro_f1']:.4f}")
print(f"V4 Macro F1       : {v4_result['macro_f1']:.4f}")


accuracy_diff = v4_result["accuracy"] - original_result["accuracy"]
f1_diff = v4_result["macro_f1"] - original_result["macro_f1"]

print("\n========================================")
print("V4 개선량")
print("========================================")
print(f"Accuracy 변화 : {accuracy_diff:+.4f}")
print(f"Macro F1 변화 : {f1_diff:+.4f}")


print("\n========================================")
print("결과 판정")
print("========================================")

if (
    v4_result["accuracy"] > original_result["accuracy"]
    and v4_result["macro_f1"] > original_result["macro_f1"]
):
    print("V4가 Original보다 Accuracy와 Macro F1 모두 향상되었습니다.")
elif v4_result["macro_f1"] > original_result["macro_f1"]:
    print("V4의 Macro F1은 향상되었지만 Accuracy는 향상되지 않았습니다.")
elif v4_result["accuracy"] > original_result["accuracy"]:
    print("V4의 Accuracy는 향상되었지만 Macro F1은 향상되지 않았습니다.")
else:
    print("V4가 Original보다 명확한 성능 향상을 보이지 못했습니다.")
