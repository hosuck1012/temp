# trend-travel-deepseek-qlora-v4

Final QLoRA adapter metadata for the travel-trend classification experiment.

- Base model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- PEFT type: LoRA
- Rank: 8
- Alpha: 16
- Dropout: 0.05
- Target modules: `q_proj`, `v_proj`
- Labels: `A=HIGH`, `B=MEDIUM`, `C=LOW`

## Adapter weights

The trained local artifact is named `adapter_model.safetensors`.

- Local file size: 4,372,840 bytes (~4.2 MiB)
- SHA-256: `a33e8e10c632635c8a58d20210d385050d1c1466b8a895a90aae4a2594218d69`

The binary weight file is intentionally not embedded in this source-oriented portfolio snapshot. The repository preserves the exact training/evaluation code, adapter configuration, modified chat template, experiment history, and metrics. To reproduce a compatible adapter, run `train_qlora_v4.py` with the documented dataset and environment.

The base model weights are also not committed; they are downloaded from Hugging Face at runtime.
