# Small Language Models for Financial Question Answering

## Abstract

> Large language models (LLMs) have demonstrated strong performance in financial reasoning, but their computational demands limit their applicability in resource-constrained environments. This work investigates the performance of **small-scale LLMs (1B–2B parameters)** on the [FinanceBench dataset](https://huggingface.co/datasets/PatronusAI/financebench), a benchmark of 150 annotated financial questions spanning **metrics-generated**, **domain-relevant**, and **novel-generated** categories. We evaluate nine models from Hugging Face and GUFF under multiple reasoning structures, including **zero-shot**, **context-enhanced zero-shot**, **retrieval-augmented generation (RAG)**, and **fine-tuning**. Results show that fine-tuned **Gemma-2 2B** achieved the highest accuracy of **25%**, outperforming all other small models across methodologies. In contrast, GUFF models such as **StableLM 1.6B** and **LLaMA 3.2–1B Instruct** achieved peak accuracies of **13%** (zero-shot + RAG) and **11%** (zero-shot + context), respectively. Notably, **fine-tuned RAG underperformed zero-shot RAG**, highlighting that fine-tuning is not always beneficial in retrieval-augmented settings. These findings suggest that while small LLMs hold promise for financial reasoning tasks in constrained environments, performance remains limited and highly sensitive to both model architecture and evaluation methodology.

---

## Methodology

### Dataset

* **FinanceBench** (150 examples): domain-relevant, metrics-generated, and novel-generated financial questions.
* Open-source and publicly available [here](https://huggingface.co/datasets/PatronusAI/financebench).

### Workflow

Models were fine-tuned, evaluated, and compared across multiple methodologies.
![Pipeline Workflow](https://github.com/aisha1021/llms-financebench/blob/d7b71ba2ed2f26e0b6caf1e2888bae8e572c2b28/llms-pipeline-diagram.png)

### How to Run

```bash
pip install -r requirements.txt
python scripts/startup.py
python -m scripts.hf.run_gemma_finetune  # replace `hf` with `guff` and code file of interest
```

---

## Models

### Hugging Face Models

* **Gemma-2 2B IT** (`gemma-2-2b-it`)
* **Phi 2.0** (`phi-2`)
* **TinyLLaMA 1.1B** (`TinyLLaMA 1.1B`)
* **DeepSeek Coder 1.3B** (`deepseek-coder-1.3b-base`)

Applied Structures:

* Zero-shot
* Zero-shot + Context
* Zero-shot + RAG
* Fine-tuned + Context
* Fine-tuned + RAG

### GUFF Models

* **BLING 1B** (`BLING-1B`)
* **LLaMA 3.2–1B Instruct** (`LLaMA 3.2–1B Instruct`)
* **Phi 2.0** (`phi-2`)
* **StableLM 1.6B** (`StableLM 1.6B`)
* **TinyLLaMA 1.1B OpenOrca** (`TinyLLaMA 1.1B OpenOrca`)

Applied Structures:

* Zero-shot
* Zero-shot + Context
* Zero-shot + RAG

---

## Results

### Hugging Face Models

* Fine-tuned Gemma 2B achieved **25% accuracy**, the highest among tested models.
* Gemma consistently outperformed other small models across all structures.
* Fine-tuned RAG performed worse than zero-shot RAG, suggesting that fine-tuning does not always improve retrieval-augmented performance.

![HF Results](https://github.com/aisha1021/llms-financebench/blob/78ef6ec6e03c638fb3ab50ea2d004b4dd626e725/hf_model_results.png)

### GUFF Models

* LLaMA 3.2–1B Instruct performed best with **\~11% accuracy** using zero-shot + context.
* StableLM 1.6B outperformed others with **\~13% accuracy** under zero-shot + RAG.
* Performance differences suggest model-specific strengths depending on methodology.

![GUFF Results](https://github.com/aisha1021/llms-financebench/blob/195ac22e58ba8bb32ef16981e3d66397a1b58fb9/guff_model_results.png)

---

## Analysis

* **Smaller models show potential** in financial reasoning, though accuracy remains limited (<25%).
* **Context-based prompting** generally improves performance, but effectiveness varies by model.
* **Fine-tuning trade-offs**: While beneficial for Gemma, it degraded performance for RAG-based approaches.
* **Model variability** indicates that architecture and training data influence how well models leverage context or retrieval.

---

## Future Work

* Extend evaluation to larger datasets and real-world financial reports.
* Experiment with hybrid methods (LoRA fine-tuning + RAG).
* Explore instruction-tuned small models optimized for reasoning tasks.
