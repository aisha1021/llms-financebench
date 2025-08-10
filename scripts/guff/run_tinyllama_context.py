import os
import re
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from llama_cpp import Llama
from tqdm import tqdm
import numpy as np
from scipy.stats import pointbiserialr, bootstrap
from sklearn.metrics import precision_score, recall_score, f1_score

# -----------------------------
# Load FinanceBench dataset
# -----------------------------
df = load_dataset("PatronusAI/financebench", split="train").to_pandas()
df = df[df['question_type'].isin(['metrics-generated', 'novel-generated', 'domain-relevant'])].copy()
df = df.rename(columns={'answer': 'actual_answer'})

def extract_numeric(x):
    if pd.isna(x): return None
    x_clean = re.sub(r'[^\d\.\-\+eE]', '', str(x))
    try: return float(x_clean)
    except: return None

df['actual_answer_num'] = df.apply(
    lambda row: extract_numeric(row.actual_answer) if row.question_type == 'metrics-generated' else None,
    axis=1
)
df = df.dropna(subset=['evidence', 'actual_answer']).reset_index(drop=True)

# -----------------------------
# Prompt templates
# -----------------------------
prompt_numeric = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Extract the exact numeric value that answers the question from the context below.\n"
        "Respond with only the number and no explanation.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
)

prompt_semantic = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Using the context below, answer the question in 1–2 clear sentences.\n"
        "Avoid generic explanations. Start your answer directly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
)

# -----------------------------
# GGUF model paths
# -----------------------------
gguf_models = {
    # "LLaMA 3.2–1B Instruct": "/kaggle/input/models/Llama-3.2-1B-Instruct-Q6_K_L.gguf",
    "TinyLLaMA 1.1B OpenOrca": "/kaggle/input/models/tinyllama-1.1b-1t-openorca.Q5_K_M.gguf",
    # "BLING-1B": "/kaggle/input/models/bling-1b-0.1.Q4_K_M.gguf",
    # "Phi-2": "/kaggle/input/models/phi-2.Q4_K_M.gguf",
    # "StableLM 1.6B": "/kaggle/input/models/stablelm-2-1_6b-chat.Q4_K_M.imx.gguf"
}

# -----------------------------
# Utility functions
# -----------------------------
semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_answer(decoded):
    decoded = decoded.replace(",", "")
    nums = re.findall(r'-?\d+(?:\.\d+)?', decoded)
    return nums[-1] if nums else decoded.strip()

def numeric_match(pred, actual, tol=0.05):
    try:
        pred_val = float(re.findall(r'-?\d+\.?\d*', pred)[-1])
        return abs(pred_val - actual) / (abs(actual) + 1e-6) <= tol
    except:
        return False

def semantic_match(pred, actual_text, threshold=0.80):
    emb_p = semantic_model.encode(pred, convert_to_tensor=True)
    emb_a = semantic_model.encode(actual_text, convert_to_tensor=True)
    score = util.cos_sim(emb_p, emb_a).item()
    return score, score >= threshold

# -----------------------------
# Inference function with llama.cpp
# -----------------------------
def run_model(gguf_path, model_name):
    CONTEXT_LIMIT = 2048
    print(f"\n Loading model: {model_name}")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=CONTEXT_LIMIT,
        n_gpu_layers=1,
        f16_kv=True,
        verbose=False
    )

    results, errors = [], []

    for row in tqdm(df.itertuples(), total=len(df), desc=f"Evaluating {model_name}"):
        context = row.evidence
        prompt = (
            prompt_numeric if row.question_type == "metrics-generated"
            else prompt_semantic
        ).format(context=context, question=row.question)

        prompt_tokens = len(llm.tokenize(prompt.encode("utf-8")))
        if prompt_tokens > CONTEXT_LIMIT - 128:
            excess_tokens = prompt_tokens - (CONTEXT_LIMIT - 128)
            context_words = str(context).split()
            truncated_context = " ".join(context_words[len(context_words) - max(10, len(context_words) - excess_tokens):])
            prompt = (
                prompt_numeric if row.question_type == "metrics-generated"
                else prompt_semantic
            ).format(context=truncated_context, question=row.question)

        output = llm(prompt, max_tokens=128, stop=["\n", "Question:", "Context:"])
        decoded = output["choices"][0]["text"].strip()
        print(decoded)

        if row.question_type == "metrics-generated":
            pred = extract_answer(decoded)
            print(pred)
            correct = numeric_match(pred, row.actual_answer_num)
            score = None
        else:
            pred = decoded
            score, correct = semantic_match(pred, row.actual_answer)

        results.append({
            "id": row.financebench_id,
            "type": row.question_type,
            "prediction": pred,
            "actual": row.actual_answer,
            "numeric_correct": correct if row.question_type == "metrics-generated" else None,
            "semantic_score": score if row.question_type != "metrics-generated" else None,
            "semantic_correct": correct if row.question_type != "metrics-generated" else None
        })

        if not correct:
            errors.append({
                "id": row.financebench_id,
                "question": row.question,
                "prediction": pred,
                "actual": row.actual_answer,
                "context": row.evidence,
                "error_type": "numeric_mismatch" if row.question_type == "metrics-generated" else "semantic_mismatch",
                "semantic_score": score
            })

    df_results = pd.DataFrame(results)
    df_errors = pd.DataFrame(errors)

    print(f"\n Results for {model_name}")
    print(f"  - Numeric accuracy: {df_results[df_results.type == 'metrics-generated'].numeric_correct.mean():.4f}")
    print(f"  - Semantic accuracy: {df_results[df_results.type != 'metrics-generated'].semantic_correct.mean():.4f}")
    print(f"  - Total errors: {len(df_errors)}")

    return df_results, df_errors

# -----------------------------
# Run all models
# -----------------------------
all_results = {}

for name, path in gguf_models.items():
    res, err = run_model(path, name)
    all_results[name] = {"results": res, "errors": err}

def ci_proportion(successes, total, z=1.96):
    p = successes / total
    err = z * np.sqrt(p * (1 - p) / total)
    return p, (p - err, p + err)

for model_name, model_data in all_results.items():
    print(f"\n=== Summary for {model_name} ===")
    
    res_df = model_data["results"]
    metrics_df = res_df[res_df.type == 'metrics-generated']
    semantic_df = res_df[res_df.type.isin(['novel-generated', 'domain-relevant'])]

    num_acc, num_ci = ci_proportion(metrics_df.numeric_correct.sum(), len(metrics_df))
    sem_acc, sem_ci = ci_proportion(semantic_df.semantic_correct.sum(), len(semantic_df))

    semantic_y_true = semantic_df.semantic_correct.astype(int)
    semantic_y_pred = (semantic_df.semantic_score > 0.80).astype(int)

    f1 = f1_score(semantic_y_true, semantic_y_pred)
    prec = precision_score(semantic_y_true, semantic_y_pred)
    rec = recall_score(semantic_y_true, semantic_y_pred)

    corr, p_val = pointbiserialr(semantic_y_true, semantic_df.semantic_score)

    boot_result = bootstrap(
        (semantic_df.semantic_score.dropna().values,),
        np.mean,
        confidence_level=0.95,
        n_resamples=1000,
        method="basic"
    )

    print(f"Numeric Accuracy: {num_acc:.3f} | 95% CI: {num_ci}")
    print(f"Semantic Accuracy: {sem_acc:.3f} | 95% CI: {sem_ci}")
    print(f"F1 Score: {f1:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")
    print(f"Semantic Score-Correctness Correlation: {corr:.3f} (p={p_val:.4f})")
    print(f"Bootstrapped CI for Mean Semantic Score: {boot_result.confidence_interval}")

per_model_res = {}
per_model_err = {}

for model_name, model_data in all_results.items():
    res_df = model_data["results"].copy()
    err_df = model_data["errors"].copy()

    res_df["model"] = model_name
    res_df["run_type"] = "base_model"

    err_df["model"] = model_name
    err_df["run_type"] = "base_model"

    per_model_res[model_name] = res_df
    per_model_err[model_name] = err_df

combined_results_df = pd.concat(per_model_res.values(), ignore_index=True)
combined_errors_df = pd.concat(per_model_err.values(), ignore_index=True)
combined_results_df.to_csv("results/guff_tinyllama-1.1b-1t-openorca.Q5_K_M_results.csv", index=False)
combined_errors_df.to_csv("results/guff_tinyllama-1.1b-1t-openorca.Q5_K_M_errors.csv", index=False)
