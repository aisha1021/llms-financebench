import re
import pandas as pd
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

df = load_dataset("PatronusAI/financebench", split="train").to_pandas()
df = df[df['question_type'].isin(['metrics-generated', 'novel-generated', 'domain-relevant'])].copy()
df = df.rename(columns={'answer': 'actual_answer'})

def extract_numeric(x):
    if pd.isna(x): return None
    x_clean = re.sub(r'[^\d\.\-\+eE]', '', str(x))
    try:
        return float(x_clean)
    except:
        return None

df['actual_answer_num'] = df.apply(
    lambda row: extract_numeric(row.actual_answer) if row.question_type == 'metrics-generated' else None,
    axis=1
)
df = df.dropna(subset=['evidence', 'actual_answer']).reset_index(drop=True)

def clean_context(evidence):
    import ast
    try:
        evidence_list = ast.literal_eval(evidence)
    except:
        evidence_list = evidence
    texts = []
    for ev in evidence_list:
        if isinstance(ev, dict) and 'evidence_text' in ev:
            texts.append(ev['evidence_text'])
        elif isinstance(ev, str):
            texts.append(ev)
    return "\n".join(texts)

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

prompt_numeric = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Extract the exact numeric value that answers the question from the context below.\n"
        "Respond with only the number and no explanation.\n"
        "If the number is a percentage or ratio, include decimals as needed.\n\n"
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

def extract_answer(decoded, inp):
    if decoded.startswith(inp):
        decoded = decoded[len(inp):]
    decoded = decoded.replace(",", "")
    nums = re.findall(r'-?\d+(?:\.\d+)?', decoded)
    if nums:
        answer = nums[-1]
        if answer.endswith('.0'):
            answer = answer[:-2]
        elif answer.endswith('.'):
            answer = answer[:-1]
        return answer.strip()
    return decoded.strip()

def numeric_match(pred, actual, tol=0.05):
    try:
        nums = re.findall(r'-?\d+\.?\d*', pred.replace(",", ""))
        if not nums or actual is None:
            return False
        pred_val = float(nums[-1])
        return abs(abs(pred_val) - abs(actual)) / (abs(actual) + 1e-6) <= tol
    except:
        return False

def semantic_match(pred, actual_text, threshold=0.80):
    emb_p = semantic_model.encode(pred, convert_to_tensor=True)
    emb_a = semantic_model.encode(actual_text, convert_to_tensor=True)
    score = util.cos_sim(emb_p, emb_a).item()
    return score, score >= threshold

base_models = {
    "TinyLLaMA 1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Phi-2": "microsoft/phi-2",
    "DeepSeek Coder 1.3B": "deepseek-ai/deepseek-coder-1.3b-base",
    "Gemma 2B IT": "google/gemma-2-2b-it"
}

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

prompt_numeric = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Extract the exact numeric value that answers the question from the context below.\n"
        "Respond with only the number and no explanation.\n"
        "If the number is a percentage or ratio, include decimals as needed.\n\n"
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

def get_pred(row, model, tokenizer, max_prompt_len_ratio=0.9, default_max_len=4096):
    context = clean_context(row.evidence)

    chosen_prompt = prompt_numeric if row.question_type == "metrics-generated" else prompt_semantic
    inp = chosen_prompt.format(context=context, question=row.question)

    model_max_len = getattr(tokenizer, "model_max_length", default_max_len)
    if model_max_len > 1e6:
        model_max_len = default_max_len

    max_allowed = int(model_max_len * max_prompt_len_ratio)

    tokenized_len = len(tokenizer(inp)["input_ids"])
    if tokenized_len > max_allowed:
        inp = f"You are a financial analyst. Context:\n{context[:1000]} \nQuestion: {row.question}\nAnswer:"

    inputs = tokenizer(inp, return_tensors="pt", truncation=True, padding=True, max_length=max_allowed).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=128,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False
    )

    decoded = tokenizer.decode(out[0], skip_special_tokens=True)

    if row.question_type == "metrics-generated":
        pred = extract_answer(decoded, inp)
    else:
        pred = decoded.replace(inp, "").strip()

    return pred

all_results = {}

for model_name, model_id in base_models.items():
    print(f"\nLoading model: {model_name} ({model_id})")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        load_in_4bit=False,
        trust_remote_code=False
    )
    model.eval()
    
    results = []
    errors = []
    
    for row in tqdm(df.itertuples(), total=len(df), desc=f"Evaluating {model_name}"):
        pred = get_pred(row, model, tokenizer)
        
        if row.question_type == "metrics-generated":
            correct = numeric_match(pred, row.actual_answer_num)
            score = None
            if not correct:
                errors.append({
                    "id": row.financebench_id,
                    "type": row.question_type,
                    "question": row.question,
                    "prediction": pred,
                    "actual": row.actual_answer_num,
                    "context": row.evidence,
                    "error_type": "numeric_mismatch"
                })
        else:
            score, correct = semantic_match(pred, row.actual_answer)
            if not correct:
                errors.append({
                    "id": row.financebench_id,
                    "type": row.question_type,
                    "question": row.question,
                    "prediction": pred,
                    "actual": row.actual_answer,
                    "semantic_score": score,
                    "context": row.evidence,
                    "error_type": "semantic_mismatch"
                })
        
        results.append({
            "id": row.financebench_id,
            "type": row.question_type,
            "prediction": pred,
            "actual": row.actual_answer,
            "numeric_correct": correct if row.question_type == "metrics-generated" else None,
            "semantic_score": score if row.question_type != "metrics-generated" else None,
            "semantic_correct": correct if row.question_type != "metrics-generated" else None
        })

    res_df = pd.DataFrame(results)
    errors_df = pd.DataFrame(errors)

    print(f"Model: {model_name}")
    print(f" Numeric accuracy (metrics-generated): {res_df[res_df.type=='metrics-generated'].numeric_correct.mean():.4f}")
    print(f" Semantic accuracy (novel + domain): {res_df[res_df.type.isin(['novel-generated','domain-relevant'])].semantic_correct.mean():.4f}")
    print(f" Total incorrect logged: {len(errors_df)}")
    
    all_results[model_name] = {
        "results": res_df,
        "errors": errors_df
    }

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
combined_results_df.to_csv("results/hf_base_model_context_results.csv", index=False)
combined_errors_df.to_csv("results/hf_base_model_context_errors.csv", index=False)
