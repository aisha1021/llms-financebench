import re
import ast
import gc
import pandas as pd
import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig
from langchain.prompts import PromptTemplate

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

def make_ft_prompt(row):
    context = clean_context(row.evidence)
    question = row.question

    if row.question_type == "metrics-generated":
        answer = str(row.actual_answer_num)
        prompt = (
            "You are a financial analysis assistant.\n"
            "Using the following context, answer the question with only the final numeric value.\n"
            "Respond with only the number and no explanation.\n"
            "If the number is a percentage or ratio, include decimals as needed.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}:"
        )
    else:
        answer = str(row.actual_answer)
        prompt = (
            "You are a financial analysis assistant.\n"
            "Using the context below, answer the question in 1–2 clear and concise sentences.\n"
            "Provide an explanation if the question asks for one. Avoid generic explanations. Start your answer directly.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}"
        )
    
    return prompt.format(context=context, question=question, answer=answer)


df['text'] = df.apply(make_ft_prompt, axis=1)
df_train, df_test = train_test_split(df, test_size=0.8, stratify=df['question_type'], random_state=42)

train_ds = Dataset.from_pandas(df_train[['text']].reset_index(drop=True))
test_df = df_test.reset_index(drop=True)


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

semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

prompt_numeric = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Using the following context, answer the question with only the final numeric value.\n"
        "Respond with only the number and no explanation.\n"
        "If the number is a percentage or ratio, include decimals as needed.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
)

prompt_semantic = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a financial analysis assistant.\n"
        "Using the context below, answer the question in 1–2 clear and concise sentences.\n"
        "Provide an explanation if the question asks for one. Avoid generic explanations. Start your answer directly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
)

def get_pred(row, model, tokenizer, max_prompt_len_ratio=0.9):
    context = clean_context(row.evidence)
    
    chosen_prompt = prompt_numeric if row.question_type == "metrics-generated" else prompt_semantic
    inp = chosen_prompt.format(context=context, question=row.question)

    tokenized_len = len(tokenizer(inp)['input_ids'])
    max_allowed = int(tokenizer.model_max_length * max_prompt_len_ratio)

    if tokenized_len > max_allowed:
        inp = f"You are a financial analyst. Using the context below, answer the question in 1–2 clear and concise sentences. Context:\n{context} \nQuestion: {row.question}\nAnswer:"

    inputs = tokenizer(inp, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(model.device)

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

def find_lora_target_modules(model):
    target_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Match on names commonly used for attention and MLP layers
            if any(kw in name.lower() for kw in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
                target_modules.add(name.split('.')[-1])
    return sorted(list(target_modules))

def run_model_iteration(model_name, model_id, train_dataset, test_df, all_results):
    print(f"\nFine-tuning and evaluating model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True
    )

    target_modules = find_lora_target_modules(model)
    print(f"LoRA target modules for {model_name}: {target_modules}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(ex):
        tok = tokenizer(ex['text'], truncation=True, padding="max_length", max_length=768)
        tok['labels'] = tok['input_ids'].copy()
        return tok

    tokenized_train_ds = train_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=f"./ft_{model_name.replace(' ', '_')}",
        num_train_epochs=4,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
        logging_steps=5,
        save_strategy="no"
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train_ds)
    trainer.train()

    results, errors = [], []
    model.eval()

    with torch.no_grad():
        for row in tqdm(test_df.itertuples(), total=len(test_df), desc=f"Evaluating {model_name}"):
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

    print(f"\nResults for {model_name}")
    print(f"Numeric accuracy: {res_df[res_df.type=='metrics-generated'].numeric_correct.mean():.4f}")
    print(f"Semantic accuracy: {res_df[res_df.type.isin(['novel-generated','domain-relevant'])].semantic_correct.mean():.4f}")
    print(f"Total incorrect logged: {len(errors_df)}")

    all_results[model_name] = {
        "results": res_df,
        "errors": errors_df
    }

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

all_results = {}
run_model_iteration("TinyLLaMA 1.1B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", train_ds, test_df, all_results)

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
combined_results_df.to_csv("results/hf_TinyLlama-1.1B-Chat-v1.0_base_model_finetune_results.csv", index=False)
combined_errors_df.to_csv("results/hf_TinyLlama-1.1B-Chat-v1.0_base_model_finetune_errors.csv", index=False)