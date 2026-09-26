import streamlit as st
import pandas as pd
import numpy as np
import json
import urllib.request
import urllib.error
import re
import time
import traceback
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Table 12 COMPAS Replication", layout="wide")
st.title("🎯 COMPAS Dataset - Table 12 Replication Console")
st.caption("Powered by Groq API")

# Sidebar API authorization
st.sidebar.header("1. API Authorization")

api_key_from_secrets = st.secrets.get("GROQ_API_KEY", "")
groq_api_key = st.sidebar.text_input("Groq API Key (gsk_...)", value=api_key_from_secrets, type="password")

# ==========================================
# DYNAMIC MODEL FETCH & FILTERING
# ==========================================
def get_available_groq_models(api_key: str):
    """Fetches active text-chat models from Groq API, filtering out TTS/audio/guard models."""
    fallback_models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    if not api_key:
        return fallback_models
    
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=5.0) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            models = []
            for m in res_body.get("data", []):
                m_id = m["id"]
                # Filter out non-text/chat completion models
                excluded_keywords = ["whisper", "guard", "canopylabs", "orpheus", "tts", "stt", "audio", "vision"]
                if not any(k in m_id.lower() for k in excluded_keywords):
                    models.append(m_id)
            return models if models else fallback_models
    except Exception:
        return fallback_models

st.sidebar.header("2. Model Selection")
available_models = get_available_groq_models(groq_api_key)
active_model = st.sidebar.selectbox("Choose Primary Model", available_models)

# ==========================================
# EXTRACTION & METRICS ENGINE
# ==========================================
def extract_binary_prediction(response_text: str):
    if not response_text:
        return None, "Empty Response"
    text = str(response_text).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "prediction" in data:
                val = str(data["prediction"]).strip()
                if val in ["0", "1"]:
                    return int(val), "Valid JSON"
            if "output" in data:
                val = str(data["output"]).strip()
                if val in ["0", "1"]:
                    return int(val), "Valid JSON (output key)"
    except Exception:
        pass

    match = re.search(r'["\']?prediction["\']?\s*:\s*["\']?([01])["\']?', text, re.IGNORECASE)
    if match:
        return int(match.group(1)), "Regex Key Match"

    digits = re.findall(r'\b([01])\b', text)
    if digits:
        return int(digits[-1]), "Fallback Trailing Digit"
        
    return None, f"Parsing Failed: '{text}'"

def compute_paper_fairness_metrics(df_res, protected_col, target_col):
    y_true = df_res[target_col].values.astype(int)
    y_pred = df_res['pred'].values.astype(int)
    
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    groups = [g for g in df_res[protected_col].unique() if pd.notna(g)]
    if len(groups) < 2:
        return {'Acc': acc, 'F1': f1, 'SPR': 1.0, 'EOR': 1.0, 'EOD': 1.0, 'PP': 1.0, 'SAG': 0.0}

    g0 = df_res[df_res[protected_col] == groups[0]]
    g1 = df_res[df_res[protected_col] == groups[1]]

    def compute_smoothed_ratio(v1, v2, eps=0.01):
        ratio = (max(v1, v2) + eps) / (min(v1, v2) + eps)
        return float(min(ratio, 10.0))

    pr0 = (g0['pred'] == 1).mean() if len(g0) > 0 else 0
    pr1 = (g1['pred'] == 1).mean() if len(g1) > 0 else 0
    spr = compute_smoothed_ratio(pr0, pr1)

    tpr0 = ((g0['pred'] == 1) & (g0[target_col] == 1)).sum() / max(1, (g0[target_col] == 1).sum())
    tpr1 = ((g1['pred'] == 1) & (g1[target_col] == 1)).sum() / max(1, (g1[target_col] == 1).sum())
    eor = compute_smoothed_ratio(tpr0, tpr1)

    fpr0 = ((g0['pred'] == 1) & (g0[target_col] == 0)).sum() / max(1, (g0[target_col] == 0).sum())
    fpr1 = ((g1['pred'] == 1) & (g1[target_col] == 0)).sum() / max(1, (g1[target_col] == 0).sum())
    eod = (eor + compute_smoothed_ratio(fpr0, fpr1)) / 2.0

    ppv0 = ((g0['pred'] == 1) & (g0[target_col] == 1)).sum() / max(1, (g0['pred'] == 1).sum())
    ppv1 = ((g1['pred'] == 1) & (g1[target_col] == 1)).sum() / max(1, (g1['pred'] == 1).sum())
    pp = compute_smoothed_ratio(ppv0, ppv1)

    acc0 = accuracy_score(g0[target_col], g0['pred']) if len(g0) > 0 else 0
    acc1 = accuracy_score(g1[target_col], g1['pred']) if len(g1) > 0 else 0
    sag = float(abs(acc0 - acc1))

    return {'Acc': acc, 'F1': f1, 'SPR': spr, 'EOR': eor, 'EOD': eod, 'PP': pp, 'SAG': sag}

# ==========================================
# PROMPT BUILDERS & API ROUTER
# ==========================================
def generate_icl_text(train_df, strategy, feature_cols, protected_col, target_col, demo_groups):
    if strategy == "F":
        return ""
    elif strategy == "U":
        samples = train_df.sample(n=min(len(train_df), 4), random_state=42)
        out = []
        for r in samples.to_dict('records'):
            feats = {k: v for k, v in r.items() if k in feature_cols and k != protected_col}
            out.append(f"Input: {json.dumps(feats)} -> Output: {{\"prediction\": {r[target_col]}}}")
        return "\n".join(out)
    elif strategy == "E":
        strata = []
        for g in demo_groups:
            for t in train_df[target_col].unique():
                sub = train_df[(train_df[protected_col] == g) & (train_df[target_col] == t)]
                if len(sub) > 0:
                    strata.append(sub.sample(n=1, random_state=42))
        esad_df = pd.concat(strata) if strata else train_df.sample(n=4, random_state=42)
        out = []
        for r in esad_df.to_dict('records'):
            feats = {k: v for k, v in r.items() if k in feature_cols}
            out.append(f"Input: {json.dumps(feats)} -> Output: {{\"prediction\": {r[target_col]}}}")
        return "\n".join(out)
    elif strategy == "C":
        samples = train_df.sample(n=min(len(train_df), 2), random_state=42)
        out = []
        for r in samples.to_dict('records'):
            f = {k: v for k, v in r.items() if k in feature_cols}
            out.append(f"Input A: {json.dumps(f)} -> Output: {{\"prediction\": {r[target_col]}}}")
            cf = f.copy()
            alt = [g for g in demo_groups if g != f.get(protected_col)]
            if alt:
                cf[protected_col] = alt[0]
            out.append(f"Input B: {json.dumps(cf)} -> Output: {{\"prediction\": {r[target_col]}}}")
        return "\n".join(out)

def construct_full_prompt(strategy, icl_text, test_instance, feature_cols, protected_col, target_col):
    if strategy == "U":
        test_feats = {k: v for k, v in test_instance.items() if k in feature_cols and k != protected_col}
    else:
        test_feats = {k: v for k, v in test_instance.items() if k in feature_cols}
        
    directive = f"Predict the binary value (0 or 1) for target '{target_col}' based on tabular attributes."
    few_shot = f"Examples:\n{icl_text}\n\n" if icl_text else ""
    return f"{directive}\n\n{few_shot}Evaluate Sample Data:\n{json.dumps(test_feats)}\n\nRespond ONLY with a JSON object: {{\"prediction\": 1}} or {{\"prediction\": 0}}"

def call_groq_api(api_key: str, model: str, prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise classifier. Return ONLY a raw JSON dict with key 'prediction' containing 0 or 1."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10.0) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            if "choices" in res_body and len(res_body["choices"]) > 0:
                content = res_body["choices"][0]["message"]["content"]
                pred, parse_msg = extract_binary_prediction(content)
                if pred is not None:
                    return pred, content, None
                else:
                    return 0, content, f"Parse Error: {parse_msg}"
            else:
                return 0, str(res_body), f"API Error: Missing 'choices' in response"
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        return 0, err_msg, f"HTTP Error {e.code}"
    except Exception as e:
        return 0, str(e), f"Exception: {str(e)}"

# ==========================================
# UI EXECUTION
# ==========================================
st.header("1. Dataset Setup")
uploaded_file = st.file_uploader("Upload COMPAS CSV File", type=["csv"])

if uploaded_file and groq_api_key:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded COMPAS dataset ({len(df)} rows)")

    target_default_idx = df.columns.get_loc('two_year_recid') if 'two_year_recid' in df.columns else 0
    available_protected_cols = [c for c in df.columns if c != 'two_year_recid']
    protected_default_idx = available_protected_cols.index('race') if 'race' in available_protected_cols else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_col = st.selectbox("Target Column", df.columns, index=target_default_idx)
    with col2:
        protected_col = st.selectbox("Protected Column", available_protected_cols, index=protected_default_idx)
    with col3:
        sample_size = st.slider("Samples Per Strategy", min_value=10, max_value=50, value=20)
    with col4:
        num_runs = st.slider("Cross Validation Seeds", min_value=1, max_value=3, value=1)

    drop_cols = {'id', 'name', 'first', 'last', 'c_case_number', 'compas_screening_date', 'dob'}
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    demo_groups = [g for g in df[protected_col].unique() if pd.notna(g)]

    st.header("2. Table 12 Replication Pipeline")
    
    if st.button("🚀 Run Table 12 Experiments", type="primary", use_container_width=True):
        strategies = ["F", "U", "E", "C"]
        formatted_rows = []
        errors_log = []

        for strat in strategies:
            run_metrics_list = []
            
            for seed in range(num_runs):
                train_df = df.sample(frac=0.7, random_state=42 + seed)
                test_df = df.drop(train_df.index).sample(n=min(sample_size, len(df)-len(train_df)), random_state=42 + seed).reset_index(drop=True)
                
                icl_text = generate_icl_text(train_df, strat, feature_cols, protected_col, target_col, demo_groups)
                preds = []

                for idx, row in test_df.iterrows():
                    prompt = construct_full_prompt(strat, icl_text, row.to_dict(), feature_cols, protected_col, target_col)
                    pred, raw_out, err = call_groq_api(groq_api_key, active_model, prompt)
                    
                    preds.append(pred)
                    
                    if err:
                        errors_log.append({
                            'Strategy': strat,
                            'Seed': seed + 1,
                            'Row Index': idx,
                            'Error Details': err,
                            'Raw Model Response': raw_out,
                            'Sent Prompt': prompt
                        })

                eval_df = test_df.copy()
                eval_df['pred'] = preds
                metrics = compute_paper_fairness_metrics(eval_df, protected_col, target_col)
                run_metrics_list.append(metrics)

            m_df = pd.DataFrame(run_metrics_list)
            row_data = {'Model': active_model, 'Type': strat}
            
            for metric_key in ['Acc', 'F1', 'SPR', 'EOR', 'EOD', 'PP', 'SAG']:
                mean_v = m_df[metric_key].mean()
                std_v = m_df[metric_key].std() if len(m_df) > 1 else 0.0
                row_data[metric_key] = f"{mean_v:.4f} ({std_v:.4f})"

            formatted_rows.append(row_data)

        st.header("3. Replicated Table 12 Output")
        result_df = pd.DataFrame(formatted_rows)
        st.dataframe(result_df, use_container_width=True)

        st.header("4. Error & Diagnostic Catch Log")
        if errors_log:
            st.error(f"⚠️ Encountered {len(errors_log)} error(s) during execution:")
            err_df = pd.DataFrame(errors_log)
            st.dataframe(err_df, use_container_width=True)
        else:
            st.success("🎉 No errors detected! All responses were parsed successfully.")

elif not groq_api_key:
    st.warning("Please enter your Groq API Key (gsk_...) in the sidebar to run the replication.")
