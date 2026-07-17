import os
import re
import pandas as pd
from datasets import load_dataset
from src.config import (
    MEDQUAD_PATH,
    MEDQUAD_HF_PATH,
    PUBMEDQA_PATH,
    CHATBOT_DATASET_PATH,
)

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Replace newlines, tabs, and carriage returns with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove control/non-printable characters, keeping standard printable ASCII
    text = re.sub(r"[^\x20-\x7E]+", " ", text)
    # Remove unnecessary/multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def is_document_style(text):
    """
    Checks if the text contains keywords/patterns indicating clinical reports,
    referral letters, discharge summaries, or general correspondence.
    """
    if not isinstance(text, str):
        return False
    # Clinical documents, letter greetings/signatures, and report keywords
    blacklist = [
        "dear doctor", "dear dr", "sincerely", "referral", "discharge summary", 
        "operative note", "operative report", "consultation report", "consultation note",
        "feel free to contact", "thank you very much", "yours sincerely", "best regards", 
        "best wishes", "subjective:", "objective:", "hospital course", "chief complaint",
        "thank you for posting", "thank you for your query", "post your query", 
        "thank you for asking", "dear patient", "dear anonymous", "chat doctor", "online doctor",
        "wishing you a speedy", "speedy recovery", "wish you all the best", "yours truly", "regards,"
    ]
    text_lower = text.lower()
    for term in blacklist:
        if term in text_lower:
            return True
    return False

def clean_and_normalize_df(df):
    """
    Normalizes columns 'question' and 'answer', removes duplicates, empty rows, 
    and short answers. Also filters out clinical documents and correspondence.
    """
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).apply(normalize_text)
    df["answer"] = df["answer"].astype(str).apply(normalize_text)
    
    # Filter empty rows after normalization
    df = df[(df["question"] != "") & (df["answer"] != "")]
    
    # Filter out document-style, referrals, correspondence and signature greetings
    df = df[~df["question"].apply(is_document_style)]
    df = df[~df["answer"].apply(is_document_style)]
    
    # Filter out very short answers (less than 20 characters)
    df = df[df["answer"].str.len() >= 20]
    
    # Deduplicate questions
    df = df.drop_duplicates(subset=["question"])
    return df

def main():
    print("Starting dataset preprocessing pipeline...")
    all_data = []

    # 1. Load Local MedQuAD CSV
    if os.path.exists(MEDQUAD_PATH):
        print(f"Loading local MedQuAD CSV from {MEDQUAD_PATH}...")
        medquad = pd.read_csv(MEDQUAD_PATH)
        df_mq = medquad[["question", "answer"]].copy()
        all_data.append(("MedQuAD CSV", df_mq))
    else:
        print(f"Warning: Local MedQuAD CSV not found at {MEDQUAD_PATH}")

    # 2. Load Local MedQuAD HF Parquet
    if os.path.exists(MEDQUAD_HF_PATH):
        print(f"Loading local MedQuAD HF Parquet from {MEDQUAD_HF_PATH}...")
        medquad_hf = pd.read_parquet(MEDQUAD_HF_PATH)
        df_mq_hf = medquad_hf[["question", "answer"]].copy()
        all_data.append(("MedQuAD HF Parquet", df_mq_hf))
    else:
        print(f"Warning: Local MedQuAD HF Parquet not found at {MEDQUAD_HF_PATH}")

    # 3. Load Local PubMedQA Parquet
    if os.path.exists(PUBMEDQA_PATH):
        print(f"Loading local PubMedQA Parquet from {PUBMEDQA_PATH}...")
        pubmedqa = pd.read_parquet(PUBMEDQA_PATH)
        df_pm = pubmedqa[["question", "long_answer"]].rename(columns={"long_answer": "answer"}).copy()
        all_data.append(("PubMedQA Parquet", df_pm))
    else:
        print(f"Warning: Local PubMedQA Parquet not found at {PUBMEDQA_PATH}")

    # 4. Download and load Hugging Face MedMCQA
    print("Downloading/Loading MedMCQA from Hugging Face...")
    try:
        medmcqa = load_dataset("openlifescienceai/medmcqa", split="train")
        print(f"MedMCQA loaded. Formatting {len(medmcqa)} rows...")
        mc_questions, mc_answers = [], []
        for sample in medmcqa:
            q = sample.get("question", "")
            opa = sample.get("opa", "")
            opb = sample.get("opb", "")
            opc = sample.get("opc", "")
            opd = sample.get("opd", "")
            cop = sample.get("cop")
            exp = sample.get("exp", "")
            
            options = [opa, opb, opc, opd]
            correct_option = ""
            if cop is not None and isinstance(cop, int) and 0 <= cop < 4:
                correct_option = options[cop]
            
            parts = []
            if correct_option:
                parts.append(f"correct option: {correct_option}.")
            if exp:
                parts.append(exp)
            
            ans = " ".join(parts)
            mc_questions.append(q)
            mc_answers.append(ans)
            
        df_mc = pd.DataFrame({"question": mc_questions, "answer": mc_answers})
        all_data.append(("MedMCQA", df_mc))
    except Exception as e:
        print(f"Error loading MedMCQA: {e}")



    # 6. Download and load Hugging Face MedDialog (English 100k subset)
    print("Downloading/Loading MedDialog English 100k from Hugging Face...")
    try:
        meddialog = load_dataset("khoaliamle/MedDialog-EN-100k", split="train")
        print(f"MedDialog loaded. Formatting {len(meddialog)} rows...")
        df_md = pd.DataFrame({
            "question": meddialog["input"],
            "answer": meddialog["output"]
        })
        all_data.append(("MedDialog", df_md))
    except Exception as e:
        print(f"Error loading MedDialog: {e}")

    # 7. Download and load Hugging Face PubMedQA (Labeled & Artificial datasets)
    print("Downloading/Loading PubMedQA from Hugging Face...")
    try:
        pubmed_qa_labeled = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        print(f"PubMedQA Labeled loaded. Formatting {len(pubmed_qa_labeled)} rows...")
        df_pql = pd.DataFrame({
            "question": pubmed_qa_labeled["question"],
            "answer": pubmed_qa_labeled["long_answer"]
        })
        all_data.append(("PubMedQA HF Labeled", df_pql))
        
        # Load subset of pqa_artificial for additional coverage
        pubmed_qa_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial", split="train")
        print(f"PubMedQA Artificial loaded. Formatting {len(pubmed_qa_art)} rows...")
        df_pqa = pd.DataFrame({
            "question": pubmed_qa_art["question"],
            "answer": pubmed_qa_art["long_answer"]
        })
        all_data.append(("PubMedQA HF Artificial", df_pqa))
    except Exception as e:
        print(f"Error loading PubMedQA from Hugging Face: {e}")

    # Combine all datasets
    print("\nMerging all datasets...")
    processed_dfs = []
    for name, df in all_data:
        print(f"Preprocessing & cleaning {name} (original count: {len(df)})...")
        clean_df = clean_and_normalize_df(df)
        print(f"-> {name} cleaned count: {len(clean_df)}")
        processed_dfs.append(clean_df)

    merged_df = pd.concat(processed_dfs, ignore_index=True)
    print(f"\nInitial merged count before final deduplication: {len(merged_df)}")
    
    # Final cleanup and deduplication across all sources
    merged_df = merged_df.dropna(subset=["question", "answer"])
    merged_df = merged_df.drop_duplicates(subset=["question"])
    merged_df = merged_df[merged_df["answer"].str.len() >= 20]
    
    print(f"Final merged count after global deduplication and filter: {len(merged_df)}")
    
    # Save output
    os.makedirs(os.path.dirname(CHATBOT_DATASET_PATH), exist_ok=True)
    merged_df.to_csv(CHATBOT_DATASET_PATH, index=False)
    print(f"Merged dataset saved successfully to {CHATBOT_DATASET_PATH}!")

if __name__ == "__main__":
    main()
