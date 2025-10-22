# analysis/LLM/prepare_full_input.py
import re, unicodedata, pandas as pd, tiktoken
from pathlib import Path

DATA = Path("dataset/temporaryFileToWorkWith/CRS_textprep.parquet")
OUT  = Path("analysis/LLM/artifacts/full/inference_input.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)

MAX_TOKENS = 1200
HEAD_SENTENCES = 5
CTX_WINDOW = 1

def normalize(s):
    if not isinstance(s,str): return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+"," ",s).strip()

def split_sentences(text):
    parts = re.split(r"(?<=[\.\?!])\s+(?=[^\s])", text.replace("\n"," "))
    return [p.strip() for p in parts if p.strip()]

ANCHOR = re.compile(
    r"\b(?:biodiversit|ecosystem|conservation|protected area|wildlife|forest|wetland|ramsar|"
    r"peatland|coral reef|seagrass|kba|natura 2000|land degradation|desertification|"
    r"invasive species|human[- ]wildlife conflict|poaching|anti[- ]poaching|cites|redd\+|"
    r"reforestation|afforestation|restoration|watershed|mangrove|park|habitat|species|"
    r"fisher(?:y|ies)|marine|coastal|mpa|river|lake|agroforestry)\b", re.I
)

def shorten_preserve_triggers(text):
    sents = split_sentences(text)
    keep = set(range(min(HEAD_SENTENCES, len(sents))))
    norm_sents = [normalize(s) for s in sents]
    hits = [i for i,s in enumerate(norm_sents) if ANCHOR.search(s)]
    for i in hits:
        for j in range(max(0,i-CTX_WINDOW), min(len(sents), i+CTX_WINDOW+1)):
            keep.add(j)
    return " ".join(sents[i] for i in sorted(keep))

def get_encoder():
    try: return tiktoken.get_encoding("o200k_base")
    except Exception: return tiktoken.get_encoding("cl100k_base")
ENC = get_encoder()

def cap_tokens(text, max_tokens=MAX_TOKENS):
    toks = ENC.encode(text); 
    return text if len(toks) <= max_tokens else ENC.decode(toks[:max_tokens])

use_cols = ["activity_id","year","donor_name","recipient_name","text_en","text_clean"]
df = pd.read_parquet(DATA, columns=use_cols).rename(columns={
    "donor_name":"donor","recipient_name":"recipient","text_en":"text"
})
df["text_for_llm"] = df["text"].fillna(df["text_clean"]).fillna("").astype(str)
df = df[df["text_for_llm"].map(lambda s: len(normalize(s)) >= 30)].copy()

# anchor-filter to save cost
df = df[df["text_for_llm"].map(lambda s: bool(ANCHOR.search(normalize(s))))].copy()

# shorten + cap
df["text_for_llm"] = df["text_for_llm"].map(shorten_preserve_triggers).map(cap_tokens)

df[["activity_id","year","donor","recipient","text_for_llm"]].to_csv(OUT, index=False)
print("wrote", OUT, "rows:", len(df))
