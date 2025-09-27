# analysis/LLM/label_llm.py
import argparse, json, hashlib, os, time, pandas as pd
from pathlib import Path
from openai import OpenAI

# ---- config
SYSTEM_PATH = Path("analysis/LLM/prompts/classifier.system.txt")
USER_PATH   = Path("analysis/LLM/prompts/classifier.user.json")
MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
BASE    = os.getenv("OPENAI_API_BASE")  # leave unset for OpenAI cloud
KEY     = os.getenv("OPENAI_API_KEY")
TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))

if BASE:
    if not KEY: KEY = "lm-studio"  # allow local servers
    client = OpenAI(base_url=BASE, api_key=KEY, timeout=TIMEOUT)
else:
    if not KEY:
        raise SystemExit("OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-...'")
    client = OpenAI(api_key=KEY, timeout=TIMEOUT)

SYSTEM = SYSTEM_PATH.read_text()
USER_TPL = json.loads(USER_PATH.read_text())

CACHE = Path("analysis/LLM/cache"); CACHE.mkdir(parents=True, exist_ok=True)

def cache_key(text: str) -> str:
    h = hashlib.sha1()
    h.update(MODEL.encode()); h.update(SYSTEM.encode()); h.update(json.dumps(USER_TPL).encode())
    h.update((text or "").encode("utf-8","ignore"))
    return h.hexdigest()

# robust normalization for sdg15 target list from model output (string or list)
def norm_targets_list(tg) -> list:
    if isinstance(tg, str):
        items = [t.strip() for t in tg.split(",")]
    elif isinstance(tg, list):
        items = tg
    else:
        items = []
    valid = {"1","2","3","4","5","7","8","9"}
    out = []
    for t in items:
        s = str(t).strip().upper()
        if not s or s == "NONE":
            continue
        s = s.replace("SDG 15", "").replace("SDG-15", "").replace("SDG15", "").strip(" .:")
        if s.startswith("15."):
            s = s.split(".", 1)[1]
        if s in valid:
            out.append(f"SDG15.{s}")
    return sorted(set(out))

def call_model(text: str, retries: int = 2, backoff: float = 1.0) -> dict:
    user = USER_TPL.copy(); user["text"] = text
    last_err = None
    for attempt in range(retries + 1):
        try:
            # prefer JSON mode
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role":"system","content": SYSTEM},
                    {"role":"user","content": json.dumps(user, ensure_ascii=False)}
                ],
                timeout=TIMEOUT,
            )
            data = json.loads(resp.choices[0].message.content)
            return data
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            # known quota issue -> return placeholder immediately
            if "insufficient_quota" in msg or "quota" in msg:
                return {"primary_label":"NONE","sdg15_targets":[],"rationale":"insufficient quota"}
            # otherwise retry transient errors with backoff
            if attempt < retries:
                time.sleep(backoff); backoff *= 2
                continue
            # retries exhausted -> placeholder with reason
            return {"primary_label":"NONE","sdg15_targets":[],"rationale": f"error: {type(e).__name__}"}

def classify(text: str) -> dict:
    key = cache_key(text); cj = CACHE / f"{key}.json"
    if cj.exists():
        return json.loads(cj.read_text())

    data = call_model(text)
    # coerce schema
    pl = str(data.get("primary_label","NONE")).strip() or "NONE"
    tg = norm_targets_list(data.get("sdg15_targets", []))
    rat = str(data.get("rationale","")).strip()[:600]

    rec = {"primary_label": pl, "sdg15_targets": sorted(set(tg)), "rationale": rat}
    cj.write_text(json.dumps(rec, ensure_ascii=False))
    return rec

if __name__ == "__main__":
    print(f"[label_llm] base={BASE or '<openai-cloud>'} model={MODEL} key_present={bool(KEY)}")
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    rows = []
    for _, r in df.iterrows():
        text = str(r.get("text_for_llm",""))
        print(f"[label_llm] classifying {r.get('activity_id')} len={len(text)}")
        y = classify(text)
        rows.append({
            "activity_id": r["activity_id"],
            "llm_primary_label": y["primary_label"],
            "llm_sdg15_targets": ",".join(y["sdg15_targets"]),
            "llm_rationale": y["rationale"],
        })
    pd.DataFrame(rows).to_csv(args.outp, index=False)
    print(f"[label_llm] wrote {args.outp}")
