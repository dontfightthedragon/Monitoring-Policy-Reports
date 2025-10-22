# analysis/LLM/validation/make_manual_labeled_sample_pure_robust.py
import csv, re

IN_PATH  = "analysis/LLM/validation/manual_validation_sample_blind.csv"
OUT_PATH = "analysis/LLM/validation/manual_labeled_sample_pure.csv"

ALLOWED = {"FORESTS_LAND","WATER_MARINE","WILDLIFE_SPECIES","CROSS_CUTTING","NONE"}

def canon_label(x: str) -> str:
    if not isinstance(x, str): return ""
    t = (x.strip().upper()
         .replace("&","AND").replace("/","_").replace("-","_").replace(" ","_"))
    t = re.sub(r"[^A-Z_]", "", t)
    m = {"FOREST_LAND":"FORESTS_LAND","FORESTS":"FORESTS_LAND","FOREST":"FORESTS_LAND",
         "WATER":"WATER_MARINE","MARINE":"WATER_MARINE",
         "WILDLIFE":"WILDLIFE_SPECIES","SPECIES":"WILDLIFE_SPECIES",
         "CROSSCUTTING":"CROSS_CUTTING","CROSS__CUTTING":"CROSS_CUTTING",
         "N_A":"NONE","N/A":"NONE","NO_BIODIVERSITY":"NONE","NOT_BIODIVERSITY":"NONE"}
    return m.get(t, t)

def norm_targets(s: str) -> str:
    if not isinstance(s, str) or not s.strip(): return ""
    s = s.strip().strip('"').strip("'")
    if s.upper() == "NONE": return ""
    toks = re.split(r"[,\s;]+", s)
    vals = []
    for t in toks:
        m = re.fullmatch(r"(?:SDG)?15\.(1|2|3|4|5|7|8|9)", t.strip().upper())
        if m: vals.append(f"SDG15.{m.group(1)}")
    return ",".join(sorted(set(vals)))

def parse_row(parts):
    """
    parts = CSV fields of a line. We reconstruct:
      activity_id = first field
      human_primary_label = first tail field canonicalizing into ALLOWED
      human_sdg15_targets = rest (normalized); empty if NONE/blank
    """
    if not parts: return None
    if len(parts) == 1:
        parts = [parts[0], ""]
    activity_id = str(parts[0]).strip()
    if not activity_id:
        return None
    tail = parts[1:]

    # detect label in tail
    label_idx, label_val = None, ""
    for i, v in enumerate(tail):
        c = canon_label(v)
        if c in ALLOWED:
            label_idx, label_val = i, c
            break

    if label_idx is None:
        # if no explicit label token found, try to fall back to a column literally named 'human_primary_label'
        # (some exports shift columns; last field could be the label)
        # last non-empty field as last resort
        cand = canon_label(tail[-1]) if tail else ""
        if cand in ALLOWED:
            label_idx, label_val = len(tail)-1, cand
        else:
            label_val = ""  # keep blank; we still emit the row

    targets_raw = ",".join(tail[label_idx+1:]).strip() if label_idx is not None else ""
    targets = norm_targets(targets_raw)

    return [activity_id, label_val, targets]

# --- read raw, write pure (no orig_text), no row skipping ---
rows_in, rows_out = 0, 0
with open(IN_PATH, "r", encoding="utf-8", errors="replace", newline="") as fin, \
     open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:

    rdr = csv.reader(fin, delimiter=",", quotechar='"', skipinitialspace=False)
    w   = csv.writer(fout)

    # header detection
    first = next(rdr, None)
    if first is None:
        raise SystemExit("Input file is empty.")
    maybe_header = [h.strip().lower() for h in first]
    header_like = any(h in ("activity_id","orig_text","human_primary_label","human_sdg15_targets") for h in maybe_header)

    w.writerow(["activity_id","human_primary_label","human_sdg15_targets"])

    # process first row (as data) if it wasn't a header
    if not header_like:
        rows_in += 1
        rec = parse_row(first)
        if rec:
            w.writerow(rec); rows_out += 1

    # process remaining rows
    for parts in rdr:
        rows_in += 1
        rec = parse_row(parts)
        if rec:
            w.writerow(rec); rows_out += 1

print(f"Read {rows_in} data lines; wrote {rows_out} rows to {OUT_PATH}")
