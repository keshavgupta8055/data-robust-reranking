# ===== EXPERIMENT v4 =====  (one file, ASCII only)
# What is new compared with v3 (each item answers a point of the reviewer checklist):
#  1. article-disjoint dev/test split, larger eval set (N_QUERIES, HOLD10), optional harder pools (N_DISTR)
#  2. realistic date corruption: re-save jump, batch ingestion date, crawl-snapshot dates, year/month granularity,
#     plus mixture-weight sensitivity and a "realistic mix"
#  3. robust date-prior baselines (thresholded tie-break), optional larger re-ranker (Qwen3-Reranker-4B)
#  4. intervals for AVERAGE/WORST, per-query files for joint seed bootstrap (merge_seeds.py), lambda sweep for gate and no-gate,
#     window size K=5, tanh date-scale sweep, breakdown by number of outdated versions
#  5. date-reliability estimator ("adaptive_ft"): estimates how far the dates can be trusted from agreement between date order
#     and judge order on the (unlabelled) collection, and scales the date term accordingly
#  6. judge diagnostics: calibration (ECE, reliability bins, temperature scaling), shortcut checks (edits, digits, years, length)
#  7. end-to-end: EM and token-F1 next to containment, more queries, optional second re-ranker, samples saved for manual checking
# Control everything with environment variables (see CONFIG). DRYRUN=1 runs the whole pipeline on fake data on CPU in minutes.

# ---------------- CONFIG ----------------
import os as _os
def _env(k, d): return _os.environ.get(k, d)
JSEED = int(_env("JSEED", 0))
JUDGE_MODEL = _env("JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GEN_MODEL = _env("GEN_MODEL", "")                                   # optional second generator for the end-to-end test
N_QUERIES = int(_env("N_QUERIES", 2400))                            # dev + test instances (half each, different articles)
HOLD10 = int(_env("HOLD10", 4))                                     # hold out titles with md5 % 10 < HOLD10 (4 = 40% of articles)
N_DISTR = int(_env("N_DISTR", 6))                                   # distractors per query (12 = harder pools)
PASS_SET = _env("PASS_SET", "full")                                 # "full" adds the realistic corruption conditions, "core" is v3
SCORE_NEW_DA = _env("SCORE_NEW_DA", "0") == "1"                     # also score date-aware re-rankers on the new conditions (slower)
RR_SEL = [x for x in _env("RERANKERS_SEL", "qwen3-0.6B,bge-v2-m3").split(",") if x]   # also: qwen3-4B
GRID = _env("GRID", "full")                                         # "small" = fewer settings, about 3x faster
RUN_EXTRAS = _env("RUN_EXTRAS", "1") == "1"                         # adaptive estimator, sweeps, K=5, breakdowns, sensitivity tables
K_EXTRA = int(_env("K_EXTRA", 5))                                   # second window size (0 = off)
RUN_DIAG = _env("RUN_DIAG", "1") == "1"                             # judge diagnostics (calibration, shortcuts, length)
RUN_ADAPT_ONLY = _env("RUN_ADAPT_ONLY", "0") == "1"                     # with RUN_EXTRAS=1: run only the adaptive date-trust estimator (skips lambda/date-scale/K/n-outdated sweeps)
RUN_E2E_VARIANTS = _env("RUN_E2E_VARIANTS", "0") == "1"             # position-controlled end-to-end test
E2E_N = int(_env("E2E_N", 600))
E2E_RR = [x for x in _env("E2E_RERANKERS", "qwen3-0.6B").split(",") if x]
DRYRUN = _env("DRYRUN", "0") == "1"
# ----------------------------------------
def _short(m): return m.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
TAG = "j" + _short(JUDGE_MODEL) + "_s" + str(JSEED) + ("_g" + _short(GEN_MODEL) if GEN_MODEL else "") + ("_" + "+".join(RR_SEL) if RR_SEL != ["qwen3-0.6B", "bge-v2-m3"] else "") + ("_DRY" if DRYRUN else "")
TAGP = "j" + _short(JUDGE_MODEL) + "_s" + str(JSEED) + ("_" + "+".join(RR_SEL) if RR_SEL != ["qwen3-0.6B", "bge-v2-m3"] else "") + ("_DRY" if DRYRUN else "")   # per-query file name (no generator part)
print("RUN TAG:", TAG)

import subprocess, sys
if not DRYRUN:
    import urllib.request
    print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True, capture_output=True, text=True).stdout or "NO GPU: open Settings > Accelerator and pick GPU T4 x2")
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10); print("internet: OK")
    except Exception as e:
        raise SystemExit("NO INTERNET: open Settings > Internet and switch it On. " + repr(e)[:100])
    subprocess.run([sys.executable, "-m", "pip", "-q", "uninstall", "-y", "torchao"])
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "transformers>=4.51.0", "datasets", "accelerate", "peft", "scikit-learn"], check=True)
    os_env_alloc = "expandable_segments:True"

import os
if not DRYRUN: os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch is imported
import json, random, re, time, math, hashlib, pickle, contextlib, glob, zipfile, gc, io, zlib
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import numpy as np, pandas as pd
if not DRYRUN: import torch
SEED = 0
random.seed(SEED); np.random.seed(SEED)
TIMES = {}
@contextlib.contextmanager
def stage(name):
    t0 = time.time(); yield; TIMES[name] = round(time.time() - t0, 1); print(f"[stage {name}: {TIMES[name]:.0f}s]")

# ---------------- data ----------------
def _maybe_json(x):
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: return x
    return x
def _date(s):
    s = str(s)[:10]
    try: datetime.strptime(s, "%Y-%m-%d"); return s
    except Exception: return None

def make_fake_qas(n_titles=700, seed=0):
    r = random.Random(seed); base = datetime(2012, 1, 1); qs = []
    for t in range(n_titles):
        title = f"Article {t}"
        for j in range(r.randint(2, 3)):
            ds = sorted(base + timedelta(days=r.randrange(0, 2500) + 400 * v + r.randrange(0, 300)) for v in range(3))
            ans = [f"ans{t}x{j}x{v}" for v in range(3)]
            txt = lambda v: f"{title} fact {j} version rev={ds[v].toordinal()} states value {ans[v]}."
            outs = [(txt(1), ds[1].strftime("%Y-%m-%d"), ans[1]), (txt(0), ds[0].strftime("%Y-%m-%d"), ans[0])]
            qs.append(dict(question=f"What is the status of [{title}] item {j}?", title=title, cur=txt(2), cur_date=ds[2].strftime("%Y-%m-%d"), cur_ans=ans[2], outs=outs))
    return qs

if DRYRUN:
    qas = make_fake_qas()
else:
    from datasets import load_dataset, get_dataset_config_names
    try: ds = load_dataset("russwest404/HoH-QAs")
    except Exception as e:
        print("default load failed:", repr(e)[:300]); cfgs = get_dataset_config_names("russwest404/HoH-QAs"); ds = load_dataset("russwest404/HoH-QAs", cfgs[0])
    print(ds); split = "train" if "train" in ds else list(ds.keys())[0]; raw_rows = ds[split]; print("columns:", raw_rows.column_names)
    def norm(r):
        try:
            doc = _maybe_json(r["document"]); old = _maybe_json(r["outdated_infos"])
            cur_date = _date(r["last_modified_time"]); cur = r["evidence"]
            title = doc.get("title", "") if isinstance(doc, dict) else str(doc)
            outs = []
            for o in (old or []):
                od = _date(o.get("last_modified_time"))
                if od and cur_date and od < cur_date and o.get("evidence") and o["evidence"].strip() != cur.strip():
                    outs.append((o["evidence"], od, str(o.get("answer", ""))))
            if not (cur and cur_date and outs and r["question"]): return None
            outs.sort(key=lambda t: t[1], reverse=True)
            return dict(question=r["question"], title=title, cur=cur, cur_date=cur_date, cur_ans=str(r.get("answer", "")), outs=outs[:2])
        except KeyError as e:
            raise KeyError(f"Field {e} not found. Columns are: {raw_rows.column_names}. Send me this message.")
    qas = [q for q in (norm(r) for r in raw_rows) if q]

all_dates = sorted({q["cur_date"] for q in qas} | {d for q in qas for _, d, _ in q["outs"]})
def _h(t, salt=""): return int(hashlib.md5((salt + t).encode()).hexdigest(), 16)
def is_eval_title(t): return _h(t) % 10 < HOLD10                     # articles never used to train the judge
def is_dev_title(t): return _h(t, "dev|") % 2 == 0                   # dev and test halves use DIFFERENT articles
qas_eval = [q for q in qas if is_eval_title(q["title"])]; qas_train = [q for q in qas if not is_eval_title(q["title"])]
qas_dev = [q for q in qas_eval if is_dev_title(q["title"])]; qas_te = [q for q in qas_eval if not is_dev_title(q["title"])]
assert not ({q["title"] for q in qas_dev} & {q["title"] for q in qas_te}) and not ({q["title"] for q in qas_eval} & {q["title"] for q in qas_train})
print("usable QAs:", len(qas), "| judge-train:", len(qas_train), "| dev:", len(qas_dev), "| test:", len(qas_te))

def build_instances(qas_, n, n_distr, seed=0):
    rnd = random.Random(seed); by_title = defaultdict(list)
    for i, q in enumerate(qas_): by_title[q["title"]].append(i)
    order = list(range(len(qas_))); rnd.shuffle(order); qdate = max(q["cur_date"] for q in qas)
    insts = []
    for i in order:
        if len(insts) >= n: break
        q = qas_[i]; used = {q["cur"].strip()} | {t.strip() for t, _, _ in q["outs"]}
        docs = [dict(text=q["cur"], title=q["title"], date=q["cur_date"], kind="pos")]
        for t, d, _ in q["outs"]: docs.append(dict(text=t, title=q["title"], date=d, kind="old"))
        same = [j for j in by_title[q["title"]] if j != i]; rnd.shuffle(same)
        rand = [rnd.randrange(len(qas_)) for _ in range(n_distr * 3)]; nd = 0
        for j in same + rand:
            if nd >= n_distr: break
            o = qas_[j]
            if j == i or o["cur"].strip() in used: continue
            used.add(o["cur"].strip()); docs.append(dict(text=o["cur"], title=o["title"], date=o["cur_date"], kind="distr")); nd += 1
        if nd < n_distr: continue
        rnd.shuffle(docs)
        insts.append(dict(query=q["question"], qdate=qdate, docs=docs, ans_cur=q["cur_ans"], ans_old=[a for _, _, a in q["outs"]], title=q["title"]))
    return insts
_nh = N_QUERIES // 2
insts_dev = build_instances(qas_dev, _nh, N_DISTR, SEED); insts_te = build_instances(qas_te, _nh, N_DISTR, SEED + 1)
_n = min(len(insts_dev), len(insts_te)); insts_dev, insts_te = insts_dev[:_n], insts_te[:_n]      # equal-sized halves
insts = insts_dev + insts_te; N = len(insts); dev, test = list(range(0, _n)), list(range(_n, N))
print(f"instances: {N} (dev {len(dev)} / test {len(test)}), pool size {len(insts[0]['docs'])}")
NOLD = [sum(d["kind"] == "old" for d in inst["docs"]) for inst in insts]
# reproducibility: save the exact pools (query, article title, per-passage title/kind/date, answers) without the passage texts
import gzip as _gz
with _gz.open("pools_" + TAG.split("_g")[0] + ".json.gz", "wt") as _f:
    json.dump([dict(split="dev" if k < len(dev) else "test", query=i["query"], title=i["title"], ans_cur=i["ans_cur"], ans_old=i["ans_old"],
                    docs=[dict(title=d["title"], kind=d["kind"], date=str(d["date"])) for d in i["docs"]]) for k, i in enumerate(insts)], _f)

# ---------------- corruption ----------------
FMT = "%Y-%m-%d"
def shift(d, days): return (datetime.strptime(d, FMT) + timedelta(days=int(days))).strftime(FMT)
LATE = all_dates[int(0.9 * len(all_dates)):]                                       # "re-save" dates: the newest 10% of the timeline
CRAWLS = [all_dates[min(len(all_dates) - 1, int(q * (len(all_dates) - 1)))] for q in (0.5, 0.65, 0.8, 0.9, 1.0)]   # 5 crawl snapshots
def corrupt(inst, mode, rnd):
    docs = inst["docs"]; dates = [d["date"] for d in docs]
    if mode == "clean":      return dates
    if mode == "missing":    return [None] * len(docs)
    if mode == "uniform":    return [inst["qdate"]] * len(docs)
    if mode == "random":     return [rnd.choice(all_dates) for _ in docs]
    if mode == "noise_30d":  return [shift(d, rnd.gauss(0, 30)) for d in dates]
    if mode == "noise_180d": return [shift(d, rnd.gauss(0, 180)) for d in dates]
    if mode == "resave":     return [rnd.choice(LATE) if rnd.random() < 0.5 else d for d in dates]         # modified date jumps to a re-save date
    if mode == "batch":      d0 = rnd.choice(all_dates); return [d0] * len(docs)                            # one ingestion date for the whole pool
    if mode == "crawl":      return [next((c for c in CRAWLS if c >= d), CRAWLS[-1]) for d in dates]        # date = first crawl snapshot after the edit (many ties)
    if mode == "year":       return [d[:4] + "-01-01" for d in dates]                                       # coarse granularity
    if mode == "month":      return [d[:7] + "-01" for d in dates]
    if mode == "swapped":
        pos = next(d["date"] for d in docs if d["kind"] == "pos"); old = max(d["date"] for d in docs if d["kind"] == "old"); out = []
        for d in docs: out.append(old if d["kind"] == "pos" else pos if (d["kind"] == "old" and d["date"] == old) else d["date"])
        return out
    raise ValueError(mode)
def fmt_doc(d, date):
    s = f"Title: {d['title']}\n{d['text']}"; return s + (f"\nTimestamp: {date}" if date else "")
def fmt_query(inst): return f"{inst['query']}\nTimestamp: {inst['qdate']}"

PASSES_CORE = [("clean", "clean", 0), ("missing", "missing", 0), ("swapped", "swapped", 0), ("random0", "random", 1), ("random1", "random", 2), ("noise30", "noise_30d", 3), ("noise180", "noise_180d", 4)]
PASSES_NEW = [("resave", "resave", 5), ("batch", "batch", 6), ("crawl", "crawl", 7), ("year", "year", 8), ("month", "month", 9)]
PASSES = PASSES_CORE + (PASSES_NEW if PASS_SET == "full" else [])
DT = {}
for name, base, seed in PASSES:
    rnd_ = random.Random(seed * 1000 + 123); DT[name] = [corrupt(inst, base, rnd_) for inst in insts]
MODES5 = [p[0] for p in PASSES]
FMT_ = "%Y-%m-%d"
DD = {m: [[datetime.strptime(x, FMT_).toordinal() if x else None for x in DT[m][k]] for k in range(N)] for m in DT}
QD = [datetime.strptime(inst["qdate"], FMT_).toordinal() for inst in insts]
KINDS = [[d["kind"] for d in inst["docs"]] for inst in insts]
print("example query:", fmt_query(insts[0]).replace("\n", " | "))
for m in ["clean", "swapped", "crawl", "resave"]:
    if m in DT: print(f"--- {m}:", [(d["kind"][:3], dt) for d, dt in zip(insts[0]["docs"][:4], DT[m][0][:4])])

# ---------------- re-rankers (scores are cached, so repeated runs and seed runs skip this) ----------------
if not DRYRUN:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
    import torch.nn.functional as F

INSTR_DEFAULT = "Given a web search query, retrieve relevant passages that answer the query"
def load_scorer(rn):
    """returns (score_fn(pairs)->list[float], cleanup()). Models are loaded only when the cache is missing and freed afterwards."""
    if DRYRUN:
        def fn(pairs):
            out = []
            for q, d in pairs:
                t = d.split("\n")[0][7:]; rel = 1.0 if f"[{t}]" in q else 0.0
                out.append(0.6 * rel + 0.4 * ((zlib.crc32((q + d).encode()) % 1000) / 1000.0))
            return out
        return fn, (lambda: None)
    if rn.startswith("qwen3-"):
        name = {"qwen3-0.6B": "Qwen/Qwen3-Reranker-0.6B", "qwen3-4B": "Qwen/Qwen3-Reranker-4B"}[rn]
        tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        mdl = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
        t_no, t_yes = tok.convert_tokens_to_ids("no"), tok.convert_tokens_to_ids("yes")
        PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        pre, suf = tok.encode(PREFIX, add_special_tokens=False), tok.encode(SUFFIX, add_special_tokens=False)
        MAXLEN, BS = (768, 8) if "4B" in rn else (1024, 32)
        @torch.no_grad()
        def fn(pairs):
            out = []
            for i in range(0, len(pairs), BS):
                texts = [f"<Instruct>: {INSTR_DEFAULT}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[i:i+BS]]
                enc = tok(texts, padding=False, truncation="longest_first", return_attention_mask=False, max_length=MAXLEN - len(pre) - len(suf))
                enc["input_ids"] = [pre + ids + suf for ids in enc["input_ids"]]
                enc = tok.pad(enc, padding=True, return_tensors="pt").to("cuda")
                lg = mdl(**enc).logits[:, -1, :]
                out += torch.log_softmax(torch.stack([lg[:, t_no], lg[:, t_yes]], 1).float(), 1)[:, 1].exp().cpu().tolist()
            return out
        def cleanup():
            nonlocal mdl; del mdl; gc.collect(); torch.cuda.empty_cache()
        return fn, cleanup
    if rn == "bge-v2-m3":
        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        mdl = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16).cuda().eval()
        @torch.no_grad()
        def fn(pairs):
            out = []
            for i in range(0, len(pairs), 32):
                q = [p[0] for p in pairs[i:i+32]]; d = [p[1] for p in pairs[i:i+32]]
                enc = tok(q, d, padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                out += torch.sigmoid(mdl(**enc).logits.view(-1).float()).cpu().tolist()
            return out
        def cleanup():
            nonlocal mdl; del mdl; gc.collect(); torch.cuda.empty_cache()
        return fn, cleanup
    raise ValueError(rn)

SIGN = f"N{N}_d{N_DISTR}_h{HOLD10}_{PASS_SET}_{int(SCORE_NEW_DA)}" + ("_DRY" if DRYRUN else "")
SC = {}
lens_ = [len(inst["docs"]) for inst in insts]; cut_ = np.cumsum([0] + lens_)
with stage("rerank_scoring"):
    for rn in RR_SEL:
        path = f"sc_{rn}_{SIGN}.pkl"
        if os.path.exists(path):
            SC[rn] = pickle.load(open(path, "rb")); print(rn, "loaded cached scores"); continue
        fn, cleanup = load_scorer(rn); SC[rn] = {}; t0 = time.time()
        for name, base, seed in PASSES:
            if name not in [p[0] for p in PASSES_CORE] and not SCORE_NEW_DA: SC[rn][name] = None; continue
            pairs = [(fmt_query(inst), fmt_doc(d, dt)) for inst, dts in zip(insts, DT[name]) for d, dt in zip(inst["docs"], dts)]
            flat = np.array(fn(pairs)); SC[rn][name] = [flat[cut_[k]:cut_[k+1]] for k in range(N)]
            print(f"{name:8s} {rn:11s} {time.time()-t0:.0f}s")
        pickle.dump(SC[rn], open(path, "wb")); cleanup()
RERANKERS = list(SC.keys())
R0 = {rn: SC[rn]["missing"] for rn in RERANKERS}
KS = [3] + ([K_EXTRA] if K_EXTRA and K_EXTRA > 3 else [])
TOPK = {kk: {rn: [list(np.argsort(-R0[rn][k])[:kk]) for k in range(N)] for rn in RERANKERS} for kk in KS}
for rn in RERANKERS: print(rn, "| pos in date-free top-3:", np.mean([any(KINDS[k][i] == "pos" for i in TOPK[3][rn][k]) for k in range(N)]).round(3))

# ---------------- judge ----------------
jm_fake_rng = None
if not DRYRUN:
    JUDGE = JUDGE_MODEL
    jtok = AutoTokenizer.from_pretrained(JUDGE, padding_side="left")
    judge = AutoModelForCausalLM.from_pretrained(JUDGE, torch_dtype=torch.float16).cuda()
    jidA, jidB = jtok.encode("A", add_special_tokens=False)[0], jtok.encode("B", add_special_tokens=False)[0]
    def jprompt(q, a, b):
        msg = [{"role": "user", "content": "Two versions of a statement answering the same question were written at different times. Which version was written more recently?\n\nQuestion: " + q + "\n\nA: " + a[:1000] + "\n\nB: " + b[:1000] + "\n\nAnswer with only the letter A or B."}]
        return jtok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    @torch.no_grad()
    def jmargins(triples, bs=16):
        judge.eval(); out = []
        for i in range(0, len(triples), bs):
            enc = jtok([jprompt(*t) for t in triples[i:i+bs]], return_tensors="pt", padding=True).to("cuda")
            lg = judge(**enc, logits_to_keep=1).logits[:, -1, :]
            out += (lg[:, jidA] - lg[:, jidB]).float().cpu().tolist()
        return out
    def train_judge():
        from peft import LoraConfig, get_peft_model
        from torch.optim.lr_scheduler import LambdaLR
        global judge
        tr_pairs = [(q["question"], q["cur"], t) for q in qas_train for t, _, _ in q["outs"]]
        BS, ACC, LR = ((4, 4, 1e-4) if "1.5B" in JUDGE else (2, 8, 1e-4)); rt = random.Random(JSEED + 1); ex = []
        for q, n, o in rt.sample(tr_pairs, min(3000, len(tr_pairs))): ex += [(q, n, o, 0), (q, o, n, 1)]
        rt.shuffle(ex)
        judge.config.use_cache = False; judge.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); judge.enable_input_require_grads()
        torch.manual_seed(JSEED); judge = get_peft_model(judge, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        opt = torch.optim.AdamW([p for p in judge.parameters() if p.requires_grad], lr=LR); tot = len(ex) // (BS * ACC)
        sched = LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / max(tot, 1))); scaler = torch.amp.GradScaler("cuda")
        judge.train(); cnt = step = 0; t0 = time.time()
        for i in range(0, len(ex) - BS + 1, BS):
            b = ex[i:i+BS]; enc = jtok([jprompt(q, a, c) for q, a, c, _ in b], return_tensors="pt", padding=True).to("cuda")
            y = torch.tensor([l for *_, l in b], device="cuda")
            with torch.amp.autocast("cuda", dtype=torch.float16): lg = judge(**enc, logits_to_keep=1).logits[:, -1, :]
            loss = F.cross_entropy(torch.stack([lg[:, jidA], lg[:, jidB]], 1).float(), y); scaler.scale(loss / ACC).backward(); cnt += 1
            if cnt % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(judge.parameters(), 1.0); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 50 == 0: print(f"update {step}/{tot} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
        judge.eval(); judge.config.use_cache = True
    with stage("judge_training"):
        print("Training the judge from scratch (about 15 min for 1.5B, 30 min for 3B)."); train_judge(); judge.eval()
    try: judge.save_pretrained("judge_lora_" + TAG)
    except Exception as e: print("save skipped:", repr(e)[:100])
def jm(triples, which):
    """judge margins (A minus B logit). which = 'ft' (adapter on) or 'zs' (adapter off)."""
    if DRYRUN:
        out = []
        for q, a, b in triples:
            ra, rb = int(re.search(r"rev=(\d+)", a).group(1)), int(re.search(r"rev=(\d+)", b).group(1))
            r = random.Random(zlib.crc32((q + a + b + which).encode())); sgn = (ra > rb) - (ra < rb)
            out.append(sgn * (2.0 if which == "ft" else 1.0) + r.gauss(0, 2.0))
        return out
    with (contextlib.nullcontext() if which == "ft" else judge.disable_adapter()): return jmargins(triples)

JC = {"zs": {}, "ft": {}}
def ensure_judge(kk, rn):
    need = set()
    for k in range(N):
        top = TOPK[kk][rn][k]
        for x in range(kk):
            for y in range(x + 1, kk):
                a, b = sorted((top[x], top[y]))
                if (k, a, b) not in JC["ft"]: need.add((k, a, b))
    need = sorted(need); trip = []
    for k, a, b in need:
        d = insts[k]["docs"]; q = insts[k]["query"]; trip += [(q, d[a]["text"], d[b]["text"]), (q, d[b]["text"], d[a]["text"])]
    if not need: return
    for jn in ["ft", "zs"]:
        ms = jm(trip, jn)
        for j, key in enumerate(need): JC[jn][key] = (ms[2*j] - ms[2*j+1]) / 2      # >0: lower-index doc is more recent
    print(f"K={kk} {rn} new judged pairs:", len(need))
with stage("judge_pairs_K3"):
    for rn in RERANKERS: ensure_judge(3, rn)
def build_S(kk, rn, jn):
    out = []
    for k in range(N):
        top = TOPK[kk][rn][k]; d = {}
        for x in range(kk):
            for y in range(x + 1, kk):
                a, b = top[x], top[y]; lo, hi = (a, b) if a < b else (b, a); s = JC[jn][(k, lo, hi)]; d[(a, b)] = s if a < b else -s
        out.append(d)
    return out
S, SIG = {}, {}
def refresh_S(kk):
    S[kk] = {rn: {jn: build_S(kk, rn, jn) for jn in JC} for rn in RERANKERS}
    SIG[kk] = {rn: {jn: float(np.std([v for d in S[kk][rn][jn] for v in d.values()])) for jn in JC} for rn in RERANKERS}
refresh_S(3)
JACC = []
for rn in RERANKERS:
    for jn in JC:
        hit = tot = 0
        for k in range(N):
            for (a, b), s in S[3][rn][jn][k].items():
                ka, kb = KINDS[k][a], KINDS[k][b]
                if {ka, kb} == {"pos", "old"}: tot += 1; hit += (s > 0) == (ka == "pos")
        JACC.append(dict(reranker=rn, judge=jn, accuracy=round(100 * hit / max(tot, 1), 1), n=tot)); print(JACC[-1])
pd.DataFrame(JACC).to_csv(f"exp4_judge_accuracy_{TAG}.csv", index=False)

# ---------------- fusion, baselines, grid ----------------
def fuse_g(rn, mode, idx, jn, alpha, lam, theta, use_meta=True, gate=True, dscale=30.0, wc=1.0, tau=0.0, kk=3):
    S_ = S[kk][rn][jn]; SIG_ = SIG[kk][rn][jn]; TOP = TOPK[kk][rn]; R = R0[rn]; orders = {}
    for k in idx:
        r0 = R[k]; top = TOP[k]; dd = DD[mode][k]; z = defaultdict(float)
        for (a, b), s in S_[k].items():
            sh = math.tanh(s / SIG_); da, db = dd[a], dd[b]
            delta = (da - db) if (da is not None and db is not None) else 0
            v = math.copysign(math.tanh(max(abs(delta) - tau, 0.0) / dscale), delta) if (use_meta and delta != 0) else 0.0
            w = 1.0
            if gate and delta != 0 and (delta > 0) != (s > 0) and abs(s) / SIG_ > theta: w = 0.0
            z[a] += wc * w * v + lam * sh; z[b] -= wc * w * v + lam * sh
        fin = {i: r0[i] + alpha * z[i] for i in top}
        orders[k] = sorted(top, key=lambda i: -fin[i]) + [i for i in np.argsort(-r0) if i not in top]
    return orders
def decay_orders(rn, mode, idx, gamma, sigma):      # TempRALM/HoH-style: relevance x Gaussian recency decay on ALL passages
    out = {}
    for k in idx:
        r0 = R0[rn][k]; dd = DD[mode][k]
        f = np.array([1.0 if d is None else gamma + (1 - gamma) * math.exp(-max(QD[k] - d, 0) ** 2 / (2 * sigma ** 2)) for d in dd])
        out[k] = list(np.argsort(-(r0 * f)))
    return out
def base_orders(scores, idx): return {k: list(np.argsort(-scores[k])) for k in idx}
def p1_of(orders, idx): return np.array([float(KINDS[k][orders[k][0]] == "pos") for k in idx])

if GRID == "small": AL3, LA2, TH2, LA3, TAUS = [.02, .05], [0, .5, 1], [0, 1, 1.5, 3], [.5, 1, 2], [30, 365]
else:               AL3, LA2, TH2, LA3, TAUS = [.02, .05, .1], [0, .5, 1, 2], [0, .5, 1, 1.5, 2, 3], [.5, 1, 2], [30, 90, 365]
def make_cfgs(rn):
    c = []
    for g in [.5, .7, .9]:
        for s in [30, 60, 120]: c.append(("decay", dict(gamma=g, sigma=s), lambda m, idx, g=g, s=s: decay_orders(rn, m, idx, g, s)))
    for a in AL3: c.append(("metadata_only", dict(alpha=a), lambda m, idx, a=a: fuse_g(rn, m, idx, "zs", a, 0, 0, True, False)))
    for a in AL3:      # robust date prior: the date term only fires when the date gap exceeds tau days
        for t in TAUS: c.append(("metadata_thresh", dict(alpha=a, tau=t), lambda m, idx, a=a, t=t: fuse_g(rn, m, idx, "zs", a, 0, 0, True, False, tau=t)))
    for jn in ["zs", "ft"]:
        for a in AL3: c.append((f"content_{jn}", dict(alpha=a), lambda m, idx, a=a, jn=jn: fuse_g(rn, m, idx, jn, a, 1, 0, False, False)))
        for a in AL3:
            for l in LA2:
                for t in TH2: c.append((f"gated_{jn}", dict(alpha=a, lam=l, theta=t), lambda m, idx, a=a, l=l, t=t, jn=jn: fuse_g(rn, m, idx, jn, a, l, t, True, True)))
    for jn in ["zs", "ft"]:
        for a in AL3:
            for l in LA3: c.append((f"nogate_{jn}", dict(alpha=a, lam=l), lambda m, idx, a=a, l=l, jn=jn: fuse_g(rn, m, idx, jn, a, l, 0, True, False)))
    return c
def da_p1(rn, m, idx): return p1_of(base_orders(SC[rn][m], idx), idx) if SC[rn].get(m) is not None else np.full(len(idx), np.nan)
CF, V = {}, {}
with stage("grid_search"):
    for rn in RERANKERS:
        CF[rn] = make_cfgs(rn); V[rn] = []; t0 = time.time()
        for fam, kw, fn in CF[rn]:
            V[rn].append((fam, kw, {m: p1_of(fn(m, dev), dev) for m in MODES5}, {m: p1_of(fn(m, test), test) for m in MODES5}))
        V[rn] += [("date_free", {}, {m: p1_of(base_orders(R0[rn], dev), dev) for m in MODES5}, {m: p1_of(base_orders(R0[rn], test), test) for m in MODES5}),
                  ("date_aware", {}, {m: da_p1(rn, m, dev) for m in MODES5}, {m: da_p1(rn, m, test) for m in MODES5})]
        print(rn, "configs:", len(V[rn]), f"{time.time()-t0:.0f}s")

# ---------------- statistics helpers ----------------
BAD = {"swapped": 1/3, "missing": 1/3, "random0": 1/6, "random1": 1/6}
SCEN = [(f"mix rho={r:.1f}", r, BAD) for r in [0, .1, .2, .3, .4, .5]] + [("ALL swapped", 1.0, {"swapped": 1.0}), ("ALL random", 1.0, {"random0": .5, "random1": .5})]
mixv = lambda v, rho, bw: (1 - rho) * v["clean"] + rho * sum(w * v[m] for m, w in bw.items())
rng = np.random.default_rng(0)
def boot(a, b, n=5000):
    d = a - b; idx = rng.integers(0, len(d), (n, len(d))); mm = d[idx].mean(1)
    return 100 * d.mean(), 100 * np.percentile(mm, 2.5), 100 * np.percentile(mm, 97.5)
def ci_str(a, b):
    if np.isnan(a).any() or np.isnan(b).any(): return "n/a"
    d, lo, hi = boot(a, b); return f"{d:+.1f} [{lo:+.1f},{hi:+.1f}]"
FAMS = ["date_free", "date_aware", "decay", "metadata_only", "metadata_thresh", "content_zs", "content_ft", "gated_zs", "gated_ft", "nogate_zs", "nogate_ft"]
def best(rn, fam, rho, bw): return max([v for v in V[rn] if v[0] == fam], key=lambda v: np.nanmean(mixv(v[2], rho, bw)))

# ---------------- Table 1: per-scenario tuned (optimistic bound) ----------------
T1 = []
for rn in RERANKERS:
    print(f"\n===== {rn}: expected test P@1 (%), settings tuned separately for each scenario =====")
    for name, rho, bw in SCEN:
        te = {f: mixv(best(rn, f, rho, bw)[3], rho, bw) for f in FAMS}
        row = dict(reranker=rn, scenario=name, **{f: round(100 * np.nanmean(te[f]), 1) for f in FAMS})
        for ref in ["metadata_only", "metadata_thresh", "decay", "content_ft", "nogate_ft"]: row[f"gated_ft_minus_{ref}"] = ci_str(te["gated_ft"], te[ref])
        T1.append(row); print(f"{name:13s}| " + " | ".join(f"{f} {row[f]:.1f}" for f in FAMS))
pd.DataFrame(T1).to_csv(f"exp4_table1_{TAG}.csv", index=False)

# ---------------- Table 2: ONE setting tuned at rho=0.3, applied everywhere (deployment view) ----------------
SC2 = [("clean", 0.0, BAD), ("mix rho=0.3", .3, BAD), ("mix rho=0.5", .5, BAD), ("ALL random", 1.0, {"random0": .5, "random1": .5}), ("ALL swapped", 1.0, {"swapped": 1.0})]
SC3 = [("ALL noise30", 1.0, {"noise30": 1.0}), ("ALL noise180", 1.0, {"noise180": 1.0})]
REAL_BW = {"resave": .3, "batch": .2, "crawl": .3, "noise180": .2}
SC_REAL = [(f"ALL {m}", 1.0, {m: 1.0}) for m in ["resave", "batch", "crawl", "year", "month"]] + [("real mix rho=0.3", .3, REAL_BW), ("real mix rho=0.5", .5, REAL_BW)] if PASS_SET == "full" else []
_sw = {"swapped": .6, "missing": .1, "random0": .15, "random1": .15}; _mi = {"missing": .6, "swapped": .1, "random0": .15, "random1": .15}; _ra = {"random0": .35, "random1": .35, "swapped": .15, "missing": .15}
SC_ALT = [(f"{nm} rho={r}", r, bw) for r in (.3, .5) for nm, bw in (("swapped-heavy", _sw), ("missing-heavy", _mi), ("random-heavy", _ra))]
CHOSEN = {rn: {f: best(rn, f, .3, BAD) for f in FAMS} for rn in RERANKERS}
REFS = ["metadata_only", "metadata_thresh", "decay", "content_ft", "date_aware", "gated_zs", "nogate_ft"]
def eval_scen(rn, scen, refs=REFS):
    rows = []; vecs = {}
    for name, rho, bw in scen:
        te = {f: mixv(CHOSEN[rn][f][3], rho, bw) for f in FAMS}; vecs[name] = te
        row = dict(reranker=rn, scenario=name, **{f: round(100 * np.nanmean(te[f]), 1) for f in FAMS})
        for ref in refs: row[f"gated_ft_minus_{ref}"] = ci_str(te["gated_ft"], te[ref])
        rows.append(row)
    return rows, vecs
T2 = []; VEC2 = {}
for rn in RERANKERS:
    print(f"\n===== {rn}: ONE setting tuned at rho=0.3 =====\nCHOSEN:", {f: CHOSEN[rn][f][1] for f in FAMS})
    rows, VEC2[rn] = eval_scen(rn, SC2); T2 += rows
    df = pd.DataFrame(rows).set_index("scenario"); print(df[FAMS].to_string())
    print("AVERAGE:", df[FAMS].mean().round(1).to_dict()); print("WORST  :", df[FAMS].min().round(1).to_dict())
pd.DataFrame(T2).to_csv(f"exp4_table2_{TAG}.csv", index=False)
extra_tabs = {}
for tag_, scen in [("mild", SC3), ("real", SC_REAL), ("mixsens", SC_ALT)]:
    if not scen: continue
    rows = []
    for rn in RERANKERS: r_, v_ = eval_scen(rn, scen); rows += r_; VEC2[rn].update(v_)
    extra_tabs[tag_] = rows; pd.DataFrame(rows).to_csv(f"exp4_{tag_}_{TAG}.csv", index=False)
    print(f"\n===== {tag_} (same chosen settings) ====="); print(pd.DataFrame(rows)[["reranker", "scenario"] + FAMS].to_string(index=False))

# ---------------- intervals for AVERAGE and WORST over the five scenarios of Table 2 ----------------
AVG = []
for rn in RERANKERS:
    names = [s[0] for s in SC2]; B = 2000; idx = rng.integers(0, len(test), (B, len(test)))
    M = {f: np.stack([VEC2[rn][n][f] for n in names]) for f in FAMS if f != "date_aware"}      # (scenario, query)
    means = {f: M[f][:, idx].mean(2) for f in M}                                                # (scenario, B)
    pt = {f: M[f].mean(1) for f in M}
    for f in M:
        row = dict(reranker=rn, family=f, average=round(100 * pt[f].mean(), 1), worst=round(100 * pt[f].min(), 1))
        a, w = means[f].mean(0), means[f].min(0)
        row["average_ci"] = f"[{100*np.percentile(a,2.5):.1f},{100*np.percentile(a,97.5):.1f}]"; row["worst_ci"] = f"[{100*np.percentile(w,2.5):.1f},{100*np.percentile(w,97.5):.1f}]"
        if f != "gated_ft":
            da = means["gated_ft"].mean(0) - means[f].mean(0); dw = means["gated_ft"].min(0) - means[f].min(0)
            row["gated_minus_this_average"] = f"{100*(pt['gated_ft'].mean()-pt[f].mean()):+.1f} [{100*np.percentile(da,2.5):+.1f},{100*np.percentile(da,97.5):+.1f}]"
            row["gated_minus_this_worst"] = f"{100*(pt['gated_ft'].min()-pt[f].min()):+.1f} [{100*np.percentile(dw,2.5):+.1f},{100*np.percentile(dw,97.5):+.1f}]"
        AVG.append(row)
pd.DataFrame(AVG).to_csv(f"exp4_average_ci_{TAG}.csv", index=False); print("\n", pd.DataFrame(AVG).to_string(index=False))

# per-query correctness for the joint bootstrap over queries and judge seeds (see merge_seeds.py)
perq = {}
for rn in RERANKERS:
    for name, te in VEC2[rn].items():
        for f in FAMS:
            if not np.isnan(te[f]).any(): perq[f"{rn}@{name}@{f}"] = te[f].astype(np.float32)
np.savez_compressed(f"perq_{TAGP}.npz", **perq, _test_titles=np.array([insts[k]["title"] for k in test]))

# ---------------- extras ----------------
if RUN_EXTRAS:
    EX = {}
    with stage("extras"):
        if not RUN_ADAPT_ONLY:
            # (a) sensitivity to lambda for BOTH variants (gate and no gate), same alpha/theta as the chosen setting
            LS = [0, .25, .5, 1, 2, 4]; rows = []
            for rn in RERANKERS:
                for fam in ["gated_ft", "nogate_ft"]:
                    kw0 = dict(CHOSEN[rn][fam][1])
                    for l in LS:
                        kw = dict(kw0, lam=l); res = {}
                        for name, rho, bw in SC2:
                            v = {m: p1_of(fuse_g(rn, m, test, "ft", kw["alpha"], l, kw.get("theta", 0), True, fam == "gated_ft"), test) for m in set(["clean"]) | set(bw)}
                            res[name] = 100 * mixv(v, rho, bw).mean()
                        rows.append(dict(reranker=rn, family=fam, lam=l, **{k: round(x, 1) for k, x in res.items()}, average=round(np.mean(list(res.values())), 1), worst=round(min(res.values()), 1)))
            EX["lambda_sweep"] = rows; pd.DataFrame(rows).to_csv(f"exp4_lambda_sweep_{TAG}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
            # (b) sensitivity to the tanh date scale (days)
            rows = []
            for rn in RERANKERS:
                for fam in ["metadata_only", "gated_ft"]:
                    kw = CHOSEN[rn][fam][1]
                    for ds_ in [7, 30, 90, 365]:
                        res = {}
                        for name, rho, bw in SC2 + SC3:
                            v = {m: p1_of(fuse_g(rn, m, test, "ft" if fam == "gated_ft" else "zs", kw["alpha"], kw.get("lam", 0), kw.get("theta", 0), True, fam == "gated_ft", dscale=ds_), test) for m in set(["clean"]) | set(bw)}
                            res[name] = 100 * mixv(v, rho, bw).mean()
                        rows.append(dict(reranker=rn, family=fam, date_scale_days=ds_, **{k: round(x, 1) for k, x in res.items()}))
            EX["date_scale"] = rows; pd.DataFrame(rows).to_csv(f"exp4_date_scale_{TAG}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
            # (c) breakdown by number of outdated versions (1 or 2)
            rows = []
            for rn in RERANKERS:
                for nold in sorted(set(NOLD[k] for k in test)):
                    sel = [i for i, k in enumerate(test) if NOLD[k] == nold]
                    for name, rho, bw in SC2:
                        row = dict(reranker=rn, n_outdated=nold, n_queries=len(sel), scenario=name)
                        for f in ["date_free", "metadata_only", "metadata_thresh", "content_ft", "gated_ft", "nogate_ft"]:
                            row[f] = round(100 * mixv(CHOSEN[rn][f][3], rho, bw)[sel].mean(), 1)
                        rows.append(row)
            EX["n_outdated"] = rows; pd.DataFrame(rows).to_csv(f"exp4_n_outdated_{TAG}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
            # (d) larger window K (settings are NOT re-tuned)
            if len(KS) > 1:
                K2 = KS[1]
                for rn in RERANKERS: ensure_judge(K2, rn)
                refresh_S(K2); rows = []
                for rn in RERANKERS:
                    for fam in ["gated_ft", "nogate_ft", "content_ft", "metadata_only"]:
                        kw = CHOSEN[rn][fam][1]; jn = "zs" if fam == "metadata_only" else "ft"
                        for kk in (3, K2):
                            res = {}
                            for name, rho, bw in SC2:
                                v = {m: p1_of(fuse_g(rn, m, test, jn, kw["alpha"], {"gated_ft": kw.get("lam"), "nogate_ft": kw.get("lam"), "content_ft": 1, "metadata_only": 0}[fam], kw.get("theta", 0), fam not in ("content_ft",), fam == "gated_ft", kk=kk), test) for m in set(["clean"]) | set(bw)}
                                res[name] = 100 * mixv(v, rho, bw).mean()
                            rows.append(dict(reranker=rn, family=fam, K=kk, **{k: round(x, 1) for k, x in res.items()}, average=round(np.mean(list(res.values())), 1), worst=round(min(res.values()), 1)))
                EX["K"] = rows; pd.DataFrame(rows).to_csv(f"exp4_window_K_{TAG}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))
        # (e) date-reliability estimator ("adaptive_ft"): trust in dates is estimated from the unlabelled collection
        def agree(rn, mode, idx, th0=1.0):
            S_ = S[3][rn]["ft"]; SG = SIG[3][rn]["ft"]; hit = tot = 0
            for k in idx:
                dd = DD[mode][k]
                for (a, b), s in S_[k].items():
                    if dd[a] is None or dd[b] is None or dd[a] == dd[b] or abs(s) / SG < th0: continue
                    if insts[k]["docs"][a]["title"] != insts[k]["docs"][b]["title"]: continue      # only versions of the same document/URL (a deployable grouping)
                    tot += 1; hit += ((dd[a] - dd[b]) > 0) == (s > 0)
            return hit, tot
        rows = []
        for rn in RERANKERS:
            h_, t_ = agree(rn, "clean", dev); a_ref = h_ / max(t_, 1)
            AGT = {m: agree(rn, m, test) for m in MODES5 if m != "missing"}; AGT["missing"] = (0, 0)
            kw = CHOSEN[rn]["gated_ft"][1]; print(f"{rn}: reference agreement with CLEAN dates (dev, small labelled sample) = {a_ref:.3f} over {t_} confident pairs")
            for name, rho, bw in SC2 + SC3 + SC_REAL + SC_ALT:
                wts = {"clean": 1 - rho}
                for m, w in bw.items(): wts[m] = wts.get(m, 0) + rho * w
                hit = sum(w * AGT[m][0] for m, w in wts.items()); tot = sum(w * AGT[m][1] for m, w in wts.items())
                a_sc = hit / max(tot, 1e-9); wc = float(np.clip((a_sc - .5) / max(a_ref - .5, 1e-6), 0, 1))
                v = {m: p1_of(fuse_g(rn, m, test, "ft", kw["alpha"], kw["lam"], kw["theta"], True, True, wc=wc), test) for m in wts}
                ad = mixv(v, rho, bw) if rho < 1 else sum(w * v[m] for m, w in bw.items())
                te = {f: mixv(CHOSEN[rn][f][3], rho, bw) for f in ["gated_ft", "content_ft", "metadata_only", "nogate_ft"]}
                rows.append(dict(reranker=rn, scenario=name, est_agreement=round(a_sc, 3), est_trust_wc=round(wc, 2), adaptive_ft=round(100 * ad.mean(), 1),
                                 gated_ft=round(100 * te["gated_ft"].mean(), 1), nogate_ft=round(100 * te["nogate_ft"].mean(), 1), content_ft=round(100 * te["content_ft"].mean(), 1), metadata_only=round(100 * te["metadata_only"].mean(), 1),
                                 adaptive_minus_gated=ci_str(ad, te["gated_ft"]), adaptive_minus_content=ci_str(ad, te["content_ft"]), adaptive_minus_metadata=ci_str(ad, te["metadata_only"])))
        EX["adaptive"] = rows; pd.DataFrame(rows).to_csv(f"exp4_adaptive_{TAG}.csv", index=False); print(pd.DataFrame(rows).to_string(index=False))

# ---------------- position-controlled end-to-end test ----------------
def nrm(s): s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower()); s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def tok_f1(pred, gold):
    p, g = nrm(pred).split(), nrm(gold).split()
    if not p or not g: return float(p == g)
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0: return 0.0
    pr, rc = c / len(p), c / len(g); return 2 * pr * rc / (pr + rc)
if RUN_E2E_VARIANTS:
    with stage("e2e_variants"):
        GIDX = test[:E2E_N]
        GSC = [("clean", "clean"), ("random0", "ALL random"), ("swapped", "ALL swapped")] + ([("resave", "ALL resave"), ("noise180", "ALL noise180")] if PASS_SET == "full" else [])
        GFAMS = ["date_free", "date_aware", "decay", "metadata_only", "metadata_thresh", "content_ft", "gated_ft", "nogate_ft"]
        def fam_orders2(rn, fam, mode, idx):
            if fam == "date_free": return base_orders(R0[rn], idx)
            if fam == "date_aware": return base_orders(SC[rn][mode], idx) if SC[rn].get(mode) is not None else None
            kw = CHOSEN[rn][fam][1]; fn = [c[2] for c in CF[rn] if c[0] == fam and c[1] == kw][0]; return fn(mode, idx)
        def gen_prompt2(inst, doc_ids, dts, variant):
            ids = list(doc_ids[:3])
            if variant == "reversed": ids = ids[::-1]
            if variant == "top1": ids = ids[:1]
            ps = ["Passage " + str(j + 1) + ": " + inst["docs"][i]["text"] + (("\nTimestamp: " + dts[i]) if dts[i] else "") for j, i in enumerate(ids)]
            head = ("Answer the question using the passages below. If passages disagree, trust the one with the most recent Timestamp. Reply with only the short answer." if len(ids) > 1
                    else "Answer the question using the passage below. Reply with only the short answer.")
            user = head + "\n\n" + "\n\n".join(ps) + "\n\nQuestion: " + inst["query"]
            return user if DRYRUN else jtok.apply_chat_template([{"role": "user", "content": user}], add_generation_prompt=True, tokenize=False)
        def gen2(prompts, gm, bs=16):
            if DRYRUN:
                return [(re.search(r"value (\S+?)\.", p).group(1) if re.search(r"value (\S+?)\.", p) else "") for p in prompts]
            outs = []
            with torch.no_grad():
                for i in range(0, len(prompts), bs):
                    enc = jtok(prompts[i:i+bs], return_tensors="pt", padding=True).to("cuda")
                    if gm is None:
                        with judge.disable_adapter(): o = judge.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=jtok.pad_token_id)
                    else: o = gm.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=jtok.pad_token_id)
                    outs += jtok.batch_decode(o[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            return [s.strip() for s in outs]
        GENS = [("gen" + _short(JUDGE_MODEL), None)]
        if GEN_MODEL and GEN_MODEL != JUDGE_MODEL and not DRYRUN:
            torch.cuda.empty_cache(); GENS.append(("gen" + _short(GEN_MODEL), AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16).cuda().eval()))
        rows = []; CI = []; SAMPLES = []
        for ern in [r for r in E2E_RR if r in RERANKERS]:
            ORD = {(fam, mode): fam_orders2(ern, fam, mode, GIDX) for mode, _ in GSC for fam in GFAMS}
            for gname, gm in GENS:
                for variant in ["ranked", "reversed", "top1"]:
                    t0 = time.time(); PR2 = {(fam, mode): [gen_prompt2(insts[k], ORD[(fam, mode)][k], DT[mode][k], variant) for k in GIDX] for mode, _ in GSC for fam in GFAMS if ORD[(fam, mode)] is not None}
                    uniq = sorted({p for v_ in PR2.values() for p in v_}); G2 = dict(zip(uniq, gen2(uniq, gm))); res = {}
                    for (fam, mode), prs in PR2.items():
                        cur, stale, em, f1 = [], [], [], []
                        for k, p in zip(GIDX, prs):
                            g = nrm(G2[p]); ac = nrm(insts[k]["ans_cur"]); c = bool(ac) and ac in g
                            s_ = (not c) and any(bool(nrm(a)) and nrm(a) in g for a in insts[k]["ans_old"])
                            cur.append(float(c)); stale.append(float(s_)); em.append(float(g == ac)); f1.append(tok_f1(G2[p], insts[k]["ans_cur"]))
                        res[(fam, mode)] = tuple(np.array(x) for x in (cur, stale, em, f1))
                        if variant == "top1" and fam in ("gated_ft", "metadata_only") and mode in ("clean", "swapped") and ern == E2E_RR[0]:
                            for k, p in list(zip(GIDX, prs))[:50]: SAMPLES.append(dict(generator=gname, family=fam, scenario=mode, query=insts[k]["query"][:120], generation=G2[p][:80], answer_current=insts[k]["ans_cur"][:60], answers_outdated=" | ".join(a[:40] for a in insts[k]["ans_old"]), contains_current=int(bool(nrm(insts[k]["ans_cur"])) and nrm(insts[k]["ans_cur"]) in nrm(G2[p]))))
                    for mode, label in GSC:
                        for mi, mname in enumerate(["cur", "stale", "em", "f1"]):
                            rows.append(dict(reranker=ern, generator=gname, variant=variant, scenario=label, metric=mname, **{f: (round(100 * res[(f, mode)][mi].mean(), 1) if (f, mode) in res else np.nan) for f in GFAMS}))
                        for ref in ["metadata_only", "metadata_thresh", "date_free", "nogate_ft", "content_ft"]:
                            CI.append(f"{ern} {gname} {variant} {label:12s} gated_ft - {ref:15s} (cur) " + ci_str(res[("gated_ft", mode)][0], res[(ref, mode)][0]))
                    print(f"{ern} {gname} {variant} done ({time.time()-t0:.0f}s, {len(uniq)} prompts)")
        pd.DataFrame(rows).to_csv(f"exp4_e2e_variants_{TAG}.csv", index=False); open(f"e2e_variants_ci_{TAG}.txt", "w").write("\n".join(CI))
        pd.DataFrame(SAMPLES).to_csv(f"e2e_samples_for_manual_check_{TAG}.csv", index=False)
        print(pd.DataFrame(rows)[lambda d: d.metric.isin(["cur", "stale"])].to_string(index=False)); print("\n".join(CI))

# ---------------- judge diagnostics: calibration, shortcuts, length ----------------
if RUN_DIAG:
    with stage("judge_diagnostics"):
        import difflib
        ev_pairs = [(q["question"], q["cur"], t) for q in qas_eval for t, _, _ in q["outs"]]       # (question, NEW, OLD); articles never seen in training
        ev = random.Random(SEED).sample(ev_pairs, min(2000, len(ev_pairs))); print("held-out pairs:", len(ev))
        SS = {}
        for jn in ("zs", "ft"):
            m1 = np.array(jm([(q, n, o) for q, n, o in ev], jn)); m2 = np.array(jm([(q, o, n) for q, n, o in ev], jn)); SS[jn] = (m1 - m2) / 2     # >0: NEW judged newer
        diag = []
        diag.append(dict(analysis="overall accuracy", bin="all", n=len(ev), **{f"{j}_acc": round(100 * (SS[j] > 0).mean(), 1) for j in SS}))
        # --- calibration: confidence = sigmoid(|score|); labels are antisymmetric so no flipping is needed
        sig = lambda x: 1 / (1 + np.exp(-x))
        def ece_of(s, T=1.0, bins=10):
            conf = sig(np.abs(s) / T); ok = (s > 0).astype(float); edges = np.linspace(.5, 1, bins + 1); e = 0.0; rel = []
            for i in range(bins):
                m = (conf >= edges[i]) & (conf < edges[i + 1] + (1e-9 if i == bins - 1 else 0))
                if m.sum(): e += m.mean() * abs(ok[m].mean() - conf[m].mean()); rel.append((i, int(m.sum()), conf[m].mean(), ok[m].mean()))
            return e, rel
        cal_rows = []
        for jn in SS:
            s = SS[jn]; half = len(s) // 2; grid = np.exp(np.linspace(np.log(.05), np.log(30), 120))
            nll = lambda T, x: -np.mean(np.log(np.clip(np.where(x > 0, sig(np.abs(x) / T), 1 - sig(np.abs(x) / T)), 1e-9, 1)))
            T = grid[int(np.argmin([nll(t, s[:half]) for t in grid]))]
            e0, rel = ece_of(s[half:]); e1, _ = ece_of(s[half:], T)
            diag.append(dict(analysis="calibration", bin=f"ECE (10 bins) on held-out half; after temperature T={T:.2f}", n=len(s) - half, **{f"{j}_acc": (round(100 * e0, 2) if j == jn else np.nan) for j in SS}, **{f"{jn}_ece_after_T": round(100 * e1, 2)}))
            for i, n_, c_, a_ in rel: cal_rows.append(dict(judge=jn, bin=i, n=n_, mean_confidence=round(c_, 3), accuracy=round(a_, 3)))
        pd.DataFrame(cal_rows).to_csv(f"exp4_calibration_bins_{TAG}.csv", index=False)
        # --- shortcut analysis
        wc_ = lambda x: len(x.split()); dig = lambda s: sum(ch.isdigit() for ch in s); yr = lambda s: len(re.findall(r"\b(1[5-9]\d\d|20\d\d)\b", s))
        L_new, L_old = np.array([len(n) for _, n, o in ev]), np.array([len(o) for _, n, o in ev]); dc = L_new - L_old
        rel_len = np.abs(dc) / np.maximum(np.maximum(L_new, L_old), 1)
        for name, d in [("fewer words = newer", np.array([wc_(n) - wc_(o) for _, n, o in ev])), ("fewer characters = newer", dc), ("fewer digits = newer", np.array([dig(n) - dig(o) for _, n, o in ev]))]:
            diag.append(dict(analysis="trivial rule", bin=name, n=len(ev), zs_acc=round(100 * ((d < 0).mean() + .5 * (d == 0).mean()), 1), ft_acc=np.nan))
        def bins_report(name, vals, edges, labels=None):
            for i in range(len(edges) - 1):
                m = (vals >= edges[i]) & (vals < edges[i + 1])
                if m.sum(): diag.append(dict(analysis=name, bin=(labels[i] if labels else f"[{edges[i]:.2f},{edges[i+1]:.2f})"), n=int(m.sum()), **{f"{j}_acc": round(100 * (SS[j][m] > 0).mean(), 1) for j in SS}))
        bins_report("relative length difference", rel_len, [0, .05, .15, .3, .6, 2.0])
        ops = [difflib.SequenceMatcher(None, o.split(), n.split(), autojunk=False).get_opcodes() for _, n, o in ev]
        n_edits = np.array([sum(1 for t, *_ in op if t != "equal") for op in ops])
        first_pos = []
        for (_, n, o), op in zip(ev, ops):
            blk = [x for x in op if x[0] != "equal"]; first_pos.append(blk[0][1] / max(len(o.split()), 1) if blk else 1.0)
        first_pos = np.array(first_pos)
        chg_txt = []
        for (_, n, o), op in zip(ev, ops):
            nw, ow = n.split(), o.split(); chg_txt.append(" ".join(" ".join(nw[j1:j2] + ow[i1:i2]) for t, i1, i2, j1, j2 in op if t != "equal"))
        has_digit = np.array([any(ch.isdigit() for ch in c) for c in chg_txt]).astype(float); has_year = np.array([yr(c) > 0 for c in chg_txt]).astype(float)
        bins_report("number of word-level edit blocks", n_edits.astype(float), [0, 2, 3, 5, 9, 1e9], ["1", "2", "3-4", "5-8", "9+"])
        bins_report("position of first change (share of text)", first_pos, [0, .25, .5, .75, 1.01])
        bins_report("changed words contain a digit", has_digit, [0, .5, 1.01], ["no", "yes"])
        bins_report("changed words contain a year", has_year, [0, .5, 1.01], ["no", "yes"])
        # --- can version-level surface features alone predict which version is newer? (antisymmetric logistic regression, 5-fold CV)
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            feat = lambda s: [len(s), wc_(s), dig(s), yr(s), sum(1 for w in s.split() if w[:1].isupper())]
            X = np.array([np.array(feat(n)) - np.array(feat(o)) for _, n, o in ev], float); X = X / (X.std(0) + 1e-9)
            Xa = np.vstack([X, -X]); ya = np.r_[np.ones(len(X)), np.zeros(len(X))]; grp = np.r_[np.arange(len(X)), np.arange(len(X))]
            from sklearn.model_selection import GroupKFold
            acc = cross_val_score(LogisticRegression(fit_intercept=False, max_iter=1000), Xa, ya, groups=grp, cv=GroupKFold(5)).mean()
            diag.append(dict(analysis="surface-feature classifier (length, words, digits, years, capitals; 5-fold CV)", bin="all", n=len(ev), zs_acc=round(100 * acc, 1), ft_acc=np.nan))
        except Exception as e: print("surface classifier skipped:", repr(e)[:100])
        pd.DataFrame(diag).to_csv(f"exp4_judge_diagnostics_{TAG}.csv", index=False); print(pd.DataFrame(diag).to_string(index=False))

# ---------------- wrap up ----------------
pd.DataFrame([dict(stage=k, seconds=v) for k, v in TIMES.items()]).to_csv(f"exp4_runtime_{TAG}.csv", index=False)
files = sorted(f for f in glob.glob(f"*{TAG}*.csv") + glob.glob(f"*{TAG}*.txt") if not f.startswith("SUMMARY"))
with open(f"SUMMARY_{TAG}.txt", "w") as out:
    for f in files: out.write("\n" + "=" * 20 + " " + f + " " + "=" * 20 + "\n" + open(f).read())
print("DONE. Files:", files + [f"perq_{TAGP}.npz", f"SUMMARY_{TAG}.txt"])
print("Runtime per stage (s):", TIMES)
