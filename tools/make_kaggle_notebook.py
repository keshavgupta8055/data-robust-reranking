# Rebuild the Kaggle notebook from src/run_experiments.py and src/merge_seeds.py.  Run from this folder: python make_kaggle_notebook.py
import base64, json
v4 = open("../src/run_experiments.py", encoding="ascii").read(); ms = open("../src/merge_seeds.py", encoding="ascii").read()
def blk(name, s):
    enc = base64.b64encode(s.encode("ascii")).decode()
    return name + " = (\n" + "\n".join('    "%s"' % enc[i:i+100] for i in range(0, len(enc), 100)) + "\n)\n"
cell = '''# ===== RUN v4: improved experiments for the paper. ONE cell. =====
# Needs: Settings > Accelerator = GPU T4 x2 (or P100), Settings > Internet = On.
# Best used with: Save Version > Save & Run All (Commit), so it keeps running if the browser closes. (Max session about 9-12 h.)
# Edit ONLY the WHICH line. With 24 GPU hours, use THREE separate Save & Run All sessions (a Kaggle session lasts at most about 12 h):
#   session 1: WHICH = ["A", "B"]     (main run + second judge seed, about 6-8 h)
#   session 2: WHICH = ["C", "D"]     (third judge seed + 3B judge, about 3-4 h; the re-ranker score cache is rebuilt, about 1 h)
#   ADAPTIVE ON MORE SEEDS (the paper's main open caveat): WHICH = ["H", "I"]  (adaptive estimator only, seeds 1 and 2; about 2 h for H incl. cache rebuild, about 45 min more for I)
#   session 3: WHICH = ["G"]          (4B re-ranker, about 2-3 h)  -- optional
# After each session download results_all.zip. For pooled intervals over all seeds, send me the three zips (or run merge_seeds.py on the perq_*.npz files).
WHICH = ["A", "B"]
# Rough time on a T4 (first run also fills the re-ranker score cache, later runs reuse it): A about 4-6 h, B and C about 1-1.5 h each, D about 2 h, G about 2-3 h.
# If the session stops, just run the cell again: finished runs are skipped and cached scores are reused.
import base64, os, subprocess, sys, time, glob, zipfile
WD = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
os.chdir(WD)
''' + blk("V4", v4) + blk("MS", ms) + '''open("v4.py", "w").write(base64.b64decode(V4).decode("ascii"))
open("merge_seeds.py", "w").write(base64.b64decode(MS).decode("ascii"))
M15, M3 = "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
ALL_RUNS = {
    "A": dict(JSEED=0, JUDGE_MODEL=M15, GEN_MODEL=M3, RUN_EXTRAS=1, RUN_DIAG=1, RUN_E2E_VARIANTS=1),     # main run: all new analyses + end-to-end with two generators
    "B": dict(JSEED=1, JUDGE_MODEL=M15, RUN_EXTRAS=0, RUN_DIAG=0, RUN_E2E_VARIANTS=0),                   # second judge seed
    "C": dict(JSEED=2, JUDGE_MODEL=M15, RUN_EXTRAS=0, RUN_DIAG=0, RUN_E2E_VARIANTS=0),                   # third judge seed
    "D": dict(JSEED=0, JUDGE_MODEL=M3, RUN_EXTRAS=0, RUN_DIAG=1, RUN_E2E_VARIANTS=0),                    # larger judge
    "H": dict(JSEED=1, JUDGE_MODEL=M15, RUN_EXTRAS=1, RUN_ADAPT_ONLY=1, RUN_DIAG=0, RUN_E2E_VARIANTS=0),       # adaptive date-trust estimator only, judge seed 1
    "I": dict(JSEED=2, JUDGE_MODEL=M15, RUN_EXTRAS=1, RUN_ADAPT_ONLY=1, RUN_DIAG=0, RUN_E2E_VARIANTS=0),       # adaptive date-trust estimator only, judge seed 2
    "G": dict(JSEED=0, JUDGE_MODEL=M15, RERANKERS_SEL="qwen3-4B", N_QUERIES=1200, PASS_SET="core", RUN_EXTRAS=0, RUN_DIAG=0, RUN_E2E_VARIANTS=0),   # larger re-ranker
}
status = {}
for name in WHICH:
    cfg = ALL_RUNS[name]; done = f"run_{name}.done"
    if os.path.exists(done):
        print(f"##### run {name}: already finished, skipping"); status[name] = "done (earlier)"; continue
    print(f"\\n##### run {name} starting: {cfg}"); t0 = time.time()
    env = dict(os.environ, **{k: str(v) for k, v in cfg.items()})
    with open(f"log_run_{name}.txt", "w") as lg:
        p = subprocess.Popen([sys.executable, "-u", "v4.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=WD)
        for line in p.stdout:
            lg.write(line)
            if not line.startswith("update ") or "/375" in line: print(line, end="")
        p.wait()
    ok = p.returncode == 0
    status[name] = ("OK" if ok else f"FAILED (code {p.returncode}) - see log_run_{name}.txt") + f" in {(time.time()-t0)/60:.0f} min"
    if ok: open(done, "w").write("ok")
    print(f"##### run {name}: {status[name]}")
print("\\n" + "=" * 20, "STATUS", "=" * 20); [print(k, v) for k, v in status.items()]
seed_files = sorted(glob.glob("perq_j1.5B_s[0-9].npz"))
if len(seed_files) >= 2:
    print("\\nPooled intervals over queries and judge seeds:", seed_files)
    subprocess.run([sys.executable, "merge_seeds.py"] + seed_files)
with zipfile.ZipFile("results_all.zip", "w") as z:
    for f in glob.glob("exp4_*.csv") + glob.glob("e2e_*.csv") + glob.glob("e2e_*.txt") + glob.glob("SUMMARY_*.txt") + glob.glob("log_run_*.txt") + glob.glob("perq_*.npz") + glob.glob("pooled_seed_intervals.csv") + glob.glob("pools_*.json.gz"): z.write(f)
print("\\nALL DONE. Download results_all.zip (Output tab) and also upload the SUMMARY_*.txt files to the chat.")
'''
compile(cell, "wrapper", "exec")
open("../notebooks/kaggle_run_cell.txt", "w", encoding="ascii").write(cell)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["Improved experiments (v4). One code cell. Needs GPU and Internet On. Use Save Version > Save & Run All (Commit). Edit only the WHICH line."]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": cell.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("../notebooks/kaggle_run.ipynb", "w"))
print(len(cell))
