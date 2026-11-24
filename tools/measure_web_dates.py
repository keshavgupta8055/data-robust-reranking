# ===== How wrong are web document dates, in practice? (no GPU needed; Kaggle Internet must be On) =====
# There is no ground truth for the "true" date of a random web page, so this script measures (a) how often each date source
# exists, (b) how often sources CONTRADICT each other in ways that are impossible if both were right, and (c) the size of the
# disagreement in days. Impossible orderings are objective lower bounds on date errors.
# Sources per page: HTTP Last-Modified, HTTP Date (= crawl time), in-page published and modified dates (JSON-LD, <meta>, <time>),
# and the Hacker News submission time (the page must already exist when it is submitted), optional first Wayback capture
# (the page cannot be older than... it cannot have been published AFTER its first archive capture).
# Run:  python measure_web_dates.py          (about 10-20 minutes for 400 pages)   env: N_PAGES=400  WAYBACK=1
import os, re, json, time, random, statistics, concurrent.futures as cf
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
import urllib.request, urllib.error, urllib.parse
import pandas as pd, numpy as np
N_PAGES = int(os.environ.get("N_PAGES", 400)); WAYBACK = os.environ.get("WAYBACK", "1") == "1"; SEED = 0
UA = "Mozilla/5.0 (research date-metadata survey; contact: set-your-email@example.org)"      # put a real contact address here
random.seed(SEED)

def get(url, timeout=15, maxbytes=400_000, head_only=False):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD" if head_only else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = b"" if head_only else r.read(maxbytes)
        return dict(r.headers.items()), body, r.geturl()

def pdt(s):
    if not s: return None
    s = s.strip()
    try:
        d = parsedate_to_datetime(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except Exception: pass
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?", s)
    if m:
        try:
            y, mo, d, h, mi, se = (int(x) if x else 0 for x in m.groups())
            return datetime(y, mo, d, h, mi, se, tzinfo=timezone.utc)
        except Exception: return None
    return None

META_PUB = [r'property=["\']article:published_time["\'][^>]*content=["\']([^"\']+)', r'content=["\']([^"\']+)["\'][^>]*property=["\']article:published_time["\']',
            r'name=["\'](?:pubdate|publishdate|date|dc\.date(?:\.issued)?|datepublished)["\'][^>]*content=["\']([^"\']+)', r'itemprop=["\']datePublished["\'][^>]*(?:content|datetime)=["\']([^"\']+)']
META_MOD = [r'property=["\']article:modified_time["\'][^>]*content=["\']([^"\']+)', r'content=["\']([^"\']+)["\'][^>]*property=["\']article:modified_time["\']',
            r'name=["\'](?:last-modified|dc\.date\.modified|datemodified)["\'][^>]*content=["\']([^"\']+)', r'itemprop=["\']dateModified["\'][^>]*(?:content|datetime)=["\']([^"\']+)']
def inpage_dates(html):
    pub = mod = None
    for m in re.finditer(r'<script[^>]+application/ld\+json[^>]*>(.*?)</script>', html, re.S | re.I):
        try: data = json.loads(m.group(1))
        except Exception: continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, dict):
                pub = pub or pdt(x.get("datePublished")); mod = mod or pdt(x.get("dateModified")); stack += list(x.values())
            elif isinstance(x, list): stack += x
    for pat in META_PUB:
        if pub: break
        m = re.search(pat, html, re.I); pub = pdt(m.group(1)) if m else None
    for pat in META_MOD:
        if mod: break
        m = re.search(pat, html, re.I); mod = pdt(m.group(1)) if m else None
    return pub, mod

def hn_urls(n):
    ids = set()
    for lst in ("topstories", "newstories", "beststories"):
        try: ids |= set(json.loads(get(f"https://hacker-news.firebaseio.com/v0/{lst}.json")[1]))
        except Exception as e: print("HN list failed", lst, repr(e)[:80])
    ids = sorted(ids); random.shuffle(ids); out = []
    def item(i):
        try: return json.loads(get(f"https://hacker-news.firebaseio.com/v0/item/{i}.json", timeout=10)[1])
        except Exception: return None
    with cf.ThreadPoolExecutor(16) as ex:
        for it in ex.map(item, ids[: n * 3]):
            if it and it.get("url") and it.get("type") == "story" and "time" in it: out.append((it["url"], datetime.fromtimestamp(it["time"], tz=timezone.utc)))
            if len(out) >= n: break
    return out

def wayback_first(url):
    try:
        raw = get("https://web.archive.org/cdx/search/cdx?" + urllib.parse.urlencode(dict(url=url, output="json", limit=1, fl="timestamp", filter="statuscode:200")), timeout=20)[1]
        rows = json.loads(raw)
        return pdt(f"{rows[1][0][:4]}-{rows[1][0][4:6]}-{rows[1][0][6:8]}") if len(rows) > 1 else None
    except Exception: return None
import urllib.parse

def probe(item):
    url, hn_time = item; rec = dict(url=url, host=urllib.parse.urlparse(url).netloc, hn_time=hn_time, ok=False)
    try:
        h, body, final = get(url); html = body.decode("utf-8", "ignore"); h = {k.lower(): v for k, v in h.items()}
        rec.update(ok=True, last_modified=pdt(h.get("last-modified")), http_date=pdt(h.get("date")), has_etag="etag" in h)
        rec["pub"], rec["mod"] = inpage_dates(html)
        if WAYBACK: rec["wayback_first"] = wayback_first(url)
    except Exception as e: rec["err"] = repr(e)[:80]
    return rec

if __name__ == "__main__":
    t0 = time.time(); items = hn_urls(N_PAGES); print("urls:", len(items), f"({time.time()-t0:.0f}s)")
    with cf.ThreadPoolExecutor(12) as ex: recs = list(ex.map(probe, items))
    df = pd.DataFrame(recs); df = df[df.ok == True].copy(); print("pages fetched:", len(df))
    D = lambda a, b: (df[a] - df[b]).dt.total_seconds() / 86400            # days
    rows = []
    def add(name, value, note=""): rows.append(dict(measure=name, value=value, note=note)); print(f"{name:70s} {value}  {note}")
    n = len(df)
    for c, nm in [("last_modified", "HTTP Last-Modified"), ("http_date", "HTTP Date (crawl time)"), ("pub", "in-page published"), ("mod", "in-page modified"), ("wayback_first", "first Wayback capture")]:
        if c in df: add(f"availability: {nm}", f"{100*df[c].notna().mean():.1f}%", f"n={n}")
    lm, pub, mod, crawl = df.last_modified, df.pub, df["mod"], df.http_date
    both = lm.notna() & crawl.notna(); dyn = ((crawl - lm).abs() < pd.Timedelta(days=1)) & both
    add("Last-Modified within 1 day of crawl time (dynamic/regenerated stamp)", f"{100*dyn.sum()/max(both.sum(),1):.1f}%", f"of {both.sum()} pages with both")
    bp = lm.notna() & pub.notna(); d = D("last_modified", "pub")[bp]
    add("Last-Modified BEFORE in-page published date (impossible if both right)", f"{100*(d < -1).mean():.1f}%", f"of {bp.sum()}")
    add("Last-Modified AFTER published date by more than 30 days", f"{100*(d > 30).mean():.1f}%", "re-save / dynamic regeneration")
    add("Last-Modified minus published, median / sd / MAD (days)", f"{d.median():.1f} / {d.std():.1f} / {(d - d.median()).abs().median():.1f}", "empirical error scale to compare with the 30 and 180 day noise levels")
    for q in (50, 75, 90, 95): add(f"|Last-Modified - published| {q}th percentile (days)", f"{d.abs().quantile(q/100):.1f}")
    pm = mod.notna() & pub.notna(); add("in-page modified BEFORE published (impossible)", f"{100*((D('mod','pub')[pm]) < -1).mean():.1f}%", f"of {pm.sum()}")
    hp = df.pub.notna(); hv = (df.hn_time - df.pub).dt.total_seconds() / 86400
    add("published date AFTER the Hacker News submission time (impossible)", f"{100*(hv[hp] < -1).mean():.1f}%", f"of {hp.sum()}")
    if "wayback_first" in df:
        wp = df.wayback_first.notna() & pub.notna(); add("published AFTER first Wayback capture (impossible)", f"{100*((D('pub','wayback_first')[wp]) > 1).mean():.1f}%", f"of {wp.sum()}")
        wl = df.wayback_first.notna() & lm.notna(); add("Last-Modified AFTER crawl time (future date)", f"{100*((D('last_modified','http_date')[wl]) > 1).mean():.1f}%", f"of {wl.sum()}")
    any_bad = ((bp & (D('last_modified', 'pub') < -1)) | (pm & (D('mod', 'pub') < -1)) | (hp & (hv < -1)))
    add("pages with at least one impossible date ordering", f"{100*any_bad.mean():.1f}%", f"of {n} (lower bound on corrupted pages)")
    pd.DataFrame(rows).to_csv("web_date_summary.csv", index=False); df.to_csv("web_date_pages.csv", index=False)
    print("\nsaved web_date_summary.csv and web_date_pages.csv; paste web_date_summary.csv into the chat")
