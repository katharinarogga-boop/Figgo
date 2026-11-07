"""TippguruFiggo – läuft als GitHub Action. Holt OpenLigaDB-Daten, rechnet Tipps, speichert alles verschlüsselt in daten.json."""

import json, glob, math, os, sys, base64, urllib.request, time
from datetime import date
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAXG = 10  # Tore 0..9 je Team


def days(d):
    return date.fromisoformat(d).toordinal()


# ---------------------------------------------------------------- Dixon-Coles
def dc_tau(x, y, lh, la, rho):
    t = np.ones_like(lh)
    m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return t


class DixonColes:
    def __init__(self, xi=0.0019, l2=0.05):
        self.xi, self.l2 = xi, l2

    def fit(self, matches, ref_day):
        ms = [m for m in matches if m["hg"] is not None and days(m["date"]) < ref_day]
        teams = sorted({m["home"] for m in ms} | {m["away"] for m in ms})
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([self.idx[m["home"]] for m in ms]); ai = np.array([self.idx[m["away"]] for m in ms])
        hg = np.array([m["hg"] for m in ms], float); ag = np.array([m["ag"] for m in ms], float)
        lg = np.array([1 if m["league"] == "BL2" else 0 for m in ms])
        w = np.exp(-self.xi * (ref_day - np.array([days(m["date"]) for m in ms])))
        self.n = n
        # Parameter: att[n], dfn[n], mu, home_bl1, home_bl2
        def nll(p):
            att, dfn, mu, h1, h2 = p[:n], p[n:2*n], p[2*n], p[2*n+1], p[2*n+2]
            home = np.where(lg == 1, h2, h1)
            llh = mu + home + att[hi] - dfn[ai]
            lla = mu + att[ai] - dfn[hi]
            lh, la = np.exp(llh), np.exp(lla)
            ll = w * (hg * llh - lh + ag * lla - la)
            # Gradienten
            rh, ra = w * (hg - lh), w * (ag - la)
            g = np.zeros_like(p)
            np.add.at(g, hi, rh); np.add.at(g, ai, ra)            # att
            np.add.at(g, n + ai, -rh); np.add.at(g, n + hi, -ra)  # dfn
            g[2*n] = rh.sum() + ra.sum()
            g[2*n+1] = rh[lg == 0].sum(); g[2*n+2] = rh[lg == 1].sum()
            reg = self.l2 * (att @ att + dfn @ dfn)
            g[:2*n] -= 2 * self.l2 * p[:2*n]
            return -(ll.sum() - reg), -g
        p0 = np.zeros(2*n + 3); p0[2*n] = 0.3; p0[2*n+1] = p0[2*n+2] = 0.2
        res = minimize(nll, p0, jac=True, method="L-BFGS-B")
        p = res.x
        self.att, self.dfn, self.mu, self.h1, self.h2 = p[:n], p[n:2*n], p[2*n], p[2*n+1], p[2*n+2]
        # rho separat
        home = np.where(lg == 1, self.h2, self.h1)
        lh = np.exp(self.mu + home + self.att[hi] - self.dfn[ai]); la = np.exp(self.mu + self.att[ai] - self.dfn[hi])
        def f(r):
            t = dc_tau(hg, ag, lh, la, r)
            return -(w * np.log(np.clip(t, 1e-9, None))).sum()
        self.rho = minimize_scalar(f, bounds=(-0.2, 0.2), method="bounded").x
        return self

    def rates(self, home, away, league):
        h = self.h2 if league == "BL2" else self.h1
        # Unbekanntes Team (Neuling ohne Historie): leicht unterdurchschnittlich
        ah = self.att[self.idx[home]] if home in self.idx else -0.15
        dh = self.dfn[self.idx[home]] if home in self.idx else -0.15
        aa = self.att[self.idx[away]] if away in self.idx else -0.15
        da = self.dfn[self.idx[away]] if away in self.idx else -0.15
        return math.exp(self.mu + h + ah - da), math.exp(self.mu + aa - dh)

    def score_matrix(self, home, away, league):
        lh, la = self.rates(home, away, league)
        M = np.outer(poisson.pmf(range(MAXG), lh), poisson.pmf(range(MAXG), la))
        r = self.rho
        M[0, 0] *= 1 - lh * la * r; M[0, 1] *= 1 + lh * r; M[1, 0] *= 1 + la * r; M[1, 1] *= 1 - r
        return M / M.sum(), lh, la


def outcome_probs(M):
    return np.array([np.tril(M, -1).sum(), np.trace(M), np.triu(M, 1).sum()])  # H, D, A


# ---------------------------------------------------------------- Elo
class Elo:
    def __init__(self, k=20, hfa=60, carry=0.8, bl2_start=1400, bl1_start=1550):
        self.k, self.hfa, self.carry = k, hfa, carry
        self.start = {"BL1": bl1_start, "BL2": bl2_start}
        self.r, self.season = {}, None

    def get(self, t, league):
        return self.r.get(t, self.start[league])

    def expect(self, home, away, league):
        d = self.get(home, league) + self.hfa - self.get(away, league)
        return 1 / (1 + 10 ** (-d / 400)), d

    def update(self, m):
        if self.season and m["season"] != self.season:  # Rückkehr zum Mittelwert zwischen Saisons
            mean = np.mean(list(self.r.values()))
            self.r = {t: mean + self.carry * (v - mean) for t, v in self.r.items()}
        self.season = m["season"]
        e, _ = self.expect(m["home"], m["away"], m["league"])
        s = 1.0 if m["hg"] > m["ag"] else 0.5 if m["hg"] == m["ag"] else 0.0
        gd = abs(m["hg"] - m["ag"]); mult = 1 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = self.k * mult * (s - e)
        self.r[m["home"]] = self.get(m["home"], m["league"]) + delta
        self.r[m["away"]] = self.get(m["away"], m["league"]) - delta


# ---------------------------------------------------------------- Tipp
def points(tip, res, rule=(4, 3, 2)):
    exact, diff, tend = rule
    th, ta = tip; rh, ra = res
    if (th, ta) == (rh, ra): return exact
    if th - ta == rh - ra: return diff   # inkl. Remis mit falschem Ergebnis
    if np.sign(th - ta) == np.sign(rh - ra): return tend
    return 0


def best_tip(M, rule=(4, 3, 2), maxg=6):
    best, bestv = None, -1
    for th in range(maxg):
        for ta in range(maxg):
            v = sum(M[i, j] * points((th, ta), (i, j), rule) for i in range(MAXG) for j in range(MAXG))
            if v > bestv: best, bestv = (th, ta), v
    return best, bestv


def rps(p, o):
    c = np.cumsum(p)[:2]; oc = np.cumsum(o)[:2]
    return 0.5 * ((c - oc) ** 2).sum()


_PT = {}
def best_tip_fast(M, rule=(4, 3, 2), maxg=6):
    key = (rule, maxg)
    if key not in _PT:
        tips = [(a, b) for a in range(maxg) for b in range(maxg)]
        P = np.array([[[points(t, (i, j), rule) for j in range(MAXG)] for i in range(MAXG)] for t in tips], float)
        _PT[key] = (tips, P)
    tips, P = _PT[key]
    ev = (P * M[None]).sum(axis=(1, 2))
    k = int(ev.argmax())
    return tips[k], float(ev[k])


def run(matches, seasons, xi, l2=0.05, verbose=False):
    test = [m for m in matches if m["season"] in seasons and m["hg"] is not None]
    # Refit wöchentlich: Gruppen nach ISO-Woche
    weeks = {}
    for m in test:
        y, w, _ = date.fromisoformat(m["date"]).isocalendar(); weeks.setdefault((y, w), []).append(m)
    out = []
    elo = Elo(); hist = [m for m in matches if m["hg"] is not None]
    ei = 0
    for wk in sorted(weeks):
        ms = weeks[wk]; ref = min(days(m["date"]) for m in ms)
        while ei < len(hist) and days(hist[ei]["date"]) < ref:
            elo.update(hist[ei]); ei += 1
        dc = DixonColes(xi=xi, l2=l2).fit(matches, ref)
        for m in ms:
            M, lh, la = dc.score_matrix(m["home"], m["away"], m["league"])
            p = outcome_probs(M)
            tip, ev = best_tip_fast(M)
            o = np.array([m["hg"] > m["ag"], m["hg"] == m["ag"], m["hg"] < m["ag"]], float)
            eh, ed = elo.expect(m["home"], m["away"], m["league"])
            out.append(dict(season=m["season"], league=m["league"], round=m["round"], date=m["date"],
                            home=m["home"], away=m["away"], hg=m["hg"], ag=m["ag"],
                            p=p.tolist(), lh=lh, la=la, tip=tip, ev=ev, elo_e=eh, elo_d=ed,
                            rps=rps(p, o), brier=float(((p - o) ** 2).sum()),
                            ll=float(-np.log(max(p[o.argmax()], 1e-12))),
                            pts=points(tip, (m["hg"], m["ag"]))))
    return out


def evaluate(r, ms, seasons):
    res = {}
    base = {}
    for lg in ["BL1","BL2"]:
        tr = [m for m in ms if m["league"]==lg and m["season"]<seasons[0] and m["hg"] is not None]
        base[lg] = np.mean([[m["hg"]>m["ag"], m["hg"]==m["ag"], m["hg"]<m["ag"]] for m in tr], axis=0)
    groups = {f"{lg} {s}": [x for x in r if x["league"]==lg and x["season"]==s] for lg in ["BL1","BL2"] for s in seasons}
    groups["Gesamt"] = r
    for name, sel in groups.items():
        o = np.array([[x["hg"]>x["ag"], x["hg"]==x["ag"], x["hg"]<x["ag"]] for x in sel], float)
        P = np.array([x["p"] for x in sel]); B = np.array([base[x["league"]] for x in sel])
        rp = lambda P: float(np.mean([rps(p,oo) for p,oo in zip(P,o)]))
        res[name] = dict(n=len(sel), rps=rp(P), rps_base=rp(B),
            brier=float(np.mean(((P-o)**2).sum(1))), brier_base=float(np.mean(((B-o)**2).sum(1))),
            fav=float(np.mean(P.argmax(1)==o.argmax(1))), home=float(o[:,0].mean()),
            tend=float(np.mean([np.sign(x["tip"][0]-x["tip"][1])==np.sign(x["hg"]-x["ag"]) for x in sel])),
            diff=float(np.mean([x["tip"][0]-x["tip"][1]==x["hg"]-x["ag"] for x in sel])),
            exact=float(np.mean([tuple(x["tip"])==(x["hg"],x["ag"]) for x in sel])),
            pts=float(np.mean([x["pts"] for x in sel])),
            pts21=float(np.mean([points((2,1),(x["hg"],x["ag"])) for x in sel])))
    # Kalibrierung
    P = np.array([x["p"] for x in r]); O = np.array([[x["hg"]>x["ag"], x["hg"]==x["ag"], x["hg"]<x["ag"]] for x in r], float)
    cal = {}
    for k,nm in enumerate(["H","D","A"]):
        pts_=[]
        for a in np.arange(0,1,0.1):
            m=(P[:,k]>=a)&(P[:,k]<a+0.1)
            if m.sum()>=15: pts_.append([round(float(P[m,k].mean()),3), round(float(O[m,k].mean()),3), int(m.sum())])
        cal[nm]=pts_
    return res, cal

def team_form(hist, team, ref_day, n=5):
    gs = [m for m in hist if (m["home"] == team or m["away"] == team) and days(m["date"]) < ref_day][-n:]
    pts = gf = ga = 0
    seq = ""
    for m in gs:
        h = m["home"] == team
        f, a = (m["hg"], m["ag"]) if h else (m["ag"], m["hg"])
        gf += f; ga += a
        r = "S" if f > a else "U" if f == a else "N"
        seq += r; pts += {"S": 3, "U": 1, "N": 0}[r]
    last = gs[-1]["date"] if gs else None
    return dict(n=len(gs), pts=pts, gf=gf, ga=ga, seq=seq, last=last)


def de(x, n):
    return f"{x:.{n}f}".replace(".", ",")


def pct(x):
    return f"{round(100 * x)} %"


def reasoning(m, p, lh, la, tip, ev, top, eh, ea, edelta, fh, fa, rest_h, rest_a):
    H, A = m["home"], m["away"]
    lines = []
    fav = int(np.argmax(p))
    if max(p) < 0.42:
        lines.append(f"Offenes Spiel: Heimsieg {pct(p[0])}, Remis {pct(p[1])}, Auswärtssieg {pct(p[2])}.")
    else:
        who = {0: f"{H} (Heimsieg)", 1: "ein Remis", 2: f"{A} (Auswärtssieg)"}[fav]
        lines.append(f"Das Modell sieht {who} mit {pct(p[fav])} vorn; Remis {pct(p[1])}.")
    lines.append(f"Elo {round(eh)} gegen {round(ea)}, inklusive Heimvorteil {abs(edelta):.0f} Punkte Vorsprung für {H if edelta >= 0 else A}; "
                 f"erwartete Tore {de(lh, 1)} : {de(la, 1)}.")
    if fh["n"] and fa["n"]:
        lines.append(f"Letzte {fh['n']} Ligaspiele: {H} {fh['seq']} ({fh['pts']} Pkt, {fh['gf']}:{fh['ga']} Tore), "
                     f"{A} {fa['seq']} ({fa['pts']} Pkt, {fa['gf']}:{fa['ga']}).")
    if rest_h is not None and rest_a is not None and abs(rest_h - rest_a) >= 3:
        lines.append(f"Ruhetage seit dem letzten Ligaspiel: {H} {rest_h}, {A} {rest_a} (Pokal/Europapokal nicht erfasst; nicht eingerechnet).")
    t0 = top[0]
    why = (f"Tipp {tip[0]}:{tip[1]}: bringt nach der Regel 4/3/2 im Schnitt {de(ev, 2)} Punkte"
           + ("." if (t0[0], t0[1]) == tuple(tip) else f" – mehr als das einzeln wahrscheinlichste Ergebnis {t0[0]}:{t0[1]} ({pct(t0[2])})."))
    lines.append(why)
    return " ".join(lines)



# ================================================================ GitHub-Lauf
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
BER = ZoneInfo("Europe/Berlin")
RULE = (4, 3, 2)
XI = 0.0019
SEASON_YEAR = int(os.environ.get("TIPPGURU_SAISON", "2026"))
SEASON = f"{SEASON_YEAR}-{str(SEASON_YEAR + 1)[2:]}"
FIRST_YEAR = SEASON_YEAR - 7
LNAME = {"BL1": "1. Bundesliga", "BL2": "2. Bundesliga"}
HERE = os.path.dirname(os.path.abspath(__file__))
ITER = 250000


def fetch_json(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "tippguru-github-action", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def load_matches():
    """Alle Spiele beider Ligen von OpenLigaDB. Teams über teamId, Name = jüngster Name."""
    offline = os.environ.get("TIPPGURU_OFFLINE")
    raw = []
    for lg in ["bl1", "bl2"]:
        for y in range(FIRST_YEAR, SEASON_YEAR + 1):
            data = json.load(open(os.path.join(offline, f"{lg}_{y}.json"))) if offline else \
                fetch_json(f"https://api.openligadb.de/getmatchdata/{lg}/{y}")
            for m in data:
                raw.append((lg.upper(), y, m))
    latest = {}
    for lg, y, m in sorted(raw, key=lambda t: t[2].get("matchDateTimeUTC") or ""):
        for t in ("team1", "team2"):
            latest[m[t]["teamId"]] = m[t]["teamName"].strip()
    rows = []
    for lg, y, m in raw:
        utc = (m.get("matchDateTimeUTC") or "").replace("Z", "+00:00")
        if not utc:
            continue
        ko = datetime.fromisoformat(utc).astimezone(BER)
        res = [r for r in (m.get("matchResults") or []) if r.get("resultTypeID") == 2]
        fin = m.get("matchIsFinished") and res
        rows.append(dict(season=f"{y}-{str(y + 1)[2:]}", league=lg, round=int(m["group"]["groupOrderID"]),
                         date=ko.date().isoformat(), time=ko.strftime("%H:%M"),
                         home=latest[m["team1"]["teamId"]], away=latest[m["team2"]["teamId"]],
                         hg=int(res[0]["pointsTeam1"]) if fin else None, ag=int(res[0]["pointsTeam2"]) if fin else None))
    rows.sort(key=lambda r: (r["date"], r["time"], r["league"]))
    return rows


def kdf(pw, salt, it):
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=it).derive(pw.encode())


def decrypt_state(path, pw):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not os.path.exists(path):
        return {"docs": {}}
    e = json.load(open(path))
    b = base64.b64decode
    return json.loads(AESGCM(kdf(pw, b(e["salt"]), e["iter"])).decrypt(b(e["iv"]), b(e["ct"]), None))


def encrypt_state(path, pw, state):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, iv = os.urandom(16), os.urandom(12)
    ct = AESGCM(kdf(pw, salt, ITER)).encrypt(iv, json.dumps(state, ensure_ascii=False).encode(), None)
    b = lambda x: base64.b64encode(x).decode()
    json.dump(dict(v=1, iter=ITER, salt=b(salt), iv=b(iv), ct=b(ct)), open(path, "w"))


def kickoff(m):
    return datetime.fromisoformat(f"{m['date']}T{m['time'] or '15:30'}").replace(tzinfo=BER)


def main(now=None):
    pw = os.environ.get("TIPPGURU_PASSWORT", "")
    if not pw:
        sys.exit("TIPPGURU_PASSWORT fehlt (Repository-Secret anlegen).")
    now = now or datetime.now(timezone.utc)
    datapath = os.path.join(HERE, "daten.json")
    state = decrypt_state(datapath, pw)
    docs = state.get("docs", {})
    matches = load_matches()
    hist = [m for m in matches if m["hg"] is not None]
    print("Spiele geladen:", len(matches), "davon gespielt:", len(hist))

    elo = Elo()
    for m in hist:
        elo.update(m)
    ref = days(now.astimezone(BER).date().isoformat()) + 1
    dc = DixonColes(xi=XI).fit(matches, ref)

    cur = [m for m in matches if m["season"] == SEASON]
    for lg in ["BL1", "BL2"]:
        lm = [m for m in cur if m["league"] == lg]
        rounds = sorted({m["round"] for m in lm})
        open_rounds = [r for r in rounds if any(m["hg"] is None for m in lm if m["round"] == r)]
        target = [r for r in open_rounds
                  if min(kickoff(m) for m in lm if m["round"] == r and m["hg"] is None) <= now + timedelta(days=6)]
        for r in sorted(set(target) | {int(k.split("-")[-1]) for k in docs if k.startswith(f"{lg}-{SEASON}-")}):
            key = f"{lg}-{SEASON}-{r:02d}"
            oldm = {(x["home"], x["away"]): x for x in docs.get(key, {}).get("matches", [])}
            out = []
            for m in sorted([x for x in lm if x["round"] == r], key=lambda x: (x["date"], x["time"])):
                ko = kickoff(m)
                rec = oldm.get((m["home"], m["away"]))
                if rec is None and r in target and m["hg"] is None:
                    M, lh, la = dc.score_matrix(m["home"], m["away"], lg)
                    p = outcome_probs(M)
                    tip, ev = best_tip_fast(M, RULE)
                    flat = sorted(((i, j, float(M[i, j])) for i in range(MAXG) for j in range(MAXG)), key=lambda t: -t[2])[:5]
                    eh, ea = elo.get(m["home"], lg), elo.get(m["away"], lg)
                    _, edelta = elo.expect(m["home"], m["away"], lg)
                    rd = days(m["date"])
                    fh, fa = team_form(hist, m["home"], rd), team_form(hist, m["away"], rd)
                    rest = lambda f: (rd - days(f["last"])) if f["last"] else None
                    rec = dict(home=m["home"], away=m["away"], kickoff=ko.isoformat(),
                               p=[round(float(x), 4) for x in p], xg=[round(lh, 2), round(la, 2)],
                               top=[[i, j, round(q, 4)] for i, j, q in flat], tip=list(tip), ev=round(ev, 3),
                               elo=[round(eh), round(ea)], form=[fh["seq"], fa["seq"]],
                               reason=reasoning(m, p, lh, la, tip, ev, flat, eh, ea, edelta, fh, fa, rest(fh), rest(fa)),
                               created=now.isoformat(timespec="seconds"), before_kickoff=now < ko)
                if rec is None:
                    continue
                rec = dict(rec); rec["kickoff"] = ko.isoformat()
                if m["hg"] is not None:
                    rec["result"] = [m["hg"], m["ag"]]
                    rec["pts"] = points(tuple(rec["tip"]), (m["hg"], m["ag"]), RULE)
                out.append(rec)
            if out:
                docs[key] = dict(league=lg, league_name=LNAME[lg], season=SEASON, round=r, matches=out,
                                 updated=now.isoformat(timespec="seconds"))

    tally = {}
    for d in docs.values():
        for x in d["matches"]:
            if "result" not in x or not x.get("before_kickoff"):
                continue
            t = tally.setdefault(d["league"], dict(n=0, pts=0, exact=0, diff=0, tend=0, rps=0.0))
            h, a = x["result"]; th, ta = x["tip"]
            t["n"] += 1; t["pts"] += x["pts"]
            t["exact"] += int((th, ta) == (h, a)); t["diff"] += int(th - ta == h - a)
            t["tend"] += int(np.sign(th - ta) == np.sign(h - a))
            o = np.array([h > a, h == a, h < a], float); c = np.cumsum(x["p"])[:2]; oc = np.cumsum(o)[:2]
            t["rps"] += 0.5 * float(((c - oc) ** 2).sum())
    for t in tally.values():
        t["rps"] = round(t["rps"] / t["n"], 4) if t["n"] else None

    # Rückwärtstest: die zwei Saisons vor der Vorsaison; Rückblick laufende Saison
    bt_seasons = {f"{y}-{str(y + 1)[2:]}" for y in (SEASON_YEAR - 3, SEASON_YEAR - 2)}
    ev_res, cal = evaluate(run(matches, bt_seasons, XI), matches, sorted(bt_seasons))
    retro = {}
    for x in run(matches, {SEASON}, XI):
        t = retro.setdefault(x["league"], dict(n=0, pts=0, tend=0, exact=0, rps=0.0))
        t["n"] += 1; t["pts"] += x["pts"]; t["rps"] += x["rps"]
        t["tend"] += int(np.sign(x["tip"][0] - x["tip"][1]) == np.sign(x["hg"] - x["ag"]))
        t["exact"] += int(tuple(x["tip"]) == (x["hg"], x["ag"]))
    for t in retro.values():
        t["rps"] = round(t["rps"] / t["n"], 4)

    last_res = {lg: max((m["date"] for m in cur if m["league"] == lg and m["hg"] is not None), default=None) for lg in ["BL1", "BL2"]}
    status = dict(updated=now.isoformat(timespec="seconds"), rule="4/3/2 (exakt/Tordifferenz/Tendenz)",
                  data=dict(last_result=last_res, sources=["OpenLigaDB (api.openligadb.de)"]),
                  model=dict(name="Dixon-Coles-Poisson + Elo", xi=XI),
                  backtest=dict(groups=ev_res, calibration=cal), live=tally, retro=retro,
                  gaps=["Keine Buchmacherquoten (football-data.co.uk erlaubt keinen automatischen Abruf)",
                        "Kein xG (keine freie Quelle, für die 2. Liga gar keine) – Form über Tore/Ergebnisse",
                        "Laufleistung/Sprints/Belastung öffentlich nicht pro Spiel verfügbar – Ersatz: Ruhetage (nur Ligaspiele)",
                        "Ausfälle/Sperren werden nicht ins Modell eingerechnet"])
    encrypt_state(datapath, pw, dict(docs=docs, status=status))
    print("Spieltage mit Tipps:", sorted(docs)); print("Live-Bilanz:", tally)


if __name__ == "__main__":
    main()
