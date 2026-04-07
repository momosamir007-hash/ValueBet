٧#!/usr/bin/env python3
""" ╔══════════════════════════════════════════════════════════════════════╗
    ║ ⚽ PREMIER LEAGUE PREDICTOR PRO v3.3 ⚽ ║
    ║ ║
    ║ FIXED: All critical ML & data pipeline bugs ║
    ║ ✅ FIX 1: norm_name - safe TEAMS_MAP lookup (no cross-contamination) ║
    ║ ✅ FIX 2: ML Pipeline (no data leakage, proper scaling) ║
    ║ ✅ FIX 3: Class imbalance handling (balanced weights) ║
    ║ ✅ FIX 4: NaN imputation (mean, not zero) ║
    ║ ✅ FIX 5: CV stability (StratifiedKFold + min samples guard) ║
    ║ ✅ FIX 6: External model loads Scaler too (Pipeline) ║
    ║ ✅ FIX 7: Promoted teams handled gracefully ║
    ║ ✅ FIX 8: Streamlit state management improved ║
    ╚══════════════════════════════════════════════════════════════════════╝ """
import requests
import json
import math
import os
import sys
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# ══════════════════════════════════════════════════════════════
# ENVIRONMENT DETECTION
# ══════════════════════════════════════════════════════════════
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
try:
    import numpy as np
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        StackingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    ML_AVAILABLE = True
except ImportError:
    pass

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
PL = "PL"
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
WEIGHTS = {
    'dixon_coles': 0.25,
    'elo': 0.18,
    'form': 0.12,
    'h2h': 0.08,
    'home_advantage': 0.08,
    'fatigue': 0.04,
    'draw_model': 0.10,
    'ml': 0.15,
}
ELO_INIT = 1500
ELO_K = 32
ELO_HOME = 65
FORM_N = 8
BACKTEST_SPLIT = 0.70
CALIBRATION_FILE = "calibration_v33.pkl"
XGB_MODEL_FILE = "football_xgboost_pipeline.pkl"
TEAMS_MAP_FILE = "teams_master_map.json"
ELO_RATINGS_FILE = "teams_elo_ratings.pkl"
MIN_SAMPLES_PER_CLASS = 10

RIVALRIES = {
    frozenset({'Arsenal', 'Tottenham'}): 'North London Derby',
    frozenset({'Liverpool', 'Everton'}): 'Merseyside Derby',
    frozenset({'Man United', 'Man City'}): 'Manchester Derby',
    frozenset({'Man United', 'Liverpool'}): 'Northwest Derby',
    frozenset({'Chelsea', 'Arsenal'}): 'London Derby',
    frozenset({'Chelsea', 'Tottenham'}): 'London Derby',
    frozenset({'West Ham', 'Tottenham'}): 'London Derby',
    frozenset({'Crystal Palace', 'Brighton'}): 'M23 Derby',
    frozenset({'Nottm Forest', 'Leicester'}): 'East Midlands Derby',
    frozenset({'Wolves', 'Aston Villa'}): 'West Midlands Derby',
}

# ══════════════════════════════════════════════════════════════
# FIX 1: SAFE ALIASES - exact matches only, no substring traps
# ══════════════════════════════════════════════════════════════
ALIASES: Dict[str, str] = {
    'manchester united': 'Man United',
    'manchester city': 'Man City',
    'tottenham hotspur': 'Tottenham',
    'tottenham': 'Tottenham',
    'newcastle united': 'Newcastle',
    'west ham united': 'West Ham',
    'wolverhampton': 'Wolves',
    'wolverhampton wanderers': 'Wolves',
    'nottingham forest': 'Nottm Forest',
    'leicester city': 'Leicester',
    'brighton & hove albion': 'Brighton',
    'brighton': 'Brighton',
    'crystal palace': 'Crystal Palace',
    'aston villa': 'Aston Villa',
    'arsenal': 'Arsenal',
    'chelsea': 'Chelsea',
    'liverpool': 'Liverpool',
    'everton': 'Everton',
    'brentford': 'Brentford',
    'fulham': 'Fulham',
    'bournemouth': 'Bournemouth',
    'luton town': 'Luton',
    'sheffield united': 'Sheffield Utd',
    'burnley': 'Burnley',
    'ipswich town': 'Ipswich',
    'southampton': 'Southampton',
    'sunderland': 'Sunderland',
}

# ══════════════════════════════════════════════════════════════
# LOAD OPTIONAL EXTERNAL FILES
# ══════════════════════════════════════════════════════════════
# FIX 1b: TEAMS_MAP is now a clean name→canonical mapping ONLY
# Keys must be lowercase canonical names, values are display names
# Example: {"arsenal fc": "Arsenal", "west ham united fc": "West Ham"}
# DO NOT use match-title strings as keys.
TEAMS_MAP: Dict[str, str] = {}
ELO_RATINGS: Dict[str, float] = {}

def _build_safe_teams_map(raw: dict) -> Dict[str, str]:
    """
    FIX 1b: Filter TEAMS_MAP to only include safe, short team-name keys.
    Rejects any key that looks like a match title (contains ' v ' or digits or is longer than 40 chars)
    to prevent cross-contamination.
    """
    safe = {}
    for k, v in raw.items():
        k_str = str(k).strip()
        # Reject keys that look like match titles or contain numbers
        if (
            ' v ' in k_str.lower()
            or any(ch.isdigit() for ch in k_str)
            or len(k_str) > 40
        ):
            continue
        safe[k_str.lower()] = str(v)
    return safe

def load_external_files():
    """Load optional external data files safely."""
    global TEAMS_MAP, ELO_RATINGS
    if Path(TEAMS_MAP_FILE).exists():
        try:
            with open(TEAMS_MAP_FILE, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            TEAMS_MAP = _build_safe_teams_map(raw)
        except Exception:
            TEAMS_MAP = {}
    if Path(ELO_RATINGS_FILE).exists():
        try:
            with open(ELO_RATINGS_FILE, 'rb') as f:
                raw_elo = pickle.load(f)
            if isinstance(raw_elo, dict):
                for key, val in raw_elo.items():
                    if isinstance(key, str) and isinstance(val, (int, float)):
                        ELO_RATINGS[key] = float(val)
        except Exception:
            ELO_RATINGS = {}

load_external_files()

# ══════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════
def poisson_pmf(k: int, mu: float) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return (mu ** k) * math.exp(-mu) / math.factorial(k)

def safe_div(a: float, b: float, d: float = 0.0) -> float:
    return a / b if b else d

def norm_name(n: str) -> str:
    """
    FIX 1: Safe name normalisation.
    Priority: exact alias → prefix alias → safe TEAMS_MAP exact match.
    Never uses substring containment on TEAMS_MAP keys to prevent cross-team contamination
    (e.g. 'West Ham' matching an Arsenal key).
    """
    lo = n.lower().strip()
    # 1. Exact alias match
    if lo in ALIASES:
        return ALIASES[lo]
    # 2. Alias prefix/suffix match (still safe — ALIASES keys are short)
    for k, v in ALIASES.items():
        if lo.startswith(k) or k.startswith(lo):
            return v
    # 3. TEAMS_MAP — exact match ONLY (no substring containment)
    if TEAMS_MAP:
        if lo in TEAMS_MAP:
            return TEAMS_MAP[lo]
        # Allow singular word exact match only if unambiguous
        lo_words = set(lo.split())
        candidates = [
            v for k, v in TEAMS_MAP.items()
            if set(k.split()) == lo_words
        ]
        if len(candidates) == 1:
            return candidates[0]
    return n

def is_derby(h: str, a: str) -> Optional[str]:
    return RIVALRIES.get(frozenset({norm_name(h), norm_name(a)}))

def parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        c = s.replace('Z', '')
        fmt = '%Y-%m-%dT%H:%M:%S' if 'T' in c else '%Y-%m-%d %H:%M:%S'
        return datetime.strptime(c[:19], fmt)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# COLOUR HELPERS (CLI)
# ══════════════════════════════════════════════════════════════
class C:
    H = '\033[95m'
    B = '\033[94m'
    CN = '\033[96m'
    G = '\033[92m'
    Y = '\033[93m'
    R = '\033[91m'
    BD = '\033[1m'
    DM = '\033[2m'
    E = '\033[0m'
    W = '\033[97m'

    @staticmethod
    def bold(t):
        return f"{C.BD}{t}{C.E}"

    @staticmethod
    def green(t):
        return f"{C.G}{t}{C.E}"

    @staticmethod
    def red(t):
        return f"{C.R}{t}{C.E}"

    @staticmethod
    def yellow(t):
        return f"{C.Y}{t}{C.E}"

    @staticmethod
    def cyan(t):
        return f"{C.CN}{t}{C.E}"

    @staticmethod
    def blue(t):
        return f"{C.B}{t}{C.E}"

    @staticmethod
    def dim(t):
        return f"{C.DM}{t}{C.E}"

    @staticmethod
    def magenta(t):
        return f"{C.H}{t}{C.E}"

    @staticmethod
    def form_char(ch):
        if ch == 'W':
            return f"{C.G}{C.BD}W{C.E}"
        if ch == 'D':
            return f"{C.Y}{C.BD}D{C.E}"
        if ch == 'L':
            return f"{C.R}{C.BD}L{C.E}"
        return ch

    @staticmethod
    def form_str(s):
        return ' '.join(C.form_char(c) for c in s)

    @staticmethod
    def pct_bar(v, w=20, color=None):
        color = color or C.G
        f = int(max(0.0, min(1.0, v)) * w)
        e = w - f
        return f"{color}{'█' * f}{C.E}{C.DM}{'░' * e}{C.E}"

    @staticmethod
    def conf_color(c):
        if c >= 60:
            return C.G
        if c >= 45:
            return C.Y
        return C.R

    @staticmethod
    def value_ind(v):
        if v > 10:
            return f"{C.G}{C.BD}🔥 VALUE!{C.E}"
        if v > 5:
            return f"{C.G}✓ Good{C.E}"
        if v > 0:
            return f"{C.Y}~ Marginal{C.E}"
        return f"{C.DM}✗ No{C.E}"

def box(t):
    return f" {C.blue('│')} {t}"

# ══════════════════════════════════════════════════════════════
# API CLIENTS
# ══════════════════════════════════════════════════════════════
class FootballAPI:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({
            'X-Auth-Token': token,
            'Accept': 'application/json'
        })
        self._c: Dict[str, dict] = {}
        self._t: float = 0.0

    def _rl(self):
        e = time.time() - self._t
        if e < 6.5:
            time.sleep(6.5 - e)
        self._t = time.time()

    def _get(self, ep: str, p: dict = None, cache: bool = True):
        p = p or {}
        k = hashlib.md5(
            f"{ep}|{json.dumps(p, sort_keys=True)}".encode()
        ).hexdigest()
        if cache and k in self._c:
            return self._c[k]
        try:
            self._rl()
            r = self.s.get(
                f"{FOOTBALL_DATA_URL}/{ep}",
                params=p,
                timeout=30
            )
            if r.status_code == 429:
                wait = int(r.headers.get('X-RequestCounter-Reset', 60)) + 1
                time.sleep(wait)
                return self._get(ep, p, cache)
            if r.status_code in (401, 403, 404):
                return None
            r.raise_for_status()
            d = r.json()
            if cache:
                self._c[k] = d
            return d
        except Exception:
            return None

    def season_year(self) -> Optional[int]:
        d = self._get(f"competitions/{PL}")
        if d and d.get('currentSeason'):
            try:
                return int(d['currentSeason']['startDate'][:4])
            except Exception:
                pass
        return None

    def matchday(self) -> int:
        d = self._get(f"competitions/{PL}")
        if d and d.get('currentSeason'):
            return d['currentSeason'].get('currentMatchday', 1)
        return 1

    def finished(self, season: int = None) -> List[dict]:
        p = {'status': 'FINISHED'}
        if season:
            p['season'] = season
        d = self._get(f"competitions/{PL}/matches", p)
        if d and 'matches' in d:
            m = d['matches']
            m.sort(key=lambda x: x.get('utcDate', ''))
            return m
        return []

    def upcoming(self, days: int = 14) -> List[dict]:
        t = datetime.now().strftime('%Y-%m-%d')
        e = (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
        d = self._get(f"competitions/{PL}/matches", {
            'status': 'SCHEDULED,TIMED',
            'dateFrom': t,
            'dateTo': e
        })
        if d and 'matches' in d:
            m = d['matches']
            m.sort(key=lambda x: x.get('utcDate', ''))
            return m
        return []

    def scheduled(self, season: int = None) -> List[dict]:
        p = {'status': 'SCHEDULED,TIMED'}
        if season:
            p['season'] = season
        d = self._get(f"competitions/{PL}/matches", p)
        if d and 'matches' in d:
            m = d['matches']
            m.sort(key=lambda x: x.get('utcDate', ''))
            return m[:30]
        return []

class OddsAPI:
    def __init__(self, key: str):
        self.key = key
        self.cache: Dict[str, dict] = {}

    def ok(self) -> bool:
        return bool(self.key) and len(self.key) > 10

    def fetch(self) -> Dict[str, dict]:
        if not self.ok():
            return {}
        try:
            r = requests.get(
                ODDS_API_URL,
                params={
                    'apiKey': self.key,
                    'regions': 'uk,eu',
                    'markets': 'h2h,totals',
                    'oddsFormat': 'decimal'
                },
                timeout=15
            )
            if r.status_code != 200:
                return {}
            result: Dict[str, dict] = {}
            for ev in r.json():
                h = ev.get('home_team', '')
                a = ev.get('away_team', '')
                bms = ev.get('bookmakers', [])
                if not bms:
                    continue
                ah, ad, aa, ao, au = [], [], [], [], []
                for bm in bms:
                    for mk in bm.get('markets', []):
                        if mk['key'] == 'h2h':
                            for o in mk.get('outcomes', []):
                                if o['name'] == h:
                                    ah.append(o['price'])
                                elif o['name'] == a:
                                    aa.append(o['price'])
                                elif o['name'] == 'Draw':
                                    ad.append(o['price'])
                        elif mk['key'] == 'totals':
                            for o in mk.get('outcomes', []):
                                if o['name'] == 'Over':
                                    ao.append(o['price'])
                                elif o['name'] == 'Under':
                                    au.append(o['price'])
                if ah and ad and aa:
                    avh = sum(ah) / len(ah)
                    avd = sum(ad) / len(ad)
                    ava = sum(aa) / len(aa)
                    ih, id_, ia = 1/avh, 1/avd, 1/ava
                    result[f"{h}_vs_{a}".lower()] = {
                        'home_team': h,
                        'away_team': a,
                        'odds_home': round(avh, 2),
                        'odds_draw': round(avd, 2),
                        'odds_away': round(ava, 2),
                        'implied_home': round(ih, 4),
                        'implied_draw': round(id_, 4),
                        'implied_away': round(ia, 4),
                        'implied_1x': round(ih + id_, 4),
                        'implied_x2': round(ia + id_, 4),
                        'implied_12': round(ih + ia, 4),
                        'odds_1x': round(1/(ih+id_-0.05),2) if (ih+id_)>0.05 else None,
                        'odds_x2': round(1/(ia+id_-0.05),2) if (ia+id_)>0.05 else None,
                        'odds_12': round(1/(ih+ia -0.05),2) if (ih+ia) >0.05 else None,
                        'odds_over25': round(sum(ao)/len(ao),2) if ao else None,
                        'num_bm': len(bms)
                    }
            self.cache = result
            return result
        except Exception:
            return {}

    def find(self, hn: str, an: str) -> Optional[dict]:
        if not self.cache:
            self.fetch()
        hl, al = hn.lower(), an.lower()
        for k, d in self.cache.items():
            oh = d['home_team'].lower()
            oa = d['away_team'].lower()
            hm = (hl in oh or oh in hl or any(w in oh for w in hl.split() if len(w) > 3))
            am = (al in oa or oa in al or any(w in oa for w in al.split() if len(w) > 3))
            if hm and am:
                return d
        return None

# ══════════════════════════════════════════════════════════════
# TEAM
# ══════════════════════════════════════════════════════════════
class Team:
    def __init__(self, tid: int, name: str):
        self.id = tid
        self.name = name
        self.played = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.gf = 0
        self.ga = 0
        self.pts = 0
        self.pos = 0
        self.h_p = 0
        self.h_w = 0
        self.h_d = 0
        self.h_gf = 0
        self.h_ga = 0
        self.a_p = 0
        self.a_w = 0
        self.a_d = 0
        self.a_gf = 0
        self.a_ga = 0
        self.results: List[Tuple] = []
        self.elo = ELO_RATINGS.get(name, ELO_INIT)
        self.elo_hist = [self.elo]
        self.match_dates: List[datetime] = []
        self.cs = 0
        self.fts = 0
        self._last_draw = False
        self.consec_draws = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.unbeaten = 0

    # ── derived properties ──────────────────────────────────
    @property
    def gd(self):
        return self.gf - self.ga

    @property
    def avg_gf(self):
        return safe_div(self.gf, self.played)

    @property
    def avg_ga(self):
        return safe_div(self.ga, self.played)

    @property
    def h_avg_gf(self):
        return safe_div(self.h_gf, self.h_p)

    @property
    def h_avg_ga(self):
        return safe_div(self.h_ga, self.h_p)

    @property
    def a_avg_gf(self):
        return safe_div(self.a_gf, self.a_p)

    @property
    def a_avg_ga(self):
        return safe_div(self.a_ga, self.a_p)

    @property
    def h_wr(self):
        return safe_div(self.h_w, self.h_p, 0.45)

    @property
    def a_wr(self):
        return safe_div(self.a_w, self.a_p, 0.30)

    @property
    def wr(self):
        return safe_div(self.wins, self.played)

    @property
    def dr(self):
        return safe_div(self.draws, self.played)

    @property
    def h_dr(self):
        return safe_div(self.h_d, self.h_p)

    @property
    def a_dr(self):
        return safe_div(self.a_d, self.a_p)

    @property
    def cs_r(self):
        return safe_div(self.cs, self.played)

    @property
    def fts_r(self):
        return safe_div(self.fts, self.played)

    @property
    def ppg(self):
        return safe_div(self.pts, self.played)

    @property
    def form_score(self) -> float:
        """Exponential decay weighted form score 0-100."""
        rec = self.results[-FORM_N:]
        if not rec:
            return 50.0
        total = max_t = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.3 * (i - len(rec) + 1))
            pts = {'W': 3, 'D': 1, 'L': 0}[r[0]]
            total += pts * w
            max_t += 3 * w
        return (total / max_t) * 100 if max_t else 50.0

    @property
    def goal_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_gf
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[1] * w
            wt += w
        return total / wt if wt else self.avg_gf

    @property
    def defense_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.avg_ga
        total = wt = 0.0
        for i, r in enumerate(rec):
            w = math.exp(0.2 * (i - len(rec) + 1))
            total += r[2] * w
            wt += w
        return total / wt if wt else self.avg_ga

    @property
    def draw_form(self) -> float:
        rec = self.results[-FORM_N:]
        if not rec:
            return self.dr
        return sum(1 for r in rec if r[0] == 'D') / len(rec)

    @property
    def form_string(self) -> str:
        return ''.join(r[0] for r in self.results[-6:])

    @property
    def elo_tier(self) -> str:
        if self.elo >= 1650:
            return "Elite"
        if self.elo >= 1550:
            return "Strong"
        if self.elo >= 1475:
            return "Average"
        if self.elo >= 1400:
            return "Weak"
        return "Struggling"

    @property
    def momentum(self) -> int:
        if self.win_streak >= 5:
            return 90
        if self.win_streak >= 3:
            return 60 + self.win_streak * 5
        if self.win_streak >= 2:
            return 40
        if self.unbeaten >= 5:
            return 30
        if self.loss_streak >= 4:
            return -80
        if self.loss_streak >= 3:
            return -50
        if self.loss_streak >= 2:
            return -25
        return 0

    @property
    def volatility(self) -> float:
        rec = self.results[-10:]
        if len(rec) < 4:
            return 0.5
        goals = [r[1] + r[2] for r in rec]
        mean = sum(goals) / len(goals)
        var = sum((g - mean) ** 2 for g in goals) / len(goals)
        return min(1.0, math.sqrt(var) / 2.0)

    def days_rest(self, ref: datetime = None) -> int:
        if not self.match_dates:
            return 7
        ref = ref or datetime.now()
        return max(0, (ref - max(self.match_dates)).days)

    def matches_in(self, n: int = 14, ref: datetime = None) -> int:
        ref = ref or datetime.now()
        cut = ref - timedelta(days=n)
        return sum(1 for d in self.match_dates if d >= cut)

# ══════════════════════════════════════════════════════════════
# ELO SYSTEM
# ══════════════════════════════════════════════════════════════
class EloSystem:
    def __init__(self):
        self.k = ELO_K
        self.ha = ELO_HOME

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def gd_mult(self, gd: int) -> float:
        gd = abs(gd)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        return (11 + gd) / 8

    def update(self, h: Team, a: Team, hg: int, ag: int):
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh
        if hg > ag:
            ah, aa = 1.0, 0.0
        elif hg < ag:
            ah, aa = 0.0, 1.0
        else:
            ah, aa = 0.5, 0.5
        m = self.gd_mult(hg - ag)
        kh = self.k * (1.5 if h.played < 5 else (0.85 if h.elo > 1600 else 1.0))
        ka = self.k * (1.5 if a.played < 5 else (0.85 if a.elo > 1600 else 1.0))
        h.elo += kh * m * (ah - eh)
        a.elo += ka * m * (aa - ea)
        h.elo_hist.append(h.elo)
        a.elo_hist.append(a.elo)

    def predict(self, h: Team, a: Team) -> Tuple[float, float, float]:
        ha = h.elo + self.ha
        eh = self.expected(ha, a.elo)
        ea = 1 - eh
        dd = abs(ha - a.elo)
        db = max(0.18, 0.32 - dd / 1200)
        hw = eh * (1 - db)
        aw = ea * (1 - db)
        t = hw + db + aw
        return (hw/t, db/t, aw/t)

# ══════════════════════════════════════════════════════════════
# DIXON-COLES + DRAW PREDICTOR + FATIGUE
# ══════════════════════════════════════════════════════════════
class DixonColes:
    @staticmethod
    def tau(hg, ag, lh, la, rho):
        if hg == 0 and ag == 0:
            return 1 - lh * la * rho
        if hg == 0 and ag == 1:
            return 1 + lh * rho
        if hg == 1 and ag == 0:
            return 1 + la * rho
        if hg == 1 and ag == 1:
            return 1 - rho
        return 1.0

    @staticmethod
    def prob(hg, ag, lh, la, rho=-0.13):
        b = poisson_pmf(hg, lh) * poisson_pmf(ag, la)
        return max(0, b * DixonColes.tau(hg, ag, lh, la, rho))

    @staticmethod
    def predict(hxg, axg, rho=-0.13, mg=7):
        hw = dr = aw = 0.0
        for i in range(mg):
            for j in range(mg):
                p = DixonColes.prob(i, j, hxg, axg, rho)
                if i > j:
                    hw += p
                elif i == j:
                    dr += p
                else:
                    aw += p
        t = hw + dr + aw
        return (hw/t, dr/t, aw/t) if t > 0 else (0.4, 0.25, 0.35)

    @staticmethod
    def matrix(hxg, axg, rho=-0.13, mg=7):
        return {
            (i, j): DixonColes.prob(i, j, hxg, axg, rho)
            for i in range(mg) for j in range(mg)
        }

class DrawPredictor:
    @staticmethod
    def predict(h: Team, a: Team, derby: bool = False, elo_d: float = 0) -> Tuple[float, float, float]:
        boost = 0.0
        ad = abs(elo_d)
        if ad < 30:
            boost += 0.08
        elif ad < 60:
            boost += 0.05
        elif ad < 100:
            boost += 0.02
        avg_dr = (h.dr + a.dr) / 2
        if avg_dr > 0.35:
            boost += 0.06
        elif avg_dr > 0.25:
            boost += 0.03
        if (h.draw_form + a.draw_form) / 2 > 0.3:
            boost += 0.05
        if derby:
            boost += 0.04
        if (h.cs_r + a.cs_r) / 2 > 0.35:
            boost += 0.04
        if (h.fts_r + a.fts_r) / 2 > 0.25:
            boost += 0.03
        if (h.volatility + a.volatility) / 2 < 0.3:
            boost += 0.03
        if abs(h.momentum) > 50 or abs(a.momentum) > 50:
            boost -= 0.03
        bd = min(0.42, 0.25 + boost)
        rem = 1.0 - bd
        if elo_d > 0:
            hp, ap = rem * 0.58, rem * 0.42
        elif elo_d < 0:
            hp, ap = rem * 0.42, rem * 0.58
        else:
            hp, ap = rem * 0.50, rem * 0.50
        return (hp, bd, ap)

class Fatigue:
    @staticmethod
    def score(t: Team, ref: datetime = None) -> float:
        ref = ref or datetime.now()
        rd = t.days_rest(ref)
        m14 = t.matches_in(14, ref)
        m30 = t.matches_in(30, ref)
        rs = {0:40,1:40,2:40,3:30,4:20,5:10}.get(rd, 0 if rd <= 7 else -5)
        d14 = 35 if m14 >= 5 else (25 if m14 >= 4 else (15 if m14 >= 3 else 0))
        d30 = 25 if m30 >= 9 else (15 if m30 >= 7 else 0)
        return max(0.0, min(100.0, rs + d14 + d30))

    @staticmethod
    def impact(t: Team, ref: datetime = None) -> float:
        return 1.05 - (Fatigue.score(t, ref) / 100) * 0.17

    @staticmethod
    def predict(h: Team, a: Team, ref: datetime = None) -> Tuple[float, float, float]:
        hi = Fatigue.impact(h, ref)
        ai = Fatigue.impact(a, ref)
        t = hi + ai
        if t == 0:
            return (0.4, 0.25, 0.35)
        hp = hi / t
        ap = ai / t
        d = max(0.18, 0.30 - abs(hp - ap) * 0.3)
        hp *= (1 - d)
        ap *= (1 - d)
        tt = hp + d + ap
        return (hp/tt, d/tt, ap/tt)

# ══════════════════════════════════════════════════════════════
# CALIBRATOR
# ══════════════════════════════════════════════════════════════
class Calibrator:
    def __init__(self):
        self.ok = False
        self.models: Dict[str, IsotonicRegression] = {}
        self.hist: List[dict] = []

    def add(self, probs: Tuple[float, float, float], actual: str):
        self.hist.append({'probs': probs, 'actual': actual})

    def calibrate(self) -> bool:
        if not ML_AVAILABLE or len(self.hist) < 30:
            return False
        try:
            for idx, out in enumerate(['HOME', 'DRAW', 'AWAY']):
                ps = np.array([h['probs'][idx] for h in self.hist])
                ac = np.array([1 if h['actual'] == out else 0 for h in self.hist])
                iso = IsotonicRegression(
                    y_min=0.01, y_max=0.99, out_of_bounds='clip'
                )
                iso.fit(ps, ac)
                self.models[out] = iso
            self.ok = True
            return True
        except Exception:
            return False

    def adjust(self, probs: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if not self.ok:
            return probs
        try:
            adj = []
            for i, out in enumerate(['HOME', 'DRAW', 'AWAY']):
                if out in self.models:
                    adj.append(float(self.models[out].predict([probs[i]])[0]))
                else:
                    adj.append(probs[i])
            t = sum(adj)
            return tuple(p / t for p in adj) if t > 0 else probs
        except Exception:
            return probs

    def save(self, fn: str = CALIBRATION_FILE):
        try:
            with open(fn, 'wb') as f:
                pickle.dump({'hist': self.hist, 'ok': self.ok, 'models': self.models}, f)
        except Exception:
            pass

    def load(self, fn: str = CALIBRATION_FILE) -> bool:
        try:
            if Path(fn).exists():
                with open(fn, 'rb') as f:
                    d = pickle.load(f)
                self.hist = d.get('hist', [])
                self.ok = d.get('ok', False)
                self.models = d.get('models', {})
                return True
        except Exception:
            pass
        return False

# ══════════════════════════════════════════════════════════════
# DATA PROCESSOR
# ══════════════════════════════════════════════════════════════
class DataProc:
    def __init__(self):
        self.teams: Dict[int, Team] = {}
        self.elo = EloSystem()
        self.avg_h = 1.53
        self.avg_a = 1.16
        self.total = 0
        self.fixes: List[dict] = []
        self.h2h: Dict[str, List[dict]] = defaultdict(list)

    def process(self, matches: List[dict], do_elo: bool = True):
        matches.sort(key=lambda m: m.get('utcDate', ''))
        cnt = 0
        for m in matches:
            r = self._ext(m)
            if not r:
                continue
            hid, hn, aid, an, hg, ag, ds = r
            if hid not in self.teams:
                self.teams[hid] = Team(hid, hn)
            if aid not in self.teams:
                self.teams[aid] = Team(aid, an)
            h = self.teams[hid]
            a = self.teams[aid]
            md = parse_date(ds)
            if md:
                h.match_dates.append(md)
                a.match_dates.append(md)
            if do_elo:
                self.elo.update(h, a, hg, ag)

            h.played += 1
            a.played += 1
            h.gf += hg
            h.ga += ag
            a.gf += ag
            a.ga += hg
            h.h_p += 1
            h.h_gf += hg
            h.h_ga += ag
            a.a_p += 1
            a.a_gf += ag
            a.a_ga += hg
            if ag == 0:
                h.cs += 1
            if hg == 0:
                a.cs += 1
            h.fts += 1
            if ag == 0 and hg > 0:
                a.fts += 1

            draw = (hg == ag)
            if hg > ag:
                h.wins += 1
                h.h_w += 1
                a.losses += 1
                h.pts += 3
                h.results.append(('W', hg, ag, ds))
                a.results.append(('L', ag, hg, ds))
                h.win_streak += 1
                h.loss_streak = 0
                h.unbeaten += 1
                a.win_streak = 0
                a.loss_streak += 1
                a.unbeaten = 0
            elif hg < ag:
                a.wins += 1
                a.a_w += 1
                h.losses += 1
                a.pts += 3
                h.results.append(('L', hg, ag, ds))
                a.results.append(('W', ag, hg, ds))
                a.win_streak += 1
                a.loss_streak = 0
                a.unbeaten += 1
                h.win_streak = 0
                h.loss_streak += 1
                h.unbeaten = 0
            else:
                h.draws += 1
                a.draws += 1
                h.h_d += 1
                a.a_d += 1
                h.pts += 1
                a.pts += 1
                h.results.append(('D', hg, ag, ds))
                a.results.append(('D', ag, hg, ds))
                h.win_streak = 0
                a.win_streak = 0
                h.loss_streak = 0
                a.loss_streak = 0
                h.unbeaten += 1
                a.unbeaten += 1
                h.consec_draws = (h.consec_draws+1 if (draw and h._last_draw) else (1 if draw else 0))
                a.consec_draws = (a.consec_draws+1 if (draw and a._last_draw) else (1 if draw else 0))
                h._last_draw = draw
                a._last_draw = draw

            key = f"{min(hid,aid)}_{max(hid,aid)}"
            self.h2h[key].append({
                'home_id': hid,
                'away_id': aid,
                'home_goals': hg,
                'away_goals': ag,
                'date': ds
            })
            self.fixes.append({
                'home_id': hid,
                'away_id': aid,
                'home_goals': hg,
                'away_goals': ag,
                'date': ds,
                'home_name': hn,
                'away_name': an
            })
            cnt += 1
        self.total += cnt
        self._avgs()
        self._rank()

    def _ext(self, m: dict):
        if m.get('status') != 'FINISHED':
            return None
        ht = m.get('homeTeam', {})
        at = m.get('awayTeam', {})
        hid = ht.get('id')
        aid = at.get('id')
        if not hid or not aid:
            return None
        hn = ht.get('shortName') or ht.get('name', '?')
        an = at.get('shortName') or at.get('name', '?')
        ft = m.get('score', {}).get('fullTime', {})
        hg = ft.get('home')
        ag = ft.get('away')
        if hg is None or ag is None:
            return None
        return (hid, hn, aid, an, int(hg), int(ag), m.get('utcDate', ''))

    def get_h2h(self, t1: int, t2: int) -> List[dict]:
        return self.h2h.get(f"{min(t1,t2)}_{max(t1,t2)}", [])

    def _avgs(self):
        th = sum(t.h_gf for t in self.teams.values())
        ta = sum(t.a_gf for t in self.teams.values())
        tm = sum(t.h_p for t in self.teams.values())
        if tm:
            self.avg_h = th / tm
            self.avg_a = ta / tm

    def _rank(self):
        for i, t in enumerate(
            sorted(self.teams.values(), key=lambda t: (t.pts, t.gd, t.gf), reverse=True),
            1
        ):
            t.pos = i

    # FIX 7: Graceful lookup — never raises KeyError for unknown teams
    def team_by_name(self, name: str) -> Optional[Team]:
        """Case-insensitive partial-name team lookup."""
        lo = name.lower().strip()
        # Exact match first
        for t in self.teams.values():
            if t.name.lower() == lo:
                return t
        # Partial match
        for t in self.teams.values():
            if lo in t.name.lower() or t.name.lower() in lo:
                return t
        return None

# ══════════════════════════════════════════════════════════════
# ML v3.3 – ALL FIVE DATA-SCIENCE FIXES APPLIED
#
# FIX 2: Use sklearn Pipeline → Imputer → Scaler → Classifier
#   so scaling is ALWAYS fitted on training folds only
#   (no data leakage into CV test folds).
#
# FIX 3: class_weight='balanced' on every base learner +
#   StratifiedKFold with shuffle to keep class ratios.
#
# FIX 4: SimpleImputer(strategy='mean') replaces np.nan_to_num
#   so missing values get the training-set mean, not 0.
#
# FIX 5: Guard against too-few samples per class before CV;
#   fall back gracefully to a single RandomForest.
#
# FIX 6: External model file must contain a full Pipeline
#   (imputer + scaler + classifier); we validate n_features
#   via the pipeline's last step, not the pipeline itself.
# ══════════════════════════════════════════════════════════════
class MLPred:
    N_FEATURES = 58  # expected feature count

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None  # full sklearn Pipeline
        self.trained = False
        self.acc = 0.0
        self._external = False

    # ── feature engineering (unchanged logic, 58 features) ──
    def feats(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> List[float]:
        ah = max(data.avg_h, 0.5)
        aa = max(data.avg_a, 0.5)
        return [
            # Elo (3)
            h.elo, a.elo, h.elo - a.elo,
            # Form scores (4)
            h.form_score, a.form_score, h.form_score - a.form_score, abs(h.form_score - a.form_score),
            # Goal averages (6)
            h.h_avg_gf, a.a_avg_gf, h.goal_form, a.goal_form, h.goal_form - a.goal_form, h.h_avg_gf - a.a_avg_gf,
            # Defense (6)
            h.h_avg_ga, a.a_avg_ga, h.defense_form, a.defense_form, h.defense_form - a.defense_form, h.h_avg_ga - a.a_avg_ga,
            # Attack/Defense ratios (4)
            safe_div(h.h_avg_gf, ah, 1), safe_div(a.a_avg_gf, aa, 1),
            safe_div(h.h_avg_ga, ah, 1), safe_div(a.a_avg_ga, aa, 1),
            # Win rates (4)
            h.h_wr, a.a_wr, h.wr, a.wr,
            # League position (7)
            h.pos, a.pos, a.pos - h.pos, h.pts, a.pts, h.ppg - a.ppg, h.gd,
            # GD away (1)
            a.gd,
            # Clean sheets (4)
            h.cs_r, a.cs_r, h.fts_r, a.fts_r,
            # Fatigue (2)
            Fatigue.score(h, md), Fatigue.score(a, md),
            # Draw rates (7)
            h.dr, a.dr, (h.dr + a.dr) / 2, h.draw_form, a.draw_form, h.h_dr, a.a_dr,
            # Volatility (2)
            h.volatility, a.volatility,
            # Derby (1)
            1.0 if derby else 0.0,
            # Elo scaled (1)
            abs(h.elo - a.elo) / 100,
            # Momentum (7)
            h.momentum / 100, a.momentum / 100, (h.momentum - a.momentum) / 100,
            h.win_streak, a.win_streak, h.loss_streak, a.loss_streak,
        ]

    # ── FIX 6: external model must be a Pipeline ────────────
    def _try_load_external(self) -> bool:
        """Load a pre-trained sklearn Pipeline from disk.
        The pipeline MUST contain imputer+scaler+classifier internally
        so no raw-feature / scaling mismatch can occur.
        """
        if not Path(XGB_MODEL_FILE).exists():
            return False
        try:
            with open(XGB_MODEL_FILE, 'rb') as f:
                loaded = pickle.load(f)
            # Must be a Pipeline
            if not isinstance(loaded, Pipeline):
                return False
            # Validate feature count via the pipeline's last step
            final_step = loaded.steps[-1][1]
            n_feat = getattr(final_step, 'n_features_in_', None)
            if n_feat is not None and n_feat != self.N_FEATURES:
                return False  # incompatible — train from scratch
            self.pipeline = loaded
            self.trained = True
            self._external = True
            return True
        except Exception:
            return False

    # ── FIX 2+3+4+5: leak-free training pipeline ────────────
    def train(self, data: DataProc, fixes: List[dict] = None) -> bool:
        if not ML_AVAILABLE:
            return False
        # Try loading a compatible pre-built pipeline first
        if self._try_load_external():
            return True

        fixes = fixes or data.fixes
        if len(fixes) < 40:
            return False

        # ── Walk-forward feature extraction ──────────────────
        X: List[List[float]] = []
        y: List[int] = []
        sim = DataProc()
        sf = sorted(fixes, key=lambda f: f.get('date', ''))
        warm = int(len(sf) * 0.30)
        for idx, f in enumerate(sf):
            if idx >= warm:
                ht = sim.teams.get(f['home_id'])
                at = sim.teams.get(f['away_id'])
                if ht and at and ht.played >= 3 and at.played >= 3:
                    try:
                        md = parse_date(f.get('date', ''))
                        derby = bool(is_derby(f['home_name'], f['away_name']))
                        ft = self.feats(ht, at, sim, md, derby)
                        lb = (
                            0 if f['home_goals'] > f['away_goals'] else
                            (1 if f['home_goals'] == f['away_goals'] else 2)
                        )
                        X.append(ft)
                        y.append(lb)
                    except Exception:
                        pass
            fake = {
                'status': 'FINISHED',
                'homeTeam': {'id': f['home_id'], 'shortName': f['home_name']},
                'awayTeam': {'id': f['away_id'], 'shortName': f['away_name']},
                'score': {'fullTime': {'home': f['home_goals'], 'away': f['away_goals']}},
                'utcDate': f.get('date', '')
            }
            sim.process([fake], do_elo=True)

        if len(X) < 30:
            return False

        # FIX 4: raw numpy array — NaNs handled by the Pipeline's Imputer
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.int64)

        # FIX 5: guard against too-few samples per class
        classes, counts = np.unique(y_arr, return_counts=True)
        min_count = int(counts.min())
        if min_count < MIN_SAMPLES_PER_CLASS:
            # Not enough data for reliable CV — return False so the
            # engine runs without ML rather than producing junk.
            return False

        n_splits = min(5, max(2, min_count // 5))

        # ── FIX 2: define the full Pipeline (imputer+scaler+model) ──
        # cross_val_score will fit the ENTIRE pipeline on each
        # training fold, so the scaler never "sees" test fold stats.
        # FIX 3: balanced class weights everywhere
        base_rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            class_weight='balanced',  # FIX 3
            random_state=42,
            n_jobs=1,
        )
        base_gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        estimators = [('rf', base_rf), ('gb', base_gb)]
        if XGBOOST_AVAILABLE:
            # XGBoost handles imbalance via scale_pos_weight or
            # sample_weight; we approximate via class-frequency ratio.
            freq = counts / counts.sum()
            spw = float(freq[0] / freq[1]) if len(freq) > 1 else 1.0
            estimators.append(('xgb', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,  # FIX 3 (binary proxy)
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss',
                verbosity=0,
            )))

        meta_lr = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',  # FIX 3
            solver='lbfgs',
            
        )

        # FIX 5: StratifiedKFold keeps class ratios in every fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        try:
            stacker = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_lr,
                cv=skf,  # FIX 5: stratified folds
                passthrough=False,
                n_jobs=1,
            )
            # FIX 2: wrap classifier in a Pipeline with Imputer + Scaler
            full_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # FIX 4
                ('scaler', StandardScaler()),                # FIX 2
                ('model', CalibratedClassifierCV(stacker, cv=skf)),
            ])

            # CV on the raw X_arr — Pipeline handles imputation+scaling
            # internally on each fold → zero data leakage
            cv_scores = cross_val_score(
                full_pipeline, X_arr, y_arr,
                cv=skf,
                scoring='balanced_accuracy',  # FIX 3: balanced metric
                n_jobs=1,
            )
            self.acc = float(cv_scores.mean())

            # Final fit on ALL training data
            full_pipeline.fit(X_arr, y_arr)
            self.pipeline = full_pipeline
        except Exception as exc:
            # Graceful fallback to a simple RF pipeline
            fallback = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=1,
                )),
            ])
            try:
                fallback.fit(X_arr, y_arr)
                self.pipeline = fallback
                self.acc = 0.0
            except Exception:
                return False

        self.trained = True
        return True

    # ── FIX 2+6: predict always uses the same Pipeline ──────
    def predict(self, h: Team, a: Team, data: DataProc, md: datetime = None, derby: bool = False) -> Optional[Tuple[float, float, float]]:
        if not self.trained or self.pipeline is None:
            return None
        try:
            ft = self.feats(h, a, data, md, derby)
            # FIX 4: pass raw floats; imputer inside pipeline handles NaN
            X = np.array([ft], dtype=np.float64)
            # FIX 2+6: the pipeline contains imputer+scaler+model,
            # so we NEVER apply separate scaling here.
            probs = self.pipeline.predict_proba(X)[0]
            return (float(probs[0]), float(probs[1]), float(probs[2]))
        except Exception:
            return None

    def save_pipeline(self, path: str = XGB_MODEL_FILE):
        """Persist the trained pipeline so it can be reused."""
        if self.pipeline is not None:
            try:
                with open(path, 'wb') as f:
                    pickle.dump(self.pipeline, f)
            except Exception:
                pass

# ══════════════════════════════════════════════════════════════
# PREDICTION RESULT
# ══════════════════════════════════════════════════════════════
class Pred:
    def __init__(self):
        self.home = ""
        self.away = ""
        self.hid = 0
        self.aid = 0
        self.date = ""
        self.hp = self.dp = self.ap = 0.0
        self.raw_hp = self.raw_dp = self.raw_ap = 0.0
        self.hxg = self.axg = 0.0
        self.top_sc: List[Tuple] = []
        self.result = ""
        self.pred_sc = (0, 0)
        self.conf = 0.0
        self.btts = self.o15 = self.o25 = self.o35 = 0.0
        self.dc_1x = self.dc_x2 = self.dc_12 = 0.0
        self.dc_recommend = ""
        self.dc_value_bets: List[dict] = []
        self.h_form = self.a_form = ""
        self.h_pos = self.a_pos = 0
        self.h_elo = self.a_elo = 0.0
        self.h_fat = self.a_fat = 0.0
        self.h_rest = self.a_rest = 0
        self.h_momentum = self.a_momentum = 0
        self.models: Dict[str, Tuple] = {}
        self.odds = None
        self.value_bets: List[dict] = []
        self.ml_used = False
        self.ml_acc = 0.0
        self.calibrated = False
        self.is_derby = False
        self.derby_name = ""

# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════
class Engine:
    def __init__(self, data: DataProc, ml: MLPred = None, odds: OddsAPI = None, cal: Calibrator = None):
        self.data = data
        self.ml = ml
        self.odds = odds
        self.cal = cal
        self.w = dict(WEIGHTS)
        if not ml or not ml.trained:
            mw = self.w.pop('ml', 0.15)
            rem = sum(self.w.values())
            if rem > 0:
                for k in self.w:
                    self.w[k] += mw * (self.w[k] / rem)

    def predict(self, hid: int, aid: int, date: str = "", md: datetime = None) -> Optional[Pred]:
        h = self.data.teams.get(hid)
        a = self.data.teams.get(aid)
        if not h or not a or h.played < 2 or a.played < 2:
            return None
        p = Pred()
        p.home = h.name
        p.away = a.name
        p.hid = hid
        p.aid = aid
        p.date = date
        p.h_form = h.form_string
        p.a_form = a.form_string
        p.h_pos = h.pos
        p.a_pos = a.pos
        p.h_elo = h.elo
        p.a_elo = a.elo
        if md is None and date:
            md = parse_date(date)
        md = md or datetime.now()
        derby = is_derby(h.name, a.name)
        p.is_derby = bool(derby)
        p.derby_name = derby or ""
        p.h_fat = Fatigue.score(h, md)
        p.a_fat = Fatigue.score(a, md)
        p.h_rest = h.days_rest(md)
        p.a_rest = a.days_rest(md)
        p.h_momentum = h.momentum
        p.a_momentum = a.momentum
        p.hxg = self._xg(h, a, True) * Fatigue.impact(h, md)
        p.axg = self._xg(a, h, False) * Fatigue.impact(a, md)
        if h.momentum > 40:
            p.hxg *= 1.05
        elif h.momentum < -40:
            p.hxg *= 0.95
        if a.momentum > 40:
            p.axg *= 1.05
        elif a.momentum < -40:
            p.axg *= 0.95

        models: Dict[str, Tuple[float,float,float]] = {}
        models['dixon_coles'] = DixonColes.predict(p.hxg, p.axg)
        models['elo'] = self.data.elo.predict(h, a)
        models['form'] = self._form(h, a)
        models['h2h'] = self._h2h(hid, aid)
        models['home_advantage'] = self._hadv(h, a)
        models['fatigue'] = Fatigue.predict(h, a, md)
        ed = h.elo + ELO_HOME - a.elo
        models['draw_model'] = DrawPredictor.predict(h, a, p.is_derby, ed)
        if self.ml and self.ml.trained:
            mp = self.ml.predict(h, a, self.data, md, p.is_derby)
            if mp:
                models['ml'] = mp
                p.ml_used = True
                p.ml_acc = self.ml.acc
        p.models = models

        hp = dp = ap = tw = 0.0
        for nm, probs in models.items():
            w = self.w.get(nm, 0)
            if w > 0:
                hp += probs[0] * w
                dp += probs[1] * w
                ap += probs[2] * w
                tw += w
        if tw > 0:
            hp /= tw
            dp /= tw
            ap /= tw
        t = hp + dp + ap
        if t > 0:
            hp /= t
            dp /= t
            ap /= t
        p.raw_hp, p.raw_dp, p.raw_ap = hp, dp, ap
        if self.cal and self.cal.ok:
            hp, dp, ap = self.cal.adjust((hp, dp, ap))
            p.calibrated = True
        p.hp, p.dp, p.ap = hp, dp, ap
        p.dc_1x = hp + dp
        p.dc_x2 = ap + dp
        p.dc_12 = hp + ap
        p.dc_recommend = self._dc_recommend(p)

        mx = DixonColes.matrix(p.hxg, p.axg)
        ss = sorted(mx.items(), key=lambda x: x[1], reverse=True)
        p.top_sc = [(s[0][0], s[0][1], s[1]) for s in ss[:6]]
        p.btts = sum(pr for (hh,aa),pr in mx.items() if hh>0 and aa>0)
        p.o15 = sum(pr for (hh,aa),pr in mx.items() if hh+aa>1)
        p.o25 = sum(pr for (hh,aa),pr in mx.items() if hh+aa>2)
        p.o35 = sum(pr for (hh,aa),pr in mx.items() if hh+aa>3)

        pd_map = {'HOME': hp, 'DRAW': dp, 'AWAY': ap}
        p.result = max(pd_map, key=pd_map.get)
        p.conf = max(pd_map.values()) * 100
        if p.top_sc:
            p.pred_sc = (p.top_sc[0][0], p.top_sc[0][1])

        if self.odds and self.odds.ok():
            od = self.odds.find(h.name, a.name)
            if od:
                p.odds = od
                p.value_bets = self._value(p, od)
                p.dc_value_bets = self._dc_value(p, od)
        return p

    def _dc_recommend(self, p: Pred) -> str:
        recs = []
        if 0.40 <= p.hp <= 0.60 and p.dp > 0.20:
            recs.append(('1X', p.dc_1x, 'Home favored but draw possible'))
        if 0.30 <= p.ap <= 0.50 and p.dp > 0.20:
            recs.append(('X2', p.dc_x2, 'Away has real chance + draw likely'))
        if p.dp < 0.20:
            recs.append(('12', p.dc_12, 'Draw unlikely'))
        if p.is_derby and p.hp > p.ap:
            recs.append(('1X', p.dc_1x, f'{p.derby_name} - Home advantage'))
        if not recs:
            dc_vals = {'1X': p.dc_1x, 'X2': p.dc_x2, '12': p.dc_12}
            best = max(dc_vals, key=dc_vals.get)
            recs.append((best, dc_vals[best], 'Highest probability'))
        recs.sort(key=lambda x: -x[1])
        return f"{recs[0][0]} ({recs[0][1]*100:.1f}%) - {recs[0][2]}"

    def _xg(self, t: Team, opp: Team, home: bool) -> float:
        ah = max(self.data.avg_h, 0.5)
        aa = max(self.data.avg_a, 0.5)
        if home:
            att = safe_div(t.h_avg_gf, ah, 1)
            df = safe_div(opp.a_avg_ga, ah, 1)
            base = ah
        else:
            att = safe_div(t.a_avg_gf, aa, 1)
            df = safe_div(opp.h_avg_ga, aa, 1)
            base = aa
        fa = safe_div(t.goal_form, max(t.avg_gf, 0.5), 1.0)
        fa = 0.7 + 0.3 * min(fa, 2.0)
        return max(0.25, min(att * df * base * fa, 4.5))

    def _form(self, h: Team, a: Team) -> Tuple[float,float,float]:
        hf = h.form_score
        af = a.form_score
        t = hf + af
        if t == 0:
            return (0.4, 0.25, 0.35)
        hs = (hf / t) * 1.08
        a_s = af / t
        diff = abs(hs - a_s)
        d = max(0.15, 0.33 - diff * 0.4)
        rem = 1.0 - d
        sm = hs + a_s
        return (rem * hs / sm, d, rem * a_s / sm)

    def _h2h(self, hid: int, aid: int) -> Tuple[float,float,float]:
        default = (0.40, 0.25, 0.35)
        ms = self.data.get_h2h(hid, aid)
        if not ms:
            return default
        hw = dw = aw = 0
        for m in ms[-10:]:
            if m['home_goals'] > m['away_goals']:
                hw += 1 if m['home_id'] == hid else 0
                aw += 0 if m['home_id'] == hid else 1
            elif m['home_goals'] < m['away_goals']:
                aw += 1 if m['home_id'] == hid else 0
                hw += 0 if m['home_id'] == hid else 1
            else:
                dw += 1
        n = hw + dw + aw
        if n == 0:
            return default
        alpha = 1
        return (
            (hw+alpha)/(n+3*alpha),
            (dw+alpha)/(n+3*alpha),
            (aw+alpha)/(n+3*alpha)
        )

    def _hadv(self, h: Team, a: Team) -> Tuple[float,float,float]:
        hp = h.h_wr * 1.25
        ap = a.a_wr
        sm = hp + ap
        if sm > 0:
            hp /= sm
            ap /= sm
        d = 0.25
        hp *= 0.75
        ap *= 0.75
        t = hp + d + ap
        return (hp/t, d/t, ap/t)

    def _value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, mp, ip, odd in [
            ('Home', p.hp, od['implied_home'], od['odds_home']),
            ('Draw', p.dp, od['implied_draw'], od['odds_draw']),
            ('Away', p.ap, od['implied_away'], od['odds_away']),
        ]:
            edge = (mp - ip) * 100
            kelly = (mp * odd - 1) / (odd - 1) if mp > 0 and odd > 1 else 0
            vals.append({
                'market': nm,
                'model': float(mp * 100),
                'implied': float(ip * 100),
                'odds': float(odd),
                'edge': float(edge),
                'kelly': float(max(0, kelly) * 100),
                'is_value': edge > 3,
            })
        return vals

    def _dc_value(self, p: Pred, od: dict) -> List[dict]:
        vals = []
        for nm, model_p, implied_p, odds_val in [
            ('1X', p.dc_1x, od.get('implied_1x'), od.get('odds_1x')),
            ('X2', p.dc_x2, od.get('implied_x2'), od.get('odds_x2')),
            ('12', p.dc_12, od.get('implied_12'), od.get('odds_12')),
        ]:
            if implied_p is None or odds_val is None:
                continue
            edge = (model_p - implied_p) * 100
            kelly = (model_p * odds_val - 1) / (odds_val - 1) if model_p > 0 and odds_val > 1 else 0
            vals.append({
                'market': f'DC {nm}',
                'model': float(model_p * 100),
                'implied': float(implied_p * 100),
                'odds': float(odds_val),
                'edge': float(edge),
                'kelly': float(max(0, kelly) * 100),
                'is_value': edge > 3,
            })
        return vals

# ══════════════════════════════════════════════════════════════
# BACKTESTER
# ══════════════════════════════════════════════════════════════
class Backtester:
    def __init__(self):
        self.results: dict = {}
        self.cal = Calibrator()

    def run(self, matches: List[dict], split: float = BACKTEST_SPLIT) -> dict:
        fin = [m for m in matches if m.get('status') == 'FINISHED']
        fin.sort(key=lambda m: m.get('utcDate', ''))
        si = int(len(fin) * split)
        train = fin[:si]
        test = fin[si:]
        if len(train) < 30 or len(test) < 10:
            return {'error': 'Not enough data'}

        td = DataProc()
        td.process(train)
        ml = None
        if ML_AVAILABLE:
            ml = MLPred()
            ml.train(td)

        eng = Engine(td, ml)
        cs = len(test) // 2
        cal_set = test[:cs]
        eval_set = test[cs:]

        cr1 = t1 = 0
        for m in cal_set:
            ht = m.get('homeTeam', {})
            at = m.get('awayTeam', {})
            ft = m.get('score', {}).get('fullTime', {})
            hid = ht.get('id')
            aid = at.get('id')
            ahg = ft.get('home')
            aag = ft.get('away')
            if not hid or not aid or ahg is None or aag is None:
                continue
            pr = eng.predict(hid, aid, m.get('utcDate', ''))
            if not pr:
                continue
            ahg, aag = int(ahg), int(aag)
            actual = ('HOME' if ahg>aag else ('AWAY' if ahg<aag else 'DRAW'))
            self.cal.add((pr.hp, pr.dp, pr.ap), actual)
            t1 += 1
            if pr.result == actual:
                cr1 += 1
            td.process([m])

        cal_ok = self.cal.calibrate()
        eng2 = Engine(td, ml, cal=self.cal) if cal_ok else eng

        cr = cr1
        csc = 0
        total = t1
        preds = []
        dc_correct = {'1X':0,'X2':0,'12':0}
        dc_total = {'1X':0,'X2':0,'12':0}
        for m in eval_set:
            ht = m.get('homeTeam', {})
            at = m.get('awayTeam', {})
            ft = m.get('score', {}).get('fullTime', {})
            hid = ht.get('id')
            aid = at.get('id')
            ahg = ft.get('home')
            aag = ft.get('away')
            if not hid or not aid or ahg is None or aag is None:
                continue
            hn = ht.get('shortName') or ht.get('name', '')
            an = at.get('shortName') or at.get('name', '')
            pr = eng2.predict(hid, aid, m.get('utcDate', ''))
            if not pr:
                continue
            ahg, aag = int(ahg), int(aag)
            actual = ('HOME' if ahg>aag else ('AWAY' if ahg<aag else 'DRAW'))
            total += 1
            if pr.result == actual:
                cr += 1
            if pr.pred_sc[0]==ahg and pr.pred_sc[1]==aag:
                csc += 1
            for dc_name, covers in [
                ('1X',['HOME','DRAW']),
                ('X2',['AWAY','DRAW']),
                ('12',['HOME','AWAY'])
            ]:
                dc_total[dc_name] += 1
                if actual in covers:
                    dc_correct[dc_name] += 1
            preds.append({
                'home': hn,
                'away': an,
                'predicted': pr.result,
                'actual': actual,
                'pred_score': pr.pred_sc,
                'actual_score': (ahg, aag),
                'confidence': float(pr.conf),
                'correct': pr.result == actual,
                'probs': (float(pr.hp),float(pr.dp),float(pr.ap)),
                'dc_1x': float(pr.dc_1x),
                'dc_x2': float(pr.dc_x2),
                'dc_12': float(pr.dc_12),
                'calibrated': pr.calibrated,
            })
            td.process([m])

        if total == 0:
            return {'error': 'No matches'}
        ra = cr / total * 100
        sa = csc / total * 100
        brier = 0.0
        for p in preds:
            av = [0,0,0]
            av[['HOME','DRAW','AWAY'].index(p['actual'])] = 1
            for i in range(3):
                brier += (p['probs'][i] - av[i]) ** 2
        brier /= (total * 3)
        hi = [p for p in preds if p['confidence'] > 55]
        me = [p for p in preds if 40 <= p['confidence'] <= 55]
        lo = [p for p in preds if p['confidence'] < 40]
        self.results = {
            'total': total,
            'train': len(train),
            'test': len(test),
            'result_acc': ra,
            'score_acc': sa,
            'brier': float(brier),
            'correct': cr,
            'correct_sc': csc,
            'cal_used': cal_ok,
            'hi_acc': (sum(1 for p in hi if p['correct'])/len(hi)*100 if hi else 0),
            'me_acc': (sum(1 for p in me if p['correct'])/len(me)*100 if me else 0),
            'lo_acc': (sum(1 for p in lo if p['correct'])/len(lo)*100 if lo else 0),
            'hi_n': len(hi),
            'me_n': len(me),
            'lo_n': len(lo),
            'predictions': preds,
            'ml_acc': float(ml.acc * 100) if ml and ml.trained else 0,
            'dc_1x_acc': (dc_correct['1X']/dc_total['1X']*100 if dc_total['1X'] else 0),
            'dc_x2_acc': (dc_correct['X2']/dc_total['X2']*100 if dc_total['X2'] else 0),
            'dc_12_acc': (dc_correct['12']/dc_total['12']*100 if dc_total['12'] else 0),
            'dc_1x_n': dc_total['1X'],
            'dc_x2_n': dc_total['X2'],
            'dc_12_n': dc_total['12'],
        }
        return self.results

# ══════════════════════════════════════════════════════════════
# DISPLAY (CLI)
# ══════════════════════════════════════════════════════════════
class Disp:
    @staticmethod
    def header():
        print()
        print(C.cyan(" ╔══════════════════════════════════════════════════════════════╗"))
        print(C.cyan(" ║") + C.bold(" ⚽ PREMIER LEAGUE PREDICTOR PRO v3.3 ⚽ ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim(" Dixon-Coles • Elo • XGBoost • Double Chance ") + C.cyan("║"))
        print(C.cyan(" ║") + C.dim(" Momentum • Calibration • Leak-free Pipeline ") + C.cyan("║"))
        print(C.cyan(" ╚══════════════════════════════════════════════════════════════╝"))
        print()

    @staticmethod
    def section(t):
        print(f"\n {C.yellow(C.bold('══ ' + t + ' ══'))}\n")

    @staticmethod
    def progress(m):
        print(f" {C.cyan('⟳')} {m}")

    @staticmethod
    def success(m):
        print(f" {C.green('✓')} {m}")

    @staticmethod
    def error(m):
        print(f" {C.red('✖')} {m}")

    @staticmethod
    def info(m):
        print(f" {C.blue('ℹ')} {m}")

    @staticmethod
    def standings(teams: Dict[int, Team]):
        Disp.section("📊 Standings")
        print(f" {'#':>3} {'Team':<22} {'P':>3} {'W':>2} {'D':>2} {'L':>2} "
              f"{'GF':>3} {'GA':>3} {'GD':>4} {C.bold('Pts'):>4} {'Elo':>6} Form")
        print(f" {'─'*88}")
        for i, t in enumerate(
            sorted(teams.values(), key=lambda t: (t.pts,t.gd,t.gf), reverse=True),
            1
        ):
            pc = C.G if i<=4 else (C.CN if i<=6 else (C.R if i>=len(teams)-2 else C.W))
            ec = C.green if t.elo>1520 else (C.red if t.elo<1480 else C.yellow)
            mi = "🔥" if t.momentum>40 else ("📉" if t.momentum<-40 else "")
            print(f" {pc}{i:>3}{C.E} {t.name:<22} {t.played:>3} {t.wins:>2} "
                  f"{t.draws:>2} {t.losses:>2} {t.gf:>3} {t.ga:>3} {t.gd:>+4} "
                  f"{C.bold(str(t.pts)):>4} {ec(f'{t.elo:.0f}'):>6} "
                  f"{C.form_str(t.form_string[-5:])} {mi}")

    @staticmethod
    def card(p: Pred, idx: int):
        w = 67
        print(f"\n {C.blue('┌'+'─'*w+'┐')}")
        print(box(f" {C.bold(f'⚽ MATCH #{idx}')}"))
        if p.is_derby:
            print(box(f" {C.magenta(C.bold(f'🔥 {p.derby_name}'))}"))
        print(box(f" {C.bold(C.green('🏠 '+p.home))} {C.dim('vs')} {C.bold(C.red('✈️ '+p.away))}"))
        if p.date:
            dt = parse_date(p.date)
            ds = dt.strftime('%a %d %b %Y • %H:%M') if dt else p.date[:16]
            print(box(f" 📅 {ds}"))
        ed = p.h_elo - p.a_elo
        edc = C.green if ed > 0 else C.red
        print(box(f" 🏆 Elo: {p.h_elo:.0f} vs {p.a_elo:.0f} ({edc(f'{ed:+.0f}')})"))
        def mom_str(m, name):
            if m > 40:
                return C.green(f"🔥 {name} on fire! ({m:+d})")
            if m > 20:
                return C.green(f"📈 {name} rising ({m:+d})")
            if m < -40:
                return C.red(f"📉 {name} struggling ({m:+d})")
            if m < -20:
                return C.red(f"⚠️ {name} poor form ({m:+d})")
            return C.dim(f"→ {name} neutral ({m:+d})")
        if p.h_momentum != 0 or p.a_momentum != 0:
            print(box(f" 💪 {mom_str(p.h_momentum, p.home)}"))
            print(box(f" 💪 {mom_str(p.a_momentum, p.away)}"))
        if p.calibrated:
            print(box(C.dim(" ✅ Calibrated")))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('📊 1X2 PROBABILITIES')}"))
        print(box(f" 🏠 Home: {C.green(f'{p.hp*100:5.1f}%')} {C.pct_bar(p.hp,25,C.G)}"))
        print(box(f" 🤝 Draw: {C.yellow(f'{p.dp*100:5.1f}%')} {C.pct_bar(p.dp,25,C.Y)}"))
        print(box(f" ✈️ Away: {C.red(f'{p.ap*100:5.1f}%')} {C.pct_bar(p.ap,25,C.R)}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('🛡️ DOUBLE CHANCE')}"))
        def dc_bar(val, label, color):
            emoji = "✅" if val>0.70 else ("⚡" if val>0.55 else "⚠️")
            return f" {emoji} {label:<16} {color(f'{val*100:5.1f}%')} {C.pct_bar(val,20,color)}"
        print(box(dc_bar(p.dc_1x,"1X (Home/Draw):",C.green)))
        print(box(dc_bar(p.dc_12,"12 (No Draw):", C.cyan)))
        print(box(dc_bar(p.dc_x2,"X2 (Away/Draw):",C.yellow)))
        print(box(f" {C.bold('💡 Recommend:')} {C.bold(p.dc_recommend)}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        total_xg = p.hxg + p.axg
        print(box(f" {C.bold('⚡ xG:')} {p.home}: {C.bold(f'{p.hxg:.2f}')} "
                  f"{p.away}: {C.bold(f'{p.axg:.2f}')} Total: {C.bold(f'{total_xg:.2f}')}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('🎯 LIKELY SCORES')}"))
        for i,(hg,ag,pr2) in enumerate(p.top_sc[:5]):
            mk = "👉" if i==0 else " "
            bar = C.dim('▓' * int(pr2*60))
            print(box(f" {mk} {hg}-{ag} ({pr2*100:4.1f}%) {bar}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('📈 MARKETS')}"))
        for nm, v, th in [("Over 1.5",p.o15,.5),("Over 2.5",p.o25,.5),
                          ("Over 3.5",p.o35,.5),("BTTS",p.btts,.5)]:
            e = C.green("✅") if v>th else C.red("❌")
            print(box(f" {nm:<20} {e} {v*100:5.1f}%"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('📋 FORM & FATIGUE')}"))
        def fs(s, d):
            if s>50:
                return C.red(f"⚠️ {s:.0f} ({d}d)")
            if s>25:
                return C.yellow(f"😐 {s:.0f} ({d}d)")
            return C.green(f"✅ {s:.0f} ({d}d)")
        print(box(f" {p.home:<18} {C.form_str(p.h_form)} {fs(p.h_fat,p.h_rest)}"))
        print(box(f" {p.away:<18} {C.form_str(p.a_form)} {fs(p.a_fat,p.a_rest)}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('🔬 MODELS')}"))
        for nm,(mh,md2,ma) in p.models.items():
            wt = WEIGHTS.get(nm,0)
            if wt>0 or nm=='ml':
                print(box(f" {nm:<14} ({wt:>4.0%}): "
                          f"H={C.green(f'{mh*100:4.1f}%')} "
                          f"D={C.yellow(f'{md2*100:4.1f}%')} "
                          f"A={C.red(f'{ma*100:4.1f}%')}"))
        if p.value_bets and any(v['is_value'] for v in p.value_bets):
            print(f" {C.blue('├'+'─'*w+'┤')}")
            print(box(f" {C.bold('💰 1X2 VALUE BETS')}"))
            for v in p.value_bets:
                if v['is_value']:
                    edge_val = v['edge']
                    print(box(f" {v['market']:<8} @{v['odds']:.2f} "
                              f"Edge:{C.green(f'+{edge_val:.1f}%')} "
                              f"Kelly:{v['kelly']:.1f}% {C.value_ind(edge_val)}"))

        if p.dc_value_bets and any(v['is_value'] for v in p.dc_value_bets):
            print(f" {C.blue('├'+'─'*w+'┤')}")
            print(box(f" {C.bold('🛡️ DC VALUE BETS')}"))
            for v in p.dc_value_bets:
                if v['is_value']:
                    edge_val = v['edge']
                    print(box(f" {v['market']:<8} @{v['odds']:.2f} "
                              f"Edge:{C.green(f'+{edge_val:.1f}%')} "
                              f"Kelly:{v['kelly']:.1f}% {C.value_ind(edge_val)}"))

        print(f" {C.blue('├'+'─'*w+'┤')}")
        rm = {'HOME': f"🏆 {p.home} Win", 'DRAW': "🤝 Draw", 'AWAY': f"🏆 {p.away} Win"}
        cc = C.conf_color(p.conf)
        em = "🔥" if p.conf>55 else ("⚡" if p.conf>40 else "⚠️")
        print(box(""))
        print(box(f" 🎯 {C.bold('PREDICTION:')} {C.bold(rm.get(p.result,'?'))}"))
        print(box(f" 📊 {C.bold('SCORE:')} {C.bold(f'{p.pred_sc[0]} - {p.pred_sc[1]}')}"))
        print(box(f" {em} {C.bold('CONFIDENCE:')} {cc}{p.conf:.1f}%{C.E}"))
        print(box(f" 🛡️ {C.bold('BEST DC:')} {C.bold(p.dc_recommend.split(' - ')[0])}"))
        print(box(""))
        print(f" {C.blue('└'+'─'*w+'┘')}")

    @staticmethod
    def summary(preds: List[Pred]):
        Disp.section("📋 SUMMARY")
        print(f" {'#':>2} {'Home':<16} {'Away':<16} {'Pred':<7} "
              f"{'Sc':>4} {'Conf':>5} {'1X':>5} {'X2':>5} {'12':>5} {'DC Rec'}")
        print(f" {'─'*100}")
        for i, p in enumerate(preds, 1):
            res = {'HOME':C.green('H'),'DRAW':C.yellow('D'),'AWAY':C.red('A')}.get(p.result,'?')
            cc = C.conf_color(p.conf)
            sc = f"{p.pred_sc[0]}-{p.pred_sc[1]}"
            dc_best = p.dc_recommend.split(' (')[0] if p.dc_recommend else ''
            derby = "🔥" if p.is_derby else ""
            print(f" {i:>2} {p.home:<16} {p.away:<16} {res:<7} "
                  f"{sc:>4} {cc}{p.conf:>4.0f}%{C.E} "
                  f"{p.dc_1x*100:>4.0f}% {p.dc_x2*100:>4.0f}% "
                  f"{p.dc_12*100:>4.0f}% {dc_best} {derby}")
        all_v = [(p,v) for p in preds for v in p.value_bets if v['is_value']]
        all_dcv = [(p,v) for p in preds for v in p.dc_value_bets if v['is_value']]
        if all_v:
            print(f"\n {C.bold(C.green('💰 1X2 VALUE BETS:'))}")
            for p, v in all_v:
                edge_val = v['edge']
                print(f" 🔥 {p.home} vs {p.away}: "
                      f"{v['market']} @{v['odds']:.2f} "
                      f"(Edge:{C.green(f'+{edge_val:.1f}%')})")
        if all_dcv:
            print(f"\n {C.bold(C.cyan('🛡️ DC VALUE BETS:'))}")
            for p, v in all_dcv:
                edge_val = v['edge']
                print(f" 🛡️ {p.home} vs {p.away}: "
                      f"{v['market']} @{v['odds']:.2f} "
                      f"(Edge:{C.green(f'+{edge_val:.1f}%')})")


    @staticmethod
    def backtest(r: dict):
        Disp.section("📊 BACKTEST v3.3")
        if 'error' in r:
            Disp.error(r['error'])
            return
        w = 62
        print(f" {C.blue('┌'+'─'*w+'┐')}")
        print(box(f" {C.bold('🔬 PERFORMANCE REPORT v3.3')}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        ra = r['result_acc']
        rac = C.green if ra>50 else (C.yellow if ra>40 else C.red)
        print(box(f" {C.bold('📊 Accuracy:')}"))
        print(box(f" 1X2 Result: {rac(f'{ra:.1f}%')} ({r['correct']}/{r['total']})"))
        print(box(f" Exact Score: {r['score_acc']:.1f}%"))
        bs = r['brier']
        bsc = C.green if bs<0.15 else (C.yellow if bs<0.22 else C.red)
        print(box(f" Brier Score: {bsc(f'{bs:.4f}')}"))
        print(box(f" Calibrated: {C.green('✅') if r['cal_used'] else C.yellow('❌')}"))
        if r.get('ml_acc',0) > 0:
            ml_acc_val = r['ml_acc']
            print(box(f" ML Balanced Acc: {C.green(f'{ml_acc_val:.1f}%')}"))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('🛡️ DOUBLE CHANCE ACCURACY:')}"))
        dc_1x_val = r.get('dc_1x_acc', 0)
        dc_x2_val = r.get('dc_x2_acc', 0)
        dc_12_val = r.get('dc_12_acc', 0)
        print(box(f" 1X (Home/Draw): {C.green(f'{dc_1x_val:.1f}%')} ({r.get('dc_1x_n',0)} matches)"))
        print(box(f" X2 (Away/Draw): {C.yellow(f'{dc_x2_val:.1f}%')} ({r.get('dc_x2_n',0)} matches)"))
        print(box(f" 12 (No Draw): {C.cyan(f'{dc_12_val:.1f}%')} ({r.get('dc_12_n',0)} matches)"))

        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('📈 Confidence Tiers:')}"))
        print(box(f" High (>55%): {r['hi_acc']:5.1f}% ({r['hi_n']})"))
        print(box(f" Med (40-55): {r['me_acc']:5.1f}% ({r['me_n']})"))
        print(box(f" Low (<40%): {r['lo_acc']:5.1f}% ({r['lo_n']})"))
        rt = ("⭐"*5 if ra>=50 else ("⭐"*4 if ra>=45 else ("⭐"*3 if ra>=40 else "⭐⭐")))
        print(f" {C.blue('├'+'─'*w+'┤')}")
        print(box(f" {C.bold('Rating:')} {rt}"))
        print(f" {C.blue('└'+'─'*w+'┘')}")

# ══════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════
def export_json(preds: List[Pred], fn: str = "predictions_v33.json") -> str:
    out = []
    for p in preds:
        e = {
            'home': p.home,
            'away': p.away,
            'date': p.date,
            'prediction': p.result,
            'score': f"{p.pred_sc[0]}-{p.pred_sc[1]}",
            'confidence': round(float(p.conf), 1),
            'calibrated': p.calibrated,
            'derby': p.derby_name if p.is_derby else None,
            'probabilities': {
                'home': round(float(p.hp*100),1),
                'draw': round(float(p.dp*100),1),
                'away': round(float(p.ap*100),1),
            },
            'double_chance': {
                '1X': round(float(p.dc_1x*100),1),
                'X2': round(float(p.dc_x2*100),1),
                '12': round(float(p.dc_12*100),1),
                'recommendation': p.dc_recommend,
            },
            'xg': {'home': round(float(p.hxg),2), 'away': round(float(p.axg),2)},
            'markets': {
                'btts': round(float(p.btts*100),1),
                'o15': round(float(p.o15*100), 1),
                'o25': round(float(p.o25*100), 1),
                'o35': round(float(p.o35*100), 1),
            },
            'momentum': {'home': p.h_momentum, 'away': p.a_momentum},
            'elo': {'home': round(float(p.h_elo)), 'away': round(float(p.a_elo))},
        }
        if p.value_bets:
            e['value_bets'] = [{k:(float(v) if isinstance(v,(int,float)) else v) for k,v in vb.items()} for vb in p.value_bets if vb['is_value']]
        if p.dc_value_bets:
            e['dc_value_bets'] = [{k:(float(v) if isinstance(v,(int,float)) else v) for k,v in vb.items()} for vb in p.dc_value_bets if vb['is_value']]
        out.append(e)
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return fn

# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════
class App:
    def __init__(self, token: str, okey: str = ""):
        self.api = FootballAPI(token)
        self.data = DataProc()
        self.eng: Optional[Engine] = None
        self.ml: Optional[MLPred] = None
        self.odds = OddsAPI(okey)
        self.cal = Calibrator()
        self.bt = Backtester()
        self.sy: Optional[int] = None
        self.raw: List[dict] = []
        self.last: List[Pred] = []
        self._log: List[Tuple[str,str]] = []

    def _log_msg(self, level: str, msg: str):
        self._log.append((level, msg))
        if not STREAMLIT_AVAILABLE:
            {'progress': Disp.progress, 'success': Disp.success, 'error': Disp.error, 'info': Disp.info}.get(level, print)(msg)

    def init(self) -> bool:
        if not STREAMLIT_AVAILABLE:
            Disp.header()
        self._log_msg('progress', "Loading season...")
        self.sy = self.api.season_year()
        if not self.sy:
            self._log_msg('error', "Failed to get season")
            return False
        self._log_msg('success', f"Season: {self.sy}/{self.sy+1}")
        self._log_msg('progress', "Loading matches for last 3 seasons...")
        self.raw = []
        for year in range(self.sy-2, self.sy+1):
            self._log_msg('progress', f"Season {year}/{year+1}...")
            s_matches = self.api.finished(year)
            if s_matches:
                self.raw.extend(s_matches)
        if not self.raw:
            self._log_msg('error', "No matches found")
            return False
        self._log_msg('success', f"{len(self.raw)} matches loaded")
        self._log_msg('progress', "Processing + Elo + Momentum...")
        self.data.process(self.raw)
        if not self.data.teams:
            self._log_msg('error', "No teams")
            return False
        self._log_msg('success', f"{len(self.data.teams)} teams | "
                      f"H:{self.data.avg_h:.2f} A:{self.data.avg_a:.2f}")
        top = sorted(self.data.teams.values(), key=lambda t: t.elo, reverse=True)[:3]
        self._log_msg('success', "Elo: "+", ".join(f"{t.name}({t.elo:.0f})" for t in top))
        hot = [t for t in self.data.teams.values() if t.momentum > 40]
        cold = [t for t in self.data.teams.values() if t.momentum < -40]
        if hot:
            self._log_msg('success', "🔥 Hot: " + ", ".join(t.name for t in hot))
        if cold:
            self._log_msg('info', "📉 Cold: " + ", ".join(t.name for t in cold))
        if ML_AVAILABLE:
            mn = "XGBoost Stacking" if XGBOOST_AVAILABLE else "RF+GBM"
            self._log_msg('progress', f"Training ML Pipeline ({mn}, 58 features)...")
            self.ml = MLPred()
            if self.ml.train(self.data):
                src = "external" if self.ml._external else "trained"
                self._log_msg('success', f"ML {src}: {self.ml.acc*100:.1f}% balanced CV acc")
                if not self.ml._external:
                    self.ml.save_pipeline()
            else:
                self.ml = None
                self._log_msg('info', "ML skipped (needs 40+ matches or sufficient class samples)")
        else:
            self._log_msg('info', "ML disabled (pip install scikit-learn numpy)")
        if self.cal.load():
            self._log_msg('success', "Calibration loaded")
        if self.odds.ok():
            self._log_msg('progress', "Loading odds...")
            od = self.odds.fetch()
            if od:
                self._log_msg('success', f"Odds: {len(od)} matches")
        self.eng = Engine(self.data, self.ml, self.odds, self.cal)
        self._log_msg('success', "Engine v3.3 ready! 🚀")
        return True

    # FIX 7: use DataProc.team_by_name for robust lookup
    def custom(self, hn: str, an: str) -> Optional[Pred]:
        ht = self.data.team_by_name(hn)
        at = self.data.team_by_name(an)
        if not ht or not at:
            if not STREAMLIT_AVAILABLE:
                Disp.error("Team(s) not found")
                self.teams()
            return None
        pr = self.eng.predict(ht.id, at.id, "Custom")
        if pr:
            if not STREAMLIT_AVAILABLE:
                Disp.card(pr, 1)
            self.last = [pr]
        return pr

    def predict(self, days: int = 14) -> List[Pred]:
        if not STREAMLIT_AVAILABLE:
            Disp.section("🔮 PREDICTIONS")
            Disp.progress(f"Upcoming ({days}d)...")
        up = self.api.upcoming(days)
        if not up:
            up = self.api.scheduled(self.sy)
        if not up:
            if not STREAMLIT_AVAILABLE:
                Disp.error("None found")
            return []
        if not STREAMLIT_AVAILABLE:
            Disp.success(f"{len(up)} matches")
        preds = []
        for i, m in enumerate(up, 1):
            hid = m.get('homeTeam', {}).get('id')
            aid = m.get('awayTeam', {}).get('id')
            if not hid or not aid:
                continue
            pr = self.eng.predict(hid, aid, m.get('utcDate', ''))
            if pr:
                preds.append(pr)
                if not STREAMLIT_AVAILABLE:
                    Disp.card(pr, i)
        if preds and not STREAMLIT_AVAILABLE:
            Disp.summary(preds)
        self.last = preds
        return preds

    def backtest(self) -> dict:
        if not STREAMLIT_AVAILABLE:
            Disp.section("🔬 BACKTEST v3.3")
            Disp.progress("Testing with DC tracking...")
        r = self.bt.run(self.raw)
        if not STREAMLIT_AVAILABLE:
            Disp.backtest(r)
        if r.get('cal_used'):
            self.cal = self.bt.cal
            self.cal.save()
            self.eng = Engine(self.data, self.ml, self.odds, self.cal)
            if not STREAMLIT_AVAILABLE:
                Disp.success("Calibration saved & applied!")
        return r

    def standings(self):
        if STREAMLIT_AVAILABLE:
            return sorted(self.data.teams.values(), key=lambda t: t.pos)
        else:
            Disp.standings(self.data.teams)

    def teams(self):
        if STREAMLIT_AVAILABLE:
            return sorted(self.data.teams.values(), key=lambda t: t.pos)
        else:
            Disp.section("📋 TEAMS")
            for t in sorted(self.data.teams.values(), key=lambda t: t.pos):
                mi = "🔥" if t.momentum>40 else ("📉" if t.momentum<-40 else "")
                print(f" #{t.pos:<3} {t.name:<25} Elo:{t.elo:.0f} "
                      f"{C.form_str(t.form_string[-5:])} {mi}")

    def interactive(self):
        if STREAMLIT_AVAILABLE:
            return
        while True:
            print(f"\n {C.cyan(C.bold('═══ MENU v3.3 ═══'))}")
            for n,e,l in [(1,'🔮','Predict'),(2,'⚽','Custom Match'),
                          (3,'📊','Standings'),(4,'🔬','Backtest+Calibrate'),
                          (5,'📋','Teams'),(6,'💾','Export JSON'),(7,'🚪','Exit')]:
                print(f" {n}. {e} {l}")
            try:
                ch = input(C.cyan("\n (1-7): ")).strip()
            except (KeyboardInterrupt, EOFError):
                break
            if ch == '1':
                try:
                    d = int(input(C.cyan(" Days(14): ")).strip() or "14")
                except:
                    d = 14
                self.predict(d)
            elif ch == '2':
                try:
                    h = input(C.cyan(" Home: ")).strip()
                    a = input(C.cyan(" Away: ")).strip()
                    if h and a:
                        self.custom(h, a)
                except (KeyboardInterrupt, EOFError):
                    pass
            elif ch == '3':
                self.standings()
            elif ch == '4':
                self.backtest()
            elif ch == '5':
                self.teams()
            elif ch == '6':
                if self.last:
                    fn = export_json(self.last)
                    Disp.success(f"Exported → {fn}")
                else:
                    Disp.info("Predict first")
            elif ch == '7':
                print(C.green(C.bold("\n 👋 Good luck! ⚽\n")))
                break

# ══════════════════════════════════════════════════════════════
# STREAMLIT UI v3.3
# FIX 8: robust session-state management
# ══════════════════════════════════════════════════════════════
def run_streamlit():
    st.set_page_config(
        page_title="PL Predictor Pro v3.3",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
    .main-header{text-align:center;font-size:2.5rem;font-weight:bold;color:#00ff9d;}
    .metric-card{background:#1e1e2e;border-radius:10px;padding:15px;border-left:4px solid #00ff9d;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='main-header'>⚽ Premier League Predictor v3.3</h1>", unsafe_allow_html=True)
    st.caption("Dixon-Coles • Elo • XGBoost Stacking (58 features, leak-free Pipeline) • "
               "Double Chance • Momentum • Calibration")

    # ── Sidebar ────────────────────────────────────────────────
    st.sidebar.title("⚙️ Settings")
    fb_key = st.sidebar.text_input("🔑 football-data.org key", value=FOOTBALL_DATA_KEY, type="password")
    odds_key = st.sidebar.text_input("🔑 Odds API key (optional)", value=ODDS_API_KEY, type="password")

    # FIX 8: track the key used during last init to detect changes
    prev_key = st.session_state.get('_init_key', '')
    key_changed = (fb_key != prev_key) and prev_key != ''
    if key_changed:
        # Clear stale state when key changes
        for k in ['app','initialized','last_preds','custom_pr','backtest_results']:
            st.session_state.pop(k, None)
        st.warning("API key changed — please re-initialize.")

    init_btn = st.sidebar.button("🚀 Initialize / Reload")
    if init_btn and fb_key:
        with st.spinner("Loading data & training ML pipeline..."):
            app = App(fb_key, odds_key)
            ok = app.init()
            if ok:
                st.session_state['app'] = app
                st.session_state['initialized'] = True
                st.session_state['_init_key'] = fb_key
                # Clear old prediction cache on re-init
                for k in ['last_preds','custom_pr','backtest_results']:
                    st.session_state.pop(k, None)
                for level, msg in app._log:
                    {'success': st.success, 'error': st.error, 'info': st.info}.get(level, st.write)(msg)
            else:
                st.error("❌ Initialization failed")

    if 'initialized' not in st.session_state:
        st.info("👈 Enter your API key and click 'Initialize / Reload'")
        with st.expander("📊 v3.3 Improvements over v3.2"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔧 Bug Fixes")
                st.markdown("""
                - ✅ **FIX 1:** `norm_name` — safe TEAMS_MAP lookup only
                - ✅ **FIX 2:** Leak-free sklearn Pipeline (imputer→scaler→model)
                - ✅ **FIX 3:** `class_weight='balanced'` + `balanced_accuracy`
                - ✅ **FIX 4:** `SimpleImputer(mean)` replaces `nan_to_num(0)`
                - ✅ **FIX 5:** `StratifiedKFold` + min-samples guard
                - ✅ **FIX 6:** External model must be a full Pipeline
                - ✅ **FIX 7:** Promoted teams handled via `team_by_name()`
                - ✅ **FIX 8:** Streamlit state clears on key change
                """)
            with col2:
                st.markdown("### 🧠 ML Architecture (58 features)")
                st.markdown("""
                - Elo ratings (3) + Form scores (4)
                - Goal/Defense averages (12)
                - Attack/Defense ratios (4)
                - Win rates (4) + Position/Points (7)
                - Clean sheets (4) + Fatigue (2)
                - Draw rates (7) + Volatility (2)
                - Derby indicator (1) + Elo scaled (1)
                - **Momentum v3.1 (7)**
                """)
        st.stop()

    app = st.session_state['app']

    mode = st.sidebar.radio("📌 Function", [
        "🔮 Predict Upcoming",
        "⚽ Custom Match",
        "📊 Standings",
        "🔬 Backtest",
        "💾 Export"
    ])

    # ── Standings ──────────────────────────────────────────────
    if mode == "📊 Standings":
        st.subheader("🏆 Premier League Standings")
        rows = []
        for t in app.standings():
            mi = "🔥" if t.momentum>40 else ("📉" if t.momentum<-40 else "")
            rows.append({
                "#": t.pos,
                "Team": t.name,
                "P": t.played,
                "W": t.wins,
                "D": t.draws,
                "L": t.losses,
                "GF": t.gf,
                "GA": t.ga,
                "GD": t.gd,
                "Pts": t.pts,
                "Elo": f"{t.elo:.0f}",
                "Form": t.form_string[-5:],
                "Mom": f"{t.momentum:+d} {mi}",
                "Tier": t.elo_tier
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # ── Predict Upcoming ───────────────────────────────────────
    elif mode == "🔮 Predict Upcoming":
        days = st.slider("Days ahead", 1, 30, 14)
        if st.button("🔍 Predict"):
            with st.spinner("Fetching & predicting..."):
                preds = app.predict(days)
                st.session_state['last_preds'] = preds
            if not preds:
                st.warning("No upcoming matches found")
            else:
                st.success(f"Found {len(preds)} matches")
        if st.session_state.get('last_preds'):
            for i, pr in enumerate(st.session_state['last_preds']):
                derby_tag = f" 🔥 {pr.derby_name}" if pr.is_derby else ""
                with st.expander(f"⚽ {pr.home} vs {pr.away}{derby_tag}"):
                    _render_prediction_st(pr)

    # ── Custom Match ───────────────────────────────────────────
    elif mode == "⚽ Custom Match":
        teams_list = sorted(t.name for t in app.data.teams.values())
        c1, c2 = st.columns(2)
        home = c1.selectbox("🏠 Home Team", teams_list)
        away = c2.selectbox("✈️ Away Team", teams_list, index=min(1, len(teams_list)-1))
        if home == away:
            st.warning("Select two different teams")
        else:
            if st.button("🔮 Predict"):
                pr = app.custom(home, away)
                st.session_state['custom_pr'] = pr
                if not pr:
                    st.error("Prediction failed")
        if st.session_state.get('custom_pr'):
            _render_prediction_st(st.session_state['custom_pr'])

    # ── Backtest ───────────────────────────────────────────────
    elif mode == "🔬 Backtest":
        if st.button("▶️ Run Backtest"):
            with st.spinner("Backtesting (58-feature Pipeline, DC tracking)..."):
                r = app.backtest()
                st.session_state['backtest_results'] = r
        r = st.session_state.get('backtest_results')
        if r:
            if 'error' in r:
                st.error(r['error'])
            else:
                st.subheader("📈 Backtest Results v3.3")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("🎯 1X2 Accuracy", f"{r['result_acc']:.1f}%")
                c2.metric("⚽ Exact Score", f"{r['score_acc']:.1f}%")
                c3.metric("📊 Brier Score", f"{r['brier']:.4f}")
                c4.metric("🤖 ML Bal. Acc", f"{r.get('ml_acc',0):.1f}%")
                st.markdown("### 🛡️ Double Chance Accuracy")
                c1,c2,c3 = st.columns(3)
                c1.metric("1X (Home/Draw)", f"{r.get('dc_1x_acc',0):.1f}%", f"{r.get('dc_1x_n',0)} matches")
                c2.metric("X2 (Away/Draw)", f"{r.get('dc_x2_acc',0):.1f}%", f"{r.get('dc_x2_n',0)} matches")
                c3.metric("12 (No Draw)", f"{r.get('dc_12_acc',0):.1f}%", f"{r.get('dc_12_n',0)} matches")
                st.markdown("### 📈 Confidence Tiers")
                c1,c2,c3 = st.columns(3)
                c1.metric(f"High >55% ({r['hi_n']})", f"{r['hi_acc']:.1f}%")
                c2.metric(f"Med 40-55% ({r['me_n']})", f"{r['me_acc']:.1f}%")
                c3.metric(f"Low <40% ({r['lo_n']})", f"{r['lo_acc']:.1f}%")
                ra = r['result_acc']
                stars = "⭐"*(5 if ra>=50 else 4 if ra>=45 else 3 if ra>=40 else 2)
                st.markdown(f"### Rating: {stars}")

    # ── Export ─────────────────────────────────────────────────
    elif mode == "💾 Export":
        preds = st.session_state.get('last_preds') or app.last
        if preds:
            fn = export_json(preds)
            with open(fn, 'rb') as f:
                st.download_button("📥 Download JSON", f, file_name=fn)
        else:
            st.warning("No predictions yet — run 'Predict Upcoming' first.")

# ══════════════════════════════════════════════════════════════
# STREAMLIT PREDICTION CARD RENDERER
# ══════════════════════════════════════════════════════════════
def _render_prediction_st(pr: Pred):
    if pr.date and pr.date != "Custom":
        dt = parse_date(pr.date)
        if dt:
            st.caption(f"📅 {dt.strftime('%a %d %b %Y • %H:%M')}")
    c1,c2,c3 = st.columns(3)
    c1.metric("🏠 Home", f"{pr.hp*100:.1f}%")
    c2.metric("🤝 Draw", f"{pr.dp*100:.1f}%")
    c3.metric("✈️ Away", f"{pr.ap*100:.1f}%")
    # Clamp to [0,1] to prevent st.progress errors
    st.progress(min(1.0,max(0.0,float(pr.hp))), text=f"Home {pr.hp*100:.1f}%")
    st.progress(min(1.0,max(0.0,float(pr.dp))), text=f"Draw {pr.dp*100:.1f}%")
    st.progress(min(1.0,max(0.0,float(pr.ap))), text=f"Away {pr.ap*100:.1f}%")
    st.markdown("#### 🛡️ Double Chance")
    c1,c2,c3 = st.columns(3)
    c1.metric("1X (Home/Draw)", f"{pr.dc_1x*100:.1f}%")
    c2.metric("X2 (Away/Draw)", f"{pr.dc_x2*100:.1f}%")
    c3.metric("12 (No Draw)", f"{pr.dc_12*100:.1f}%")
    st.info(f"💡 **Recommendation:** {pr.dc_recommend}")
    c1,c2 = st.columns(2)
    c1.metric("⚡ xG Home", f"{pr.hxg:.2f}")
    c2.metric("⚡ xG Away", f"{pr.axg:.2f}")
    st.markdown(f"**🎯 Most Likely Score:** "
                f"**{pr.pred_sc[0]} - {pr.pred_sc[1]}** "
                f"(Confidence: {pr.conf:.1f}%)")
    if pr.top_sc:
        st.caption("📊 Top scores: " + " | ".join(f"{h}-{a} ({p*100:.1f}%)" for h,a,p in pr.top_sc[:5]))
    if pr.h_momentum != 0 or pr.a_momentum != 0:
        st.markdown("#### 💪 Momentum")
        c1,c2 = st.columns(2)
        h_icon = "🔥" if pr.h_momentum>40 else ("📉" if pr.h_momentum<-40 else "→")
        a_icon = "🔥" if pr.a_momentum>40 else ("📉" if pr.a_momentum<-40 else "→")
        c1.metric(f"{h_icon} {pr.home}", f"{pr.h_momentum:+d}")
        c2.metric(f"{a_icon} {pr.away}", f"{pr.a_momentum:+d}")
    st.markdown("#### 📈 Markets")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Over 1.5", f"{pr.o15*100:.0f}%")
    c2.metric("Over 2.5", f"{pr.o25*100:.0f}%")
    c3.metric("Over 3.5", f"{pr.o35*100:.0f}%")
    c4.metric("BTTS", f"{pr.btts*100:.0f}%")
    st.markdown("#### 📋 Form & Fatigue")
    c1,c2 = st.columns(2)
    c1.markdown(f"**{pr.home}:** `{pr.h_form}` (Fat:{pr.h_fat:.0f}, Rest:{pr.h_rest}d)")
    c2.markdown(f"**{pr.away}:** `{pr.a_form}` (Fat:{pr.a_fat:.0f}, Rest:{pr.a_rest}d)")
    c1,c2 = st.columns(2)
    c1.metric(f"🏆 {pr.home} Elo", f"{pr.h_elo:.0f}")
    c2.metric(f"🏆 {pr.away} Elo", f"{pr.a_elo:.0f}")
    with st.expander("🔬 Model Breakdown"):
        for nm,(mh,md,ma) in pr.models.items():
            wt = WEIGHTS.get(nm,0)
            st.markdown(f"**{nm}** ({wt:.0%}): "
                        f"H={mh*100:.1f}% | D={md*100:.1f}% | A={ma*100:.1f}%")
    if pr.value_bets:
        vf = [v for v in pr.value_bets if v['is_value']]
        if vf:
            st.markdown("#### 💰 1X2 Value Bets")
            for v in vf:
                st.success(f"**{v['market']}** @{v['odds']:.2f} | "
                           f"Edge: +{v['edge']:.1f}% | Kelly: {v['kelly']:.1f}%")
    if pr.dc_value_bets:
        vf = [v for v in pr.dc_value_bets if v['is_value']]
        if vf:
            st.markdown("#### 🛡️ DC Value Bets")
            for v in vf:
                st.success(f"**{v['market']}** @{v['odds']:.2f} | "
                           f"Edge: +{v['edge']:.1f}% | Kelly: {v['kelly']:.1f}%")
    rm = {'HOME': f"🏆 {pr.home} Win", 'DRAW': "🤝 Draw", 'AWAY': f"🏆 {pr.away} Win"}
    em = "🔥" if pr.conf>55 else ("⚡" if pr.conf>40 else "⚠️")
    st.markdown("---")
    st.markdown(f"### {em} **PREDICTION:** {rm.get(pr.result,'?')} "
                f"({pr.conf:.1f}% confidence)")
    st.markdown(f"### 🛡️ **Best DC:** {pr.dc_recommend.split(' - ')[0]}")
    if pr.calibrated:
        st.caption("✅ Calibrated probabilities")
    if pr.ml_used:
        st.caption(f"🤖 ML pipeline used (Bal.CV: {pr.ml_acc*100:.1f}%)")

# ══════════════════════════════════════════════════════════════
# CLI MAIN
# ══════════════════════════════════════════════════════════════
def cli_main():
    tok = FOOTBALL_DATA_KEY or os.environ.get("FOOTBALL_DATA_KEY", "")
    okey = ODDS_API_KEY or os.environ.get("ODDS_API_KEY", "")
    if not tok:
        Disp.header()
        try:
            tok = input(C.cyan(" 🔑 football-data.org key: ")).strip()
        except (KeyboardInterrupt, EOFError):
            return
        if not tok:
            Disp.error("No key provided")
            return
    app = App(tok, okey)
    if not app.init():
        return
    app.standings()
    try:
        mode = input(C.cyan("\n (1) Auto (2) Interactive: ")).strip()
    except (KeyboardInterrupt, EOFError):
        return
    if mode == '2':
        app.interactive()
    else:
        preds = app.predict()
        try:
            if input(C.cyan(" Backtest? (y/n): ")).strip().lower() == 'y':
                app.backtest()
        except (KeyboardInterrupt, EOFError):
            pass
        if preds:
            try:
                if input(C.cyan(" Export? (y/n): ")).strip().lower() == 'y':
                    fn = export_json(preds)
                    Disp.success(f"Exported → {fn}")
            except (KeyboardInterrupt, EOFError):
                pass
    print(C.green(C.bold("\n ✅ Done! ⚽\n")))

# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════
if STREAMLIT_AVAILABLE:
    run_streamlit()
elif __name__ == "__main__":
    cli_main()
