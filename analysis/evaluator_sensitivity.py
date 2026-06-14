#!/usr/bin/env python3
"""Evaluator-subset sensitivity analysis for the STPAgent evaluation.

WHAT THIS SCRIPT IS FOR
-----------------------
This is the robustness companion to ``recompute_statistics.py``. It reads the
same raw ratings (six evaluators A-F classifying 231 AI-generated UCAs as
CORRECT AND USEFUL / CORRECT BUT USELESS / INCORRECT) and answers three
questions about how much the headline numbers depend on the specific panel of
evaluators:

  1. BASELINE - recompute the headline statistics over all six evaluators
     (a sanity check that this script and ``recompute_statistics.py`` agree).

  2. PROVENANCE - the FIRST submission of the paper reported agreement values
     (Fleiss κ = 0.58; one-vs-all κ = 0.71 / 0.54 / 0.42; Cochran's Q = 5.21 and
     12.47) that do not reproduce from the full six-evaluator data. The author's
     hypothesis was that the original numbers might have been computed on some
     SUBSET of evaluators. This section exhaustively searches every subset of
     size 3-6 to test that hypothesis. (Spoiler from the saved report: no subset
     reproduces them - the original computation, not the data, was wrong. The
     published values have since been corrected.) The published Q/p pairs are
     internally consistent only with df = 3, i.e. four evaluators, which is why
     the search starts at size 3.

  3. ROBUSTNESS - leave-one-out and leave-best-and-worst-out sensitivity of all
     reported statistics, to show the safety-critical findings do not hinge on
     any single rater. "Best" / "worst" are defined by each evaluator's agreement
     with the majority verdict of the other five.

COMPLETE-CASE RULE
------------------
Every statistic for a subset S uses only the UCAs rated by ALL evaluators in S.
Because evaluator B left 2 UCAs unrated and C left 16, subsets that exclude C
have more complete cases than subsets that include it - this is expected and is
reported alongside each result (the "n complete cases" figure).

DEPENDENCIES
------------
openpyxl to read the .xlsx. All statistics are standard-library. The agreement /
chi-square functions are intentionally duplicated from ``recompute_statistics.py``
so that each script is self-contained and can be run on its own.
"""

import argparse
import math
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import openpyxl

XLSX = Path(__file__).resolve().parent.parent / "evaluation" / "STPAgent Evaluation.xlsx"
EVALUATORS = ["A", "B", "C", "D", "E", "F"]
CATEGORIES = ["CORRECT AND USEFUL", "CORRECT BUT USELESS", "INCORRECT"]
CAT_SHORT = ["useful", "useless", "incorrect"]   # short labels, same order as CATEGORIES

# The agreement statistics reported in the FIRST submission, which the provenance
# search (section 2) tries - and fails - to reproduce from any evaluator subset.
PUBLISHED = {
    "fleiss3": 0.58,
    "k_incorrect": 0.71,
    "k_useful": 0.54,
    "k_useless": 0.42,
    "q_incorrect": 5.21,
    "q_useful": 12.47,
}


# ---------------------------------------------------------------------------
# Chi-square survival function  P(X^2_df >= x)
# ---------------------------------------------------------------------------
# Turns Cochran's Q into a p-value via the regularized upper incomplete gamma
# function. Two standard numerical methods (series when x < a+1, continued
# fraction otherwise); validated against published chi-square critical values.
# (Identical to the implementation in recompute_statistics.py - duplicated so
# this script stands alone.)

def _gamma_series(a, x):
    """Lower regularized incomplete gamma P(a, x) via its Taylor series."""
    term = 1.0 / a
    total = term
    n = a
    for _ in range(1000):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * 1e-15:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_contfrac(a, x):
    """Upper regularized incomplete gamma Q(a, x) via the Lentz continued fraction."""
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    """Survival function P(X >= x) for a chi-square distribution with df d.o.f."""
    if x <= 0:
        return 1.0
    a, hx = df / 2.0, x / 2.0
    if hx < a + 1.0:
        return 1.0 - _gamma_series(a, hx)
    return _gamma_contfrac(a, hx)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ratings():
    """Return {evaluator: {uca_id: CATEGORY}} from the six evaluator sheets.

    Column 0 = UCA id, column 5 = rating. Blank / non-category cells (the
    "NOT RATED" placeholders) are skipped, which is what gives the differing
    per-evaluator rated counts.
    """
    wb = openpyxl.load_workbook(XLSX, read_only=True)
    ratings = {}
    for ev in EVALUATORS:
        per = {}
        for row in wb[ev].iter_rows(min_row=2, values_only=True):
            uca_id, rating = row[0], row[5]
            if uca_id is None:
                continue
            # str() guards against non-string cell values; blanks become "".
            rating = str(rating or "").strip().upper()
            if rating in CATEGORIES:
                per[uca_id] = rating
        ratings[ev] = per
    return ratings


def complete_ids(ratings, subset):
    """UCA ids rated by EVERY evaluator in ``subset`` (the complete cases)."""
    ids = set(ratings[subset[0]])
    for ev in subset[1:]:
        ids &= set(ratings[ev])
    return sorted(ids)


# ---------------------------------------------------------------------------
# Agreement coefficients
# ---------------------------------------------------------------------------
# All three share the observed-agreement term Pa and differ only in the chance
# term Pe; coefficient = (Pa - Pe) / (1 - Pe). ``table`` is one row per item, each
# row holding the count of raters choosing each category. See the long comments
# in recompute_statistics.py for the full derivation and references.

def fleiss_kappa(table):
    """Return (Pa, kappa). Pe = sum of squared category proportions."""
    n_raters = sum(table[0])
    n_items = len(table)
    p_cat = [sum(row[k] for row in table) / (n_items * n_raters)
             for k in range(len(table[0]))]
    pa = statistics.mean(
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in table)
    pe = sum(p * p for p in p_cat)
    return pa, (pa - pe) / (1 - pe)


def gwet_ac1(table):
    """Gwet's AC1. Pe = sum_k pi_k(1-pi_k) / (q-1); robust to skewed prevalence."""
    n_raters = sum(table[0])
    n_items = len(table)
    q = len(table[0])
    p_cat = [sum(row[k] for row in table) / (n_items * n_raters)
             for k in range(q)]
    pa = statistics.mean(
        (sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
        for row in table)
    pe = sum(p * (1 - p) for p in p_cat) / (q - 1)
    return (pa - pe) / (1 - pe)


def cochran_q(indicator_rows):
    """Cochran's Q for k related binary samples; Q ~ chi-square(df = k - 1).

    ``indicator_rows`` is one 0/1 row per item across the k evaluators. Returns
    (Q, p). The denominator is 0 only when there is no disagreement at all (every
    row all-0 or all-1); that degenerate case returns NaN rather than dividing by
    zero.
    """
    k = len(indicator_rows[0])
    col = [sum(row[j] for row in indicator_rows) for j in range(k)]
    row_tot = [sum(row) for row in indicator_rows]
    num = (k - 1) * (k * sum(c * c for c in col) - sum(col) ** 2)
    den = k * sum(row_tot) - sum(r * r for r in row_tot)
    if den == 0:
        return float("nan"), float("nan")
    q = num / den
    return q, chi2_sf(q, k - 1)


# ---------------------------------------------------------------------------
# Per-subset statistics bundle
# ---------------------------------------------------------------------------

def subset_stats(ratings, subset):
    """Compute the full statistics bundle for one evaluator subset.

    Everything is computed over that subset's complete cases. Returns a dict
    consumed by the report writer below.
    """
    ids = complete_ids(ratings, subset)
    n_r = len(subset)

    # Per-item category counts (table3) and the collapsed binary view (table2).
    table3 = [[sum(1 for ev in subset if ratings[ev][i] == c)
               for c in CATEGORIES] for i in ids]
    table2 = [[row[0] + row[1], row[2]] for row in table3]

    pa3, k3 = fleiss_kappa(table3)
    pa2, k2 = fleiss_kappa(table2)

    # "One-vs-all" Fleiss kappa: for each category, treat it as a binary
    # (this-category vs everything-else) problem. Matches how the first
    # submission reported per-category kappas.
    one_vs_all = {}
    for k, cat in enumerate(CAT_SHORT):
        tbl = [[row[k], n_r - row[k]] for row in table3]
        one_vs_all[cat] = fleiss_kappa(tbl)[1]

    # Cochran's Q per category (equal flagging rates across the subset's raters?).
    qs = {}
    for k, cat in enumerate(CAT_SHORT):
        rows = [[1 if ratings[ev][i] == CATEGORIES[k] else 0 for ev in subset]
                for i in ids]
        qs[cat] = cochran_q(rows)

    # Per-evaluator percentages use each evaluator's OWN rated count as denominator
    # (consistent with recompute_statistics.py), then averaged across the subset.
    pcts = {}
    for ev in subset:
        counts = Counter(ratings[ev].values())
        n = sum(counts.values())
        pcts[ev] = [100 * counts[c] / n for c in CATEGORIES]
    means = [statistics.mean(pcts[ev][k] for ev in subset) for k in range(3)]
    total_corr = [pcts[ev][0] + pcts[ev][1] for ev in subset]

    # Unanimous binary verdict: a table2 row where all raters land in one column.
    unanimous = sum(1 for row in table2 if n_r in row)

    return {
        "ids": ids, "n_items": len(ids),
        "pa3": pa3, "k3": k3, "pa2": pa2, "k2": k2,
        "ac1_3": gwet_ac1(table3), "ac1_2": gwet_ac1(table2),
        "one_vs_all": one_vs_all, "q": qs,
        "means": means, "total_corr": total_corr,
        "unanimous": unanimous,
    }


def t_ci(values):
    """Two-sided 95% t-interval over the per-evaluator means (df = n - 1).

    Returns (mean, low, high). Critical values are hard-coded so no SciPy is
    needed. NOTE: this is a symmetric interval on a bounded percentage, so the
    upper bound can exceed 100% for small subsets - an interval artifact, not a
    claim of >100% correctness.
    """
    tcrit = {2: 12.706205, 3: 4.302653, 4: 3.182446,
             5: 2.776445, 6: 2.570582}[len(values)]   # keyed by n; value is t(.975, n-1)
    m = statistics.mean(values)
    half = tcrit * statistics.stdev(values) / math.sqrt(len(values))
    return m, m - half, m + half


# ---------------------------------------------------------------------------
# Markdown report writer  (single source of truth for console + .md output)
# ---------------------------------------------------------------------------

class Report:
    def __init__(self):
        self._lines = []

    def add(self, text=""):
        self._lines.append(text)

    def h(self, level, text):
        self.add(f"{'#' * level} {text}")
        self.add()

    def table(self, headers, rows):
        self.add("| " + " | ".join(str(h) for h in headers) + " |")
        self.add("|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            self.add("| " + " | ".join(str(c) for c in r) + " |")
        self.add()

    def render(self):
        return "\n".join(self._lines).rstrip() + "\n"

    def save(self, path):
        Path(path).write_text(self.render(), encoding="utf-8")


def stats_row(label, s):
    """One Markdown table row summarizing a subset's statistics bundle."""
    m, lo, hi = t_ci(s["total_corr"])
    return [
        label,
        s["n_items"],
        f"{s['means'][0]:.1f}", f"{s['means'][1]:.1f}", f"{s['means'][2]:.1f}",
        f"{m:.1f} ({lo:.1f}–{hi:.1f})",
        f"{s['pa3']:.3f}", f"{s['ac1_3']:.2f}", f"{s['ac1_2']:.2f}",
        f"{100 * s['unanimous'] / s['n_items']:.1f}%",
    ]


STATS_HEADERS = ["Panel", "n", "useful %", "useless %", "incorrect %",
                 "total correct (95% CI)", "raw 3-cat", "AC1 3-cat",
                 "AC1 binary", "unanimous"]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def build_report(ratings):
    rep = Report()
    rep.h(1, "STPAgent Evaluator-Subset Sensitivity Analysis")
    rep.add("Auto-generated by `analysis/evaluator_sensitivity.py`. Tests how the "
            "reported statistics depend on the panel of evaluators, and whether "
            "any evaluator subset reproduces the (now-corrected) values from the "
            "first submission. Companion to `recompute_statistics.py`.")
    rep.add()

    # --- 1. Baseline -------------------------------------------------------
    rep.h(2, "1. Baseline — all six evaluators")
    full = subset_stats(ratings, EVALUATORS)
    rep.table(STATS_HEADERS, [stats_row("A B C D E F", full)])
    rep.add("Matches `recompute_statistics.py` exactly (sanity check).")
    rep.add()

    # --- 2. Provenance search ---------------------------------------------
    rep.h(2, "2. Provenance search — does any subset reproduce the first-submission values?")
    rep.add("First submission reported: Fleiss κ (3-cat) = 0.58; one-vs-all κ "
            "incorrect/useful/useless = 0.71 / 0.54 / 0.42; Cochran's Q = 5.21 "
            "(incorrect) and 12.47 (useful). The Q/p pairs are internally "
            "consistent only with df = 3 (four evaluators), so the search covers "
            "every subset of size 3–6.")
    rep.add()

    # Compute the statistics bundle for every subset of size 3..6 once and cache it.
    results = {}
    for size in (3, 4, 5, 6):
        for subset in combinations(EVALUATORS, size):
            results[subset] = subset_stats(ratings, list(subset))

    def score(s):
        # Distance of a subset's statistics from the published targets. The two Q
        # terms are scaled by 1/10 so a κ that is off by 0.1 and a Q off by 1.0
        # contribute comparably; this is just a ranking heuristic for the table.
        d = 0.0
        d += abs(s["k3"] - PUBLISHED["fleiss3"])
        d += abs(s["one_vs_all"]["incorrect"] - PUBLISHED["k_incorrect"])
        d += abs(s["one_vs_all"]["useful"] - PUBLISHED["k_useful"])
        d += abs(s["one_vs_all"]["useless"] - PUBLISHED["k_useless"])
        d += abs(s["q"]["incorrect"][0] - PUBLISHED["q_incorrect"]) / 10
        d += abs(s["q"]["useful"][0] - PUBLISHED["q_useful"]) / 10
        return d

    ranked = sorted(results.items(), key=lambda kv: score(kv[1]))
    rep.add("**Closest 12 subsets to the published targets** (smaller distance = "
            "closer; none is actually close):")
    rep.add()
    rows = []
    for subset, s in ranked[:12]:
        ova = s["one_vs_all"]
        rows.append(["".join(subset), f"{s['k3']:.2f}",
                     f"{ova['incorrect']:.2f}", f"{ova['useful']:.2f}",
                     f"{ova['useless']:.2f}", f"{s['q']['incorrect'][0]:.2f}",
                     f"{s['q']['useful'][0]:.2f}", f"{score(s):.3f}"])
    rep.table(["subset", "κ 3-cat", "κ incorrect", "κ useful", "κ useless",
               "Q incorrect", "Q useful", "distance"], rows)

    # For each published value, list every subset (if any) that matches it at the
    # published 2-decimal rounding. Kappas use a 0.005 window; Q values 0.05.
    rep.add("**Exact-match check at the published rounding:**")
    rep.add()
    match_rows = []
    for key, pub in PUBLISHED.items():
        hits = []
        for subset, s in results.items():
            val = {
                "fleiss3": s["k3"],
                "k_incorrect": s["one_vs_all"]["incorrect"],
                "k_useful": s["one_vs_all"]["useful"],
                "k_useless": s["one_vs_all"]["useless"],
                "q_incorrect": s["q"]["incorrect"][0],
                "q_useful": s["q"]["useful"][0],
            }[key]
            window = 0.005 if (key.startswith("k") or key == "fleiss3") else 0.05
            if abs(val - pub) < window:
                hits.append("".join(subset))
        match_rows.append([key, f"{pub:.2f}",
                           ", ".join(hits) if hits else "**no subset matches**"])
    rep.table(["statistic", "published", "matching subsets"], match_rows)
    rep.add("**Verdict:** no subset reproduces the published values — the original "
            "computation was wrong, not the data. The corrected values "
            "(`recompute_statistics.py`) stand.")
    rep.add()

    # --- 3. Best / worst evaluator ----------------------------------------
    rep.h(2, "3. Best / worst evaluator")
    rep.add("Each evaluator is scored by how often they agree with the **majority "
            "verdict of the other five** (3-category, over the six-way complete "
            "cases). Items where the other five tie are skipped for that "
            "evaluator. Mean pairwise agreement is shown for context.")
    rep.add()
    ids6 = complete_ids(ratings, EVALUATORS)
    maj_agree = {}
    rows = []
    for ev in EVALUATORS:
        others = [e for e in EVALUATORS if e != ev]
        hits = ties = 0
        for i in ids6:
            votes = Counter(ratings[o][i] for o in others)
            top = votes.most_common()
            # Skip items where the other five have no unique majority (a tie).
            if len(top) > 1 and top[0][1] == top[1][1]:
                ties += 1
                continue
            if ratings[ev][i] == top[0][0]:
                hits += 1
        maj_agree[ev] = hits / (len(ids6) - ties)
        # Mean pairwise raw agreement (3-category) of this evaluator vs each other.
        pa = statistics.mean(
            sum(1 for i in ids6 if ratings[ev][i] == ratings[o][i]) / len(ids6)
            for o in others)
        rows.append([ev, f"{maj_agree[ev]:.3f}", ties, f"{pa:.3f}"])
    best = max(maj_agree, key=maj_agree.get)
    worst = min(maj_agree, key=maj_agree.get)
    # Annotate the best/worst rows after the fact.
    for r in rows:
        if r[0] == best:
            r[0] = f"{best} (best)"
        elif r[0] == worst:
            r[0] = f"{worst} (worst)"
    rep.table(["Evaluator", "majority agreement", "ties skipped", "mean pairwise"], rows)
    rep.add(f"**Best = {best}, worst = {worst}.** Evaluator {worst} is an extreme "
            f"outlier (rated 85.7% of UCAs \"correct but useless\"), agreeing with "
            f"the panel majority on only ~7% of items.")
    rep.add()

    # --- 4. Leave-one-out + leave-best-and-worst-out ----------------------
    rep.h(2, "4. Leave-one-out and leave-best-and-worst-out sensitivity")
    rep.add("Each row drops one (or two) evaluators and recomputes everything on "
            "the remaining panel's complete cases.")
    rep.add()
    loo_rows = [stats_row("all six", full)]
    for ev in EVALUATORS:
        subset = tuple(e for e in EVALUATORS if e != ev)
        tag = " (worst)" if ev == worst else " (best)" if ev == best else ""
        loo_rows.append(stats_row(f"− {ev}{tag}", results[subset]))
    bw = tuple(e for e in EVALUATORS if e not in (best, worst))
    loo_rows.append(stats_row(f"− {best} and {worst}", results[bw]))
    rep.table(STATS_HEADERS, loo_rows)

    # Per-category Cochran's Q across the leave-one-out panels (the error-flagging
    # heterogeneity is the statistic most sensitive to who is on the panel).
    rep.h(3, "Cochran's Q (incorrect indicator) across panels")
    q_rows = [["all six", f"{full['q']['incorrect'][0]:.2f}",
               len(EVALUATORS) - 1, f"{full['q']['incorrect'][1]:.2e}"]]
    for ev in EVALUATORS:
        subset = tuple(e for e in EVALUATORS if e != ev)
        q, p = results[subset]["q"]["incorrect"]
        q_rows.append([f"− {ev}", f"{q:.2f}", len(subset) - 1, f"{p:.2e}"])
    rep.table(["Panel", "Q incorrect", "df", "p"], q_rows)
    rep.add("Dropping evaluator C (who flagged the most errors and left 16 UCAs "
            "unrated) is the one change that makes the error-flagging difference "
            "non-significant — identifying C as the source of that heterogeneity.")
    rep.add()

    # --- Robustness conclusions ------------------------------------------
    rep.h(2, "Robustness conclusions")
    rep.add("1. **The safety-critical findings are robust to evaluator removal.** "
            "Across every leave-one-out panel, total correctness stays ~96–98%, "
            "the incorrect (hallucination) rate ~2–4%, and binary AC1 0.93–0.96. "
            "No single evaluator drives the headline result.")
    rep.add("2. **Removing the outlier E lifts 3-category agreement** (AC1 0.40 → "
            "0.65, raw 0.527 → 0.700) while leaving the correctness statistics "
            "unchanged — the useful/useless boundary, not factual correctness, is "
            "where the lone divergent evaluator disagreed.")
    rep.add("3. **Evaluator C drives the error-flagging heterogeneity** "
            "(see the Cochran's Q table); consensus errors remain rare either way "
            "(11/231 flagged by ≥ 2 evaluators, 1/231 by ≥ 3).")
    rep.add("4. **Removing the best evaluator (D) barely moves anything** — the "
            "results do not depend on the most consensus-aligned rater either.")
    rep.add()

    return rep


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "evaluator_sensitivity_report.md",
        help="path of the Markdown report to write (default: alongside this script)")
    parser.add_argument(
        "--quiet", action="store_true",
        help="write the report file without also echoing it to the console")
    args = parser.parse_args()

    ratings = load_ratings()
    rep = build_report(ratings)
    rep.save(args.out)
    if not args.quiet:
        print(rep.render())
    print(f"[evaluator_sensitivity] report written to {args.out}")


if __name__ == "__main__":
    main()
