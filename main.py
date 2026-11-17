"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Model: a shape model learned at run time from the case's own comparables.
Each comparable is a past big launch with an hourly Twitch-viewers curve
around its release day. For every comparable we take its release-day curve,
normalise it by its own hour-1 print (log ratios ln(viewers_h / viewers_1),
h = 2..24), and aggregate across comparables with size-weighted empirical
quantiles: a comparable whose hour-1 print is close to the target case's
hour-1 print (in log space) counts more, because launch size shapes the
day's curve. The forecast for hour h is then

    q_tau(h) = p1 * exp(weighted_tau_quantile of log ratios at h),

where p1 is the case's hour-1 print. No launch's answer is ever hard-coded:
the same code runs unchanged on the practice cases and on the final GTA VI
case, learning only from whatever comparables the case file carries.

Two guards keep the shape honest. A comparable only enters the fit when its
release-day curve is complete (all 24 hourly rows present, viewers > 0) and
sane-shaped: its day peak is between 1.15x and 6x its hour-1 print. Curves
outside that band are a different kind of launch -- a single global unlock
late in the day (hour 1 near zero, then a 100-2000x explosion) or an
early-access launch already at its peak in hour 1 (flat ratio ~1.0) -- and
would bend the quantiles away from a rolling-midnight console launch like
GTA VI. If no comparable passes the guards, the forecast falls back to
persistence (every hour at the hour-1 print, spread widening like a random
walk), so the program always writes a valid forecast.

The program reads every <id>.in in the input directory (MAKEFUN_INPUT_DIR,
else /input) and writes <id>.out to the output directory (MAKEFUN_OUTPUT_DIR,
else /output): {"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}
with exactly 23 rows, hours 2..24. Output is deterministic: fixed input
order, no randomness, no wall clock, values rounded to 2 decimals half-up.

Python 3.13 standard library only; no network.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (used only by the fallback).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- shape-fit guards -------------------------------------------------------
# A comparable's release-day peak must be this many times its hour-1 print.
PEAK_MIN_RATIO = 1.15
PEAK_MAX_RATIO = 6.0
# Log-size kernel width for weighting comparables by hour-1 print closeness.
SIZE_SIGMA = 1.0
# Fallback spread (log units per square-root hour), as in the house baseline.
FALLBACK_SIGMA = 0.15


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into a naive-UTC datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00").replace("+00:00", ""))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def weighted_quantile(pairs, level):
    """level-quantile of values weighted by their weights.

    pairs: list of (value, weight) with weight > 0. Returns the smallest
    value whose cumulative weight reaches level * total weight.
    """
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[len(ordered) // 2][0]
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= level * total:
            return value
    return ordered[-1][0]


def comparable_curve(comp):
    """A comparable's release-day log-ratio curve, or None if unusable.

    Returns (hour1_viewers, [log(viewers_h / viewers_1) for h = 1..24]).
    The curve is aligned by whole hours from the comparable's day_start.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hourly = {}
    for row in comp.get("hourly") or []:
        try:
            viewers = row["viewers"]
            if viewers is None or viewers <= 0:
                continue
            idx = int((parse_ts(row["from"]) - day_start).total_seconds() // 3600)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < 24 and idx not in hourly:
            hourly[idx] = viewers
    if len(hourly) != 24:
        return None
    v1 = hourly[0]
    if v1 <= 0:
        return None
    peak = max(hourly.values())
    ratio = peak / v1
    if not (PEAK_MIN_RATIO <= ratio <= PEAK_MAX_RATIO):
        return None
    return v1, [math.log(hourly[h] / v1) for h in range(24)]


def forecast_case(case, name):
    """Forecast one case: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    curves = []
    for comp in case.get("comparables") or []:
        parsed = comparable_curve(comp)
        if parsed is None:
            continue
        v1, logr = parsed
        weight = math.exp(-((math.log(p1) - math.log(v1)) ** 2) / (2 * SIZE_SIGMA ** 2))
        curves.append((logr, weight))

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if curves:
            pairs = [(logr[h - 1], w) for logr, w in curves]
            qs = [round2(p1 * math.exp(weighted_quantile(pairs, tau))) for tau in LEVELS]
        else:
            # No usable comparable: persistence fallback (house baseline).
            qs = [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in Z]
        # Rounding can only break monotonicity by a cent; enforce it.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
