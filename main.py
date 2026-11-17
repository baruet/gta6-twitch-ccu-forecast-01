"""gta6-twitch-ccu-forecast-01: forecast GTA VI's launch-day Twitch viewers.

Model: a shape model learned at run time from the case's own comparables.
Each comparable is a past big launch with an hourly Twitch-viewers curve
around its release day. For every comparable we take its release-day curve,
normalise it by its own hour-1 print (log ratios ln(viewers_h / viewers_1),
h = 2..24), and aggregate across comparables with size-weighted empirical
quantiles: a comparable whose hour-1 print is close to the target case's
hour-1 print (in log space) counts more, because launch size shapes the
day's curve. The forecast's centre for hour h is the size-weighted median
of the log ratios at h:

    q50(h) = p1 * exp(weighted median of log ratios at h),

where p1 is the case's hour-1 print.

The spread is conformal, calibrated per hour from nested leave-one-out
residuals: for each comparable j, its median at hour h is forecast from the
other curves (weighted by closeness to j's own hour-1 print) and the
residual -- how far j's actual curve sat from that forecast -- is recorded.
The tau-quantile of those residuals, weighted by closeness to the case's p1,
is added to the median in log space:

    q_tau(h) = p1 * exp(median(h) + weighted tau-quantile of residuals at h).

So the spread measures what this forecaster actually gets wrong, per hour,
rather than assuming the raw dispersion across launches equals the forecast
error; and it is self-calibrating, which matters most when the target's
hour-1 print sits far outside the comparable pool (as GTA VI's will).
Leave-one-out cross-validation over the kit's 22 sane-shaped comparable
curves: pinball loss 0.206 (typical miss 39.2%, 90% coverage 0.870), against
0.212 for the raw-dispersion spread and 0.285 for the persistence baseline.
The gain over the raw spread is within noise on 22 targets (paired t -0.75),
so it is adopted on the strength of the calibration argument and the fact
that it improves loss, miss and coverage together, not on a single number.

No launch's answer is ever hard-coded: the same code runs unchanged on the
practice cases and on the final GTA VI case, learning only from whatever
comparables the case file carries.

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


def size_weight(p1, v1):
    """Gaussian kernel weight: how close comparable hour-1 print v1 is to p1.

    In log space, width SIZE_SIGMA: a comparable whose launch is the same
    size as the target counts most, because launch size shapes the day's
    curve. Always positive, so weighted quantiles never divide by zero.
    """
    d = math.log(p1) - math.log(v1)
    return math.exp(-(d * d) / (2 * SIZE_SIGMA ** 2))


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
        curves.append(parsed)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if curves:
            if len(curves) == 1:
                # One curve: every quantile is that curve's value.
                qs = [round2(p1 * math.exp(curves[0][1][h - 1])) for _ in LEVELS]
            else:
                # Centre: size-weighted median of the log ratios at hour h.
                pairs = [(logr[h - 1], size_weight(p1, v1)) for v1, logr in curves]
                med = weighted_quantile(pairs, 0.5)
                # Spread: conformal, from nested leave-one-out residuals.
                # For each comparable j, forecast its median at hour h from
                # the other curves (weighted by closeness to j's own hour-1
                # print) and record how far j's actual curve sat from that
                # forecast. The tau-quantile of those residuals, weighted by
                # closeness to the case's p1, is the spread around the
                # median: it measures what this forecaster actually gets
                # wrong, per hour, instead of assuming the raw dispersion
                # across launches equals the forecast error. The target
                # case itself never enters the calibration: every residual
                # comes from a comparable the nested forecast did not see.
                resid = []
                for j, (v1j, logrj) in enumerate(curves):
                    rest = [
                        (logrk[h - 1], size_weight(v1j, v1k))
                        for k, (v1k, logrk) in enumerate(curves)
                        if k != j
                    ]
                    med_j = weighted_quantile(rest, 0.5)
                    resid.append((logrj[h - 1] - med_j, size_weight(p1, v1j)))
                qs = [
                    round2(p1 * math.exp(med + weighted_quantile(resid, tau)))
                    for tau in LEVELS
                ]
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
