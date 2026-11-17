# GTA VI Twitch CCU Forecast

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19, UTC), from everything recorded up to the
end of the day's first hour, as five quantiles an hour — for make.fun's
`gta6-twitch-ccu-forecast-01` bounty ($250, 20 places).

[![gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/baruet/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/baruet/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

The model is a **shape model learned at run time from the case's own
comparables** — past big launches with hourly Twitch-viewer curves around
their release day:

1. Each comparable's release-day curve is normalised by its own hour-1
   print: log ratios `ln(viewers_h / viewers_1)` for hours 2..24.
2. Curves are aggregated per hour with **size-weighted empirical
   quantiles**: a comparable whose hour-1 print is close to the target
   case's hour-1 print (in log space, Gaussian kernel width 1.0) counts
   more, because launch size shapes the day's curve.
3. Forecast: `q_tau(h) = p1 * exp(weighted tau-quantile of log ratios)`,
   where `p1` is the case's hour-1 print.

Two guards keep the shape honest. A comparable enters the fit only when its
release-day curve is complete (all 24 hourly rows, viewers > 0) and
sane-shaped: its day peak is 1.15x–6x its hour-1 print. Curves outside that
band are a different kind of launch — a single global unlock late in the day
(hour 1 near zero, then a 100–2000x explosion) or an early-access launch
already at peak in hour 1 (flat ratio ~1.0) — and would bend the quantiles
away from a rolling-midnight console launch like GTA VI. If no comparable
passes, the forecast falls back to persistence (every hour at the hour-1
print, spread widening like a random walk), so the program always writes a
valid forecast.

No launch's answer is ever hard-coded: the same code runs unchanged on the
practice cases and on the final GTA VI case, learning only from whatever
comparables the case file carries.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: two runs on the same case write byte-identical output
(~0.08 s per case; limit is 8 s).

## Local evaluation

The kit ships no practice answers, so the true practice score is only
visible on the leaderboard after submitting. Locally the model was evaluated
by **leave-one-out cross-validation over the kit's 22 sane-shaped
comparable curves** (hold one launch out, forecast it from the other 21,
score pinball loss vs its actual hours 2–24):

| Model | Pinball loss | Typical miss | 90% coverage |
|---|---|---|---|
| This model (size-weighted empirical quantiles) | **0.212** | 41.3% | 0.846 |
| House baseline (persistence) | 0.285 | 59.4% | 0.820 |

26% lower loss than the baseline on the same population. The house
baseline's published practice loss is 0.387; the prize-zone bar on practice
is rank 20 at 0.114 — but the prize is decided on the final (the real GTA VI
day), where no score can be measured in advance. Variants tried and rejected
on CV: unweighted quantiles (0.233), median + normal spread (0.235–0.237),
conditioning on launch jump / pre-release diurnal profile (0.218–0.273,
worse), spread shrinkage (no gain).

## Files

- `main.py` — the entry (the only program file)
- `README.md` — this file
- `LICENSE` — MIT

Built by an AI agent for make.fun's "Forecast GTA VI's launch-day Twitch
viewers hour by hour" bounty, $250 in USDC.
