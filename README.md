# Reward-to-risk bench

A validation bench for wide-target FX strategies. It does not predict price. It measures
whether a fixed rule set held a positive expectancy on data it was never fitted to, and
tells you how much evidence you actually have.

## Run it

```bash
pip install -r requirements.txt
streamlit run fx_app.py
python test_engine.py     # 21 correctness checks on the engine
```

The app has two modes, chosen in the sidebar.

**Chart read** takes a screenshot and has Gemini describe what is visibly on it. No price
history needed. The prompt is in `PROMPT.md` and works in any vision model, not just inside
this app.

**Strategy bench** backtests and validates a rule set against price history.

## Files

| File | What it holds |
| --- | --- |
| `fx_engine.py` | Indicators, strategies, backtester, statistics, walk forward, Monte Carlo. No Streamlit, no UI, importable on its own. |
| `fx_app.py` | The interface. Light and dark themes, seven analysis tabs. |
| `fx_vision.py` | Gemini chart reading: the prompt, the API call, response validation, and the arithmetic that turns an invalidation level into an order. |
| `PROMPT.md` | The chart prompt on its own, with notes on why it is shaped that way. |
| `test_engine.py` | Tests for the properties that would invalidate every number if broken. |

## Chart read, and the API key

Uses the current unified `google-genai` SDK. The old `google-generativeai` package is
deprecated and should not be used for new work.

The key never goes in the repo. On Streamlit Community Cloud, open your app's settings and
add it under Secrets:

```toml
GEMINI_API_KEY = "your-key"
```

Locally, `export GEMINI_API_KEY=...`. The app also accepts a key typed into the sidebar for
one session, which is fine for testing and wrong for anything else.

What the model is asked for: structure with named swing points and prices, horizontal levels
with a count of visible touches, a description of the last few candles, the continuation
case and the reversal case side by side, the specific price at which the structural read
breaks, and an explicit list of what the image does not tell you. Returned as JSON against a
schema.

What it is not asked for: a direction, a confidence score, a probability, or a forecast. It
has no calibrated basis for any of them, and a fabricated number is more dangerous than no
number because it gets sized off.

Every response runs through `integrity_check`, which greps the output for confidence
percentages, forward-looking language and impossible touch counts. If the model breaks its
own rules the app says so and tells you to rerun rather than quietly cleaning it up.

The order shown underneath is computed by `plan_from_invalidation`, not written by the model.
The model supplies one number, the price where its read stops being true. Stop distance,
target, cash at risk and lot size are arithmetic from there.

## Data

The bench needs real bars. Sources that work:

- **MT5**: View, then Symbols, then Bars, export to CSV. Or `CopyRates` from a script.
- **HistData.com**: free one-minute history per pair per year, resample it yourself.
- **Dukascopy**: free tick and bar history through their historical data feed.

The CSV needs a date or datetime column plus open, high, low and close. Capitalisation and
column order do not matter. Three to five years on one pair and one timeframe is the target.
More pairs is not more evidence, it is more chances to find a fluke.

## What the engine guarantees

1. **No lookahead.** Indicators at bar `i` use only bars up to `i`. Signals fire on the close
   of bar `i` and fill at the open of bar `i+1`.
2. **No invented numbers.** Nothing in the signal path is random. Randomness appears only in
   the Monte Carlo tab, where it reshuffles trades that already happened.
3. **Costs on every trade.** Half spread in, half spread out, commission in pips, adverse
   slippage on stop exits, optional swap per day held.
4. **Pessimistic bar resolution.** When one bar's range contains both the stop and the
   target, the stop is recorded. With OHLC bars you cannot know the order, and assuming the
   favourable one is how backtests inflate themselves.
5. **Results in R first.** R multiples are independent of account size, leverage and pair, so
   the statistics survive changes to any of those.

## Two different things both written as 1:4

**Risking $1 to make $4** is the reward-to-risk target. It sets where the take profit sits
relative to the stop, and it is the sidebar slider. A wider one does not mean more money, it
means a lower win rate in almost exact proportion.

**Turning $1 into $4** is account growth, and the reward-to-risk setting barely touches it.
Growth per trade is `expectancy_in_R x risk_fraction`. Risking 0.5% with an expectancy of
+0.25R compounds at about 0.125% per trade, which needs roughly 1,100 trades to quadruple.
Risking 2% gets there in about 280 trades, but the account growth tab will usually show that
the same 2% gives you a large chance of hitting your give-up drawdown before you arrive.

The account growth tab runs that as a race between the two outcomes and reports which
happens first. That is the only honest answer to "can it turn 1 into 4", and it is a
property of your edge and your risk fraction, not of the target.

## The arithmetic you should internalise before anything else

A wider target is a trade-off, not an upgrade. Break-even win rate is `1 / (1 + R)` before
costs:

| Target | Win rate needed to break even |
| --- | --- |
| 1:2 | 33.3% |
| 1:3 | 25.0% |
| 1:4 | 20.0% |
| 1:6 | 14.3% |
| 1:8 | 11.1% |

Costs make each of these worse, and they bite harder the tighter your stop is, because cost
is fixed in pips while R is your stop distance. A 2 pip round-turn cost against a 20 pip stop
is 0.10 R deducted from every single trade. At an 18% win rate you pay it five times for
every time you earn it back.

The second consequence is psychological and it is the one that ends most accounts. At a 15%
win rate, a run of 15 losses in a row over 200 trades is more likely than not. It is ordinary
arithmetic, not evidence the system broke. The Monte Carlo tab prints these probabilities so
you can decide in advance which runs are normal.

## Order of work

1. Load three to five years of real bars for the one pair and timeframe you will trade.
2. Set costs from your own account statement. Then widen them by 30% and confirm the edge
   survives that too.
3. Run the target sweep. Pick the reward-to-risk from the plateau of the curve, not the
   single highest bar.
4. Run walk forward. Efficiency below 0.3 means the parameters were fitted to noise, and no
   further adjustment fixes that.
5. Run the sensitivity grid. Fewer than a third of the grid profitable means you found the
   settings by searching rather than because the idea holds.
6. Run Monte Carlo. Lower your risk per trade until the 95th percentile drawdown is a number
   you would keep trading through. Most people find this is under 1%.
7. Demo trade the exact rules for at least 50 trades or three months, whichever is longer.
   Log every trade by hand and compare the demo win rate against the backtest.
8. Go live at a quarter of intended size for the first 30 trades.

## What this does not model

Weekend and news gaps that jump straight through a stop. Variable spreads during releases
and at the daily rollover. Requotes and rejected orders. Correlation if you run several
pairs at once. Each of these makes live results worse than the bench shows, never better.

## Deploying to GitHub and Streamlit Community Cloud

It works. Repo layout:

```
fx-bench/
  fx_app.py            <- set this as the main file when you deploy
  fx_engine.py
  test_engine.py
  requirements.txt     <- must sit at the repo root
  runtime.txt          <- python-3.12
  .streamlit/
    config.toml
  README.md
```

Push the repo, go to share.streamlit.io, connect GitHub, pick the repo and set the main
file to `fx_app.py`. There is no API key and no `.env`, so nothing goes in Secrets. The
Google Fonts import loads in the visitor's browser, not on the server, so it is unaffected
by the container.

Things that will actually bite you there, and what has already been done about them:

| Constraint | Effect | Handled by |
| --- | --- | --- |
| Memory runs from roughly 690MB to 2.7GB, shared | A naive Monte Carlo allocates `paths x trades` float64 arrays several times over and kills the app | Monte Carlo and the growth projection run in float32 chunks. Peak stays near 115MB at 5,000 paths. |
| CPU can drop to 0.078 cores | Every widget change reruns the whole script, and the target sweep is nine backtests | Signals, the main backtest and the target sweep are cached on cheap string keys. The DataFrame is never hashed. |
| No persistent disk | Uploaded files vanish on reboot, and anything written to disk is lost | Nothing is written to disk. Uploads live in memory for the session only. |
| Upload cap | Defaults to 200MB, more than the container can hold as a DataFrame | `.streamlit/config.toml` lowers it to 100MB |
| Apps sleep after 12 hours without traffic | First visitor sees a wake-up page, then a cold start | Nothing to fix, just expect it |

**Use H1 or H4 bars, not M1.** Five years of H1 on one pair is about 31,000 rows and the
whole page computes in well under a second. Five years of M1 is 1.9 million rows, which the
app truncates to the most recent 400,000 and which will feel slow on a throttled core. If
you want minute resolution, resample it locally before uploading.

Anything above roughly 100 concurrent users, or private repo hosting, means moving off the
free tier. For one person testing strategies it is the right home.

## Extending it

Add a strategy by writing a function that takes an OHLC frame plus keyword parameters and
returns a frame with `long`, `short` and `stop_ref` columns, then register it in
`STRATEGIES` with its defaults and a walk-forward grid. Every value in those columns must be
computable at that bar's close. If you find yourself reaching for `.shift(-1)`, stop.

## On accuracy

There is no configuration of this or any other tool that reaches 100% accuracy on price,
and a build that chases it produces a worse system rather than a better one. The mechanism
is specific: accuracy on a fixed dataset can always be pushed toward 100% by adding
parameters and tuning them against that dataset, and every point gained that way is a point
of fit to noise that reverses out of sample. It is exactly why the walk-forward tab exists
and why the efficiency ratio is the number that decides the verdict.

The reachable goals, in order of how much they matter:

1. Expectancy above zero after real costs, holding up out of sample.
2. A t-statistic above 2 on a sample of 300 trades or more.
3. Walk-forward efficiency above 0.5.
4. A broad plateau in the sensitivity grid rather than a spike.
5. A 95th percentile Monte Carlo drawdown you would keep trading through.

A system hitting all five at a 22% win rate on a 1:4 target is a good system. It is wrong
78% of the time. Those two sentences are both true, and holding them at once is most of the
skill.

## Limits worth stating plainly

Every number here describes the past. A positive expectancy measured over three years is
evidence a pattern existed, not a promise it persists. Retail forex brokers serving Nigerian
clients are generally licensed offshore rather than by the SEC in Nigeria, which affects what
recourse you have in a dispute, so check where yours is actually licensed. Risk only money
whose loss would not change your life.
