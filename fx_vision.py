"""
fx_vision.py
Gemini chart reading. Uses the current unified SDK (google-genai); the older
google-generativeai package is deprecated and should not be used for new work.

The division of labour matters and is the whole design:

  Gemini reads the image.   It is good at this. It can find swing points, read levels off
                            a price axis, spot that a level has been touched four times,
                            and tell you the chart is unreadable.

  The engine does the maths. Stop distance, R multiple, position size and expectancy are
                            arithmetic. A language model guessing at them produces numbers
                            that look right and are not.

So the model returns observations and price levels. Everything numeric downstream is
computed from those levels by fx_engine, not written by the model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

DEFAULT_MODEL = "gemini-3.5-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite"]


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

CHART_READ_PROMPT = """
You are reading a single screenshot of a financial price chart. Your job is to report what
is visibly on the image, at the level of precision the image actually supports, and nothing
further.

# The rule that governs everything else

You are looking at the left side of a chart. The right side has not happened. You cannot see
it, you cannot infer it, and no amount of structure on the left constrains it. Every
statement you make must be checkable by someone else looking at the same image. If a claim
cannot be checked against a pixel, it does not go in the output.

# Hard prohibitions

Violating any of these makes the entire response useless:

- Never output a probability, confidence score, percentage likelihood, or any number
  expressing how sure you are. You have no calibrated basis for one and a fabricated number
  is worse than silence because it gets acted on.
- Never say what price will do next. Not "likely to continue", not "expect a bounce", not
  "bullish bias". Describe what is on the chart, not what follows it.
- Never name a pattern without listing its defining criteria and pointing at where each one
  is satisfied on this image. "Head and shoulders" with no left shoulder identified by price
  is a hallucination with a label on it.
- Never read a price off the axis you cannot actually resolve. If gridlines are 200 pips
  apart, do not report a level to the nearest pip. Round to what the image supports and say
  so.
- Never invent an instrument or timeframe. If the chart is not labelled, `instrument` and
  `timeframe` are null. Guessing from the shape of the candles is guessing.

# What to do first

Check whether this is a price chart at all. If it is a photograph, a screenshot of text, a
spreadsheet, a non-financial diagram, or anything else, set `is_price_chart` to false, say
what it actually appears to be, and stop. Do not analyse it anyway.

Then assess whether it is legible enough to read: can you resolve individual candles, is the
price axis readable, is there a time axis. If the image is too compressed, too zoomed out,
too blurry, or cropped so the axis is missing, set `readable` to false, list what is missing
in `cannot_determine`, and fill in only the fields you can honestly complete.

# What to report when it is readable

**Structure.** Describe the sequence of swing highs and swing lows in the visible window, in
order, with approximate prices. Higher highs with higher lows is an uptrend in the visible
window. Lower highs with lower lows is a downtrend. Overlapping swings with no directional
progression is a range. State which of the three the visible window shows, and say which
specific swing points led you to that. If the window contains a transition, say where.

**Levels.** Report horizontal price levels that price has visibly reacted to more than once,
with the number of touches you can count and roughly where they are in the window. A level
touched twice is weak evidence. A level touched five times with long wicks is a strong
observation. Report the count honestly; do not round it up.

**The most recent bars.** Describe the last few candles specifically: where they sit
relative to the levels you just listed, whether they are expanding or contracting in range,
and whether the latest close is above or below the levels. This is the part that matters
most and the part most likely to be skipped.

**Both readings.** Give the case a continuation trader would make from this image and the
case a reversal trader would make from the same image. Both must be grounded in specific
features you already listed. If one of them is genuinely weak, say why rather than padding
it. This section exists because the single most common failure in chart reading is finding
only the evidence for the position you already wanted.

**Invalidation.** Give the specific price level at which the structural read you described
stops being true. Not a stop loss suggestion, a structural fact: "below 1.0820 the sequence
of higher lows is broken". This is the most useful single output in the whole response.

**Blind spots.** List what someone acting on this image would not know from it: the spread,
the session, scheduled news, what the higher timeframe looks like, volume if it is absent,
how much history is cropped out to the left, and anything else the image withholds. Be
specific to this image rather than generic.

# Output format

Return a single JSON object and nothing else. No markdown fences, no commentary before or
after. Use null for anything you cannot determine. Use empty arrays rather than inventing
entries.

{
  "is_price_chart": boolean,
  "not_a_chart_reason": string or null,
  "readable": boolean,
  "instrument": string or null,
  "timeframe": string or null,
  "visible_period": string or null,
  "axis_precision": string,
  "structure": {
    "regime": "uptrend" | "downtrend" | "range" | "transition" | "unclear",
    "evidence": string,
    "swing_points": [
      {"kind": "high" | "low", "approx_price": number, "where": string}
    ]
  },
  "levels": [
    {"approx_price": number, "touches": integer, "kind": "support" | "resistance" | "both",
     "note": string}
  ],
  "recent_bars": string,
  "continuation_case": string,
  "reversal_case": string,
  "invalidation": {"price": number or null, "what_it_breaks": string},
  "cannot_determine": [string],
  "notes_for_the_trader": string
}

`notes_for_the_trader` is for anything important that did not fit the schema, including
warnings about the image itself. It is not for a recommendation.
""".strip()


RESPONSE_SCHEMA: Dict = {
    "type": "object",
    "properties": {
        "is_price_chart": {"type": "boolean"},
        "not_a_chart_reason": {"type": "string", "nullable": True},
        "readable": {"type": "boolean"},
        "instrument": {"type": "string", "nullable": True},
        "timeframe": {"type": "string", "nullable": True},
        "visible_period": {"type": "string", "nullable": True},
        "axis_precision": {"type": "string"},
        "structure": {
            "type": "object",
            "properties": {
                "regime": {"type": "string"},
                "evidence": {"type": "string"},
                "swing_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kind": {"type": "string"},
                            "approx_price": {"type": "number"},
                            "where": {"type": "string"},
                        },
                        "required": ["kind", "approx_price", "where"],
                    },
                },
            },
            "required": ["regime", "evidence", "swing_points"],
        },
        "levels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "approx_price": {"type": "number"},
                    "touches": {"type": "integer"},
                    "kind": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["approx_price", "touches", "kind", "note"],
            },
        },
        "recent_bars": {"type": "string"},
        "continuation_case": {"type": "string"},
        "reversal_case": {"type": "string"},
        "invalidation": {
            "type": "object",
            "properties": {
                "price": {"type": "number", "nullable": True},
                "what_it_breaks": {"type": "string"},
            },
            "required": ["what_it_breaks"],
        },
        "cannot_determine": {"type": "array", "items": {"type": "string"}},
        "notes_for_the_trader": {"type": "string"},
    },
    "required": ["is_price_chart", "readable", "axis_precision", "structure", "levels",
                 "recent_bars", "continuation_case", "reversal_case", "invalidation",
                 "cannot_determine", "notes_for_the_trader"],
}


# ---------------------------------------------------------------------------
# Calling the model
# ---------------------------------------------------------------------------

class VisionError(RuntimeError):
    pass


def resolve_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Streamlit Cloud secrets first, then environment. Never hardcode a key in the repo."""
    if explicit:
        return explicit.strip()
    try:
        import streamlit as st
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return str(st.secrets[name]).strip()
    except Exception:
        pass
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        v = os.getenv(name)
        if v:
            return v.strip()
    return None


def read_chart(
    image_bytes: bytes,
    mime_type: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    extra_question: str = "",
    temperature: float = 0.1,
) -> Dict:
    """
    Send one chart image to Gemini and return the parsed structured read.

    Temperature is low on purpose. This task is description, not composition, and variation
    between runs on the same image is a defect rather than creativity.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise VisionError(
            "google-genai is not installed. Add `google-genai` to requirements.txt."
        ) from e

    if not api_key:
        raise VisionError(
            "No API key. On Streamlit Community Cloud put GEMINI_API_KEY in the app's "
            "Secrets. Locally, export it as an environment variable. Never commit it."
        )

    prompt = CHART_READ_PROMPT
    if extra_question.strip():
        prompt += (
            "\n\n# Additional question from the trader\n\n"
            f"{extra_question.strip()}\n\n"
            "Answer it inside `notes_for_the_trader`, under the same rules as everything "
            "else. If answering it honestly requires predicting price or producing a "
            "confidence number, say that you cannot answer it on that basis and explain "
            "what you can observe instead."
        )

    client = genai.Client(api_key=api_key)
    last_error: Optional[Exception] = None

    for candidate in [model] + [m for m in FALLBACK_MODELS if m != model]:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    max_output_tokens=4096,
                ),
            )
            parsed = parse_response(response.text)
            parsed["_model_used"] = candidate
            return parsed
        except Exception as e:  # model unavailable, quota, transient
            last_error = e
            continue

    raise VisionError(f"All model attempts failed. Last error: {last_error}")


def parse_response(text: Optional[str]) -> Dict:
    """Tolerate a stray code fence even though the schema should prevent one."""
    if not text:
        raise VisionError("The model returned an empty response.")
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise VisionError("The model did not return JSON.")
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise VisionError("The model returned JSON that is not an object.")
    return data


# ---------------------------------------------------------------------------
# Turning a read into an order, with arithmetic done here rather than by the model
# ---------------------------------------------------------------------------

@dataclass
class LevelPlan:
    direction: str
    entry: float
    stop: float
    target: float
    stop_pips: float
    target_pips: float
    rr: float
    risk_amount: float
    lots: float
    basis: str


def plan_from_invalidation(
    entry: float,
    invalidation: float,
    rr_target: float,
    equity: float,
    risk_pct: float,
    symbol: str,
    pip_value_per_lot: float = 10.0,
    buffer_pips: float = 3.0,
) -> LevelPlan:
    """
    Build an order from a structural invalidation level the model identified on the chart.

    The model supplies one thing: the price at which its structural read stops being true.
    Everything else here is arithmetic. The stop goes past that level by a buffer, because a
    stop resting exactly on an obvious level is the one that gets taken out by the wick and
    then watched from the sidelines. The target is a multiple of whatever distance that
    produces, and the size falls out of the distance.
    """
    from fx_engine import lots_for_risk, pip_size

    pip = pip_size(symbol)
    if invalidation == entry:
        raise VisionError("Invalidation level equals the entry, so there is no stop distance.")

    direction = "long" if invalidation < entry else "short"
    sign = 1.0 if direction == "long" else -1.0
    stop = invalidation - sign * buffer_pips * pip
    stop_dist = abs(entry - stop)
    target = entry + sign * rr_target * stop_dist
    risk_amount = equity * risk_pct / 100.0
    stop_pips = stop_dist / pip

    return LevelPlan(
        direction=direction,
        entry=entry,
        stop=stop,
        target=target,
        stop_pips=stop_pips,
        target_pips=stop_pips * rr_target,
        rr=rr_target,
        risk_amount=risk_amount,
        lots=lots_for_risk(risk_amount, stop_pips, pip_value_per_lot),
        basis=f"Stop sits {buffer_pips:g} pips beyond the invalidation level at "
              f"{invalidation:.5f}. Target is {rr_target:g}x that distance.",
    )


def integrity_check(read: Dict) -> List[str]:
    """
    Catch the model breaking its own rules. Runs on every response. If these ever fire, the
    read is unreliable and should be treated as such rather than quietly cleaned up.
    """
    problems: List[str] = []
    blob = json.dumps(read).lower()

    if re.search(r"\b\d{1,3}\s?%\s?(confidence|sure|probability|certain)", blob):
        problems.append("A confidence percentage appeared despite the prompt forbidding one.")
    if re.search(r"\b(confidence|probability)[\"']?\s*[:=]\s*\d", blob):
        problems.append("A confidence or probability field was invented.")
    for phrase in ["will rise", "will fall", "will continue", "will break", "expect price to",
                   "i predict", "price target of", "bullish bias", "bearish bias"]:
        if phrase in blob:
            problems.append(f"Forward-looking language found: '{phrase}'.")

    levels = read.get("levels") or []
    for lv in levels:
        if isinstance(lv, dict) and isinstance(lv.get("touches"), int) and lv["touches"] > 12:
            problems.append(f"A level claims {lv['touches']} touches, which is rarely countable "
                            f"on one screenshot.")

    swings = (read.get("structure") or {}).get("swing_points") or []
    if read.get("readable") and not swings:
        problems.append("The read claims the chart is readable but identified no swing points.")

    return problems
