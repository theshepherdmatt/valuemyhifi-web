#!/usr/bin/env python3

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# -------------------------------------------------
# SETUP
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.4

app = Flask(__name__)
CORS(app)

# -------------------------------------------------
# CORE PROMPT BUILDER
# -------------------------------------------------
def build_prompt(units, country, include_sales):
    return f"""
You are an experienced second-hand hi-fi dealer.

You write calmly, plainly, and factually.
You do not use marketing language.
You do not use Markdown.
You output JSON only.

COUNTRY
{country}

INCLUDE SALES DESCRIPTION
{include_sales}

INPUT UNITS
{units}

TASK

For each unit, return:
- unit_name
- estimated_value (price range string)
- confidence_level ("high", "medium", or "broad")
- context (short factual paragraph)
- sales_description (string or null)

Then return:
- combined_value (price range string)
- disclaimer (string)

PRICING BEHAVIOUR (NON-NEGOTIABLE)

For well-known, commonly traded models:
- Use a narrow price range
- Total range width should be approximately ±£50 (±£100 total)
- Changing condition shifts the midpoint, not the range width
- Do not widen ranges for poor condition

For obscure, uncommon, or poorly documented items:
- Use a wider, conservative range
- Set confidence_level to "broad"
- State clearly in context that pricing is a ballpark estimate based on comparable equipment
- Do not invent specific sales history

CONDITION HANDLING

- Higher condition raises the midpoint
- Lower condition lowers the midpoint
- Range width must remain consistent for known models

SALES DESCRIPTION RULES

If include_sales is true:
- sales_description MUST be present
- Length: 2–3 short paragraphs
- Tone: calm, factual, honest
- Describe what the item is, its typical reputation or use, and the stated condition
- Do NOT exaggerate condition
- Do NOT include pricing
- Do NOT use hype or emotional language
- Do NOT use bullet points
- Do NOT mention platforms or marketplaces

OUTPUT RULES

- Output valid JSON only
- No prose outside JSON
- No explanations

JSON SCHEMA (STRICT)

{{
  "units": [
    {{
      "unit_name": "",
      "estimated_value": "",
      "confidence_level": "high | medium | broad",
      "context": "",
      "sales_description": null
    }}
  ],
  "combined_value": "",
  "disclaimer": ""
}}

DISCLAIMER TEXT (USE EXACTLY)

"These figures are indicative only and reflect typical second-hand market behaviour. Final sale prices vary depending on condition, demand, and presentation."
""".strip()


# -------------------------------------------------
# API ENDPOINT
# -------------------------------------------------
@app.route("/value", methods=["POST"])
def value_my_hifi():
    data = request.get_json(force=True)

    units = data.get("units")
    country = data.get("country", "United Kingdom")
    include_sales = data.get("include_sales", False)

    if not units or not isinstance(units, list):
        return jsonify({"error": "No units provided"}), 400

    prompt = build_prompt(units, country, include_sales)

    response = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a calm, experienced second-hand hi-fi dealer. "
                    "You are factual and understated."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    import json

    raw = response.choices[0].message.content.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return jsonify({
            "error": "Invalid JSON returned from model",
            "raw": raw
        }), 500

    return jsonify(parsed)


# -------------------------------------------------
# RUN LOCAL SERVER
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
