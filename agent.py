# agent.py
# The ICD-10 coding agent: an LLM (Groq) that calls the tools in tools.py,
# with guardrails (system prompt, max_steps, graceful finish, retry, code verification).

import os
import re
import json
import time

from dotenv import load_dotenv
from groq import Groq, BadRequestError, RateLimitError

from tools import predict_codes, search_coding_guidelines, lookup_code, search_codes, code_to_description

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# strip accidental spaces/quotes around the key (a common copy-paste problem in cloud secret boxes)
GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'").strip()
client = Groq(api_key=GROQ_API_KEY or "missing")
MODEL = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are an ICD-10-CM diagnosis coding assistant. Follow these rules strictly:
1. Always call predict_codes first with the full note.
2. Only recommend codes returned by predict_codes, search_codes or lookup_code (exists = true). Never invent codes.
3. If the classifier's codes do not match the note, use search_codes with a plain-words description of the documented diagnosis (include left/right, acute/chronic, encounter type if written) and pick the best-matching code from its results. Use lookup_code to verify a specific code or list a category. Never call the same tool with the same input twice.
4. Report classifier confidence exactly as returned. Never invent confidence numbers. For codes added via lookup_code, write "not from classifier".
5. The guidelines contain general coding rules, not definitions of individual codes. Call search_coding_guidelines at most 2 times. Cite a passage (with its page number) only if it is relevant. If nothing relevant is found, write "No specific guideline passage found" instead of searching again. Never cite a page you did not receive.
6. If the top classifier confidence is below 0.40, add the warning: "Low classifier confidence - human review recommended."
7. Diagnosis codes only. Do not suggest procedure codes (CPT or ICD-10-PCS).
8. Keep the answer short: a table of ONLY the codes you recommend (one best code per documented diagnosis; do not list alternatives), then one justification line per code, then warnings. If you reject classifier codes, mention them briefly under "Rejected classifier suggestions" with the reason, never in the main table.
9. Only if the input is not clinical text at all (e.g. greetings or random words), reply "No codable diagnosis found in the note." and suggest no codes. If the note describes any symptom, finding, injury or disease, it IS codable: low classifier confidence is not a reason to give up; use lookup_code to find the most specific code from the note text.
10. Cite guideline passages only as (ICD-10-CM Guidelines, p. X). Do not use markers like [1] or 【1】.
11. Never pick a code that adds details the note does not document (e.g. a specific part of an organ, "old" vs "current" injury, a complication). If a detail is missing, prefer the "other" or "unspecified" option that matches only what is written, and mention the missing detail as a documentation gap."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "predict_codes",
            "description": "Predict the top ICD-10-CM codes for a clinical note using a fine-tuned BioBERT classifier. Returns code, description and confidence for each.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string", "description": "The full clinical note text."}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_coding_guidelines",
            "description": "Search the official ICD-10-CM Guidelines for Coding and Reporting (FY2026). Returns the most relevant passages with page numbers for citation.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look up in plain words, e.g. a code description or a coding rule question."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_code",
            "description": "Check whether an ICD-10-CM code exists and get its official description. If you pass a category prefix (e.g. K57.3), it lists the billable codes inside it. Use this to verify or find the most specific code before recommending it.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "An ICD-10-CM code, with or without the dot, e.g. K57.31"}},
                "required": ["code"],
            },
        },
    },
]

TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "search_codes",
        "description": "Find ICD-10-CM codes by describing the diagnosis in plain words (e.g. 'tear medial meniscus right knee current injury initial encounter'). Returns up to 10 official codes whose descriptions best match. Include body side (left/right), acute/chronic and encounter type when the note states them.",
        "parameters": {
            "type": "object",
            "properties": {"description": {"type": "string", "description": "Plain-words description of the diagnosis as documented in the note."}},
            "required": ["description"],
        },
    },
})

AVAILABLE_TOOLS = {
    "predict_codes": predict_codes,
    "search_coding_guidelines": search_coding_guidelines,
    "lookup_code": lookup_code,
    "search_codes": search_codes,
}

GUIDELINE_TEXT_LIMIT = 600   # characters of each guideline passage sent to the LLM (saves tokens)

NO_DIAGNOSIS_THRESHOLD = 0.10   # below this top classifier confidence (and with no verified lookup) -> nothing codable

CODE_PATTERN = r"\b[A-Z][0-9][0-9A-Z](?:\.?[0-9A-Z]{1,4})?\b"


def extract_codes(text):
    # All code-looking strings in a text, cleaned (no dot, uppercase)
    return {c.replace(".", "").upper() for c in re.findall(CODE_PATTERN, text)}


def call_llm(messages, tool_choice="auto", max_retries=3):
    # One Groq call, retried if the model garbles a tool call (400) or we hit the rate limit (429)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL, messages=messages, tools=TOOL_DEFINITIONS, tool_choice=tool_choice
            )
            return response.choices[0].message
        except BadRequestError as error:
            retryable = "output_parse_failed" in str(error) or "tool_use_failed" in str(error)
            if retryable and attempt < max_retries - 1:
                print(f"  (retry {attempt + 1}: model produced an invalid output)")
                continue
            raise
        except RateLimitError:
            if attempt < max_retries - 1:
                print(f"  (retry {attempt + 1}: rate limit, waiting 20 seconds)")
                time.sleep(20)
                continue
            raise


def run_agent(note, max_steps=6):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Suggest ICD-10-CM codes for this note: " + note},
    ]
    steps = []
    codes_from_tools = set()
    classifier_suggestions = []
    previous_calls = {}

    for step in range(max_steps):
        reply = call_llm(messages)

        if not reply.tool_calls:
            if reply.content and reply.content.strip():
                return reply.content, steps, codes_from_tools, classifier_suggestions
            # Empty answer (the model only "thought" and wrote nothing): nudge it once more
            print(f"  (step {step + 1}: empty answer, asking the model to write its final answer)")
            messages.append({"role": "user", "content": "Your last reply was empty. Write your final answer now, following the rules."})
            continue

        messages.append({
            "role": "assistant",
            "content": reply.content,
            "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in reply.tool_calls
            ],
        })

        for call in reply.tool_calls:
            tool_name = call.function.name
            tool_args = json.loads(call.function.arguments)

            # Guardrail: the classifier always gets the ORIGINAL note, never the LLM's rewritten version
            if tool_name == "predict_codes":
                tool_args = {"note": note}

            print(f"Step {step + 1}: {tool_name}({tool_args})")
            steps.append({"step": step + 1, "tool": tool_name, "args": tool_args})

            # Guardrail: never re-run an identical call; remind the model it already has the answer
            call_key = tool_name + json.dumps(tool_args, sort_keys=True)
            if call_key in previous_calls:
                result = {"note": "You already called this tool with this exact input. Use the earlier result and move on.",
                          "earlier_result": previous_calls[call_key]}
            else:
                result = AVAILABLE_TOOLS[tool_name](**tool_args)
                if tool_name == "search_coding_guidelines":
                    result = [{**r, "text": r["text"][:GUIDELINE_TEXT_LIMIT]} for r in result]
                previous_calls[call_key] = result

            # Remember which codes came from tools (provenance), not from the LLM's memory
            if tool_name == "predict_codes":
                classifier_suggestions = result
                codes_from_tools |= extract_codes(json.dumps(result))
            if tool_name == "search_codes" and isinstance(result, list):
                for r in result:
                    codes_from_tools.add(r["code"])
            if tool_name == "lookup_code" and "exists" in result:
                # Only codes the tool CONFIRMED count - not the code the LLM asked about if it doesn't exist
                if result.get("exists"):
                    codes_from_tools.add(result["code"])
                for child in result.get("billable_codes_starting_with_this", []):
                    codes_from_tools.add(child["code"])

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result),
            })

    # Graceful finish: out of steps, so force a text answer from what we already have
    messages.append({
        "role": "user",
        "content": "Tool budget reached. Do NOT call any tools. Give your final answer now, using only the tool results above.",
    })
    try:
        reply = call_llm(messages, tool_choice="none")
        answer = reply.content
    except BadRequestError:
        # Last safety net: never crash the API, return an honest message instead
        answer = ("The agent could not finish within its step budget. "
                  "See the classifier suggestions below. Human review required.")
    return answer, steps, codes_from_tools, classifier_suggestions


def verify_codes(answer_text, codes_from_tools):
    # Python-side check of every code in the final answer: is it real, and did a tool give it?
    report = []
    for code in sorted(extract_codes(answer_text)):
        report.append({
            "code": code,
            "valid": code in code_to_description,
            "from_tools": code in codes_from_tools,
            "official_description": code_to_description.get(code, "NOT A BILLABLE ICD-10-CM CODE"),
        })
    return report


def analyze_note(note):
    answer, steps, codes_from_tools, classifier_suggestions = run_agent(note)
    answer = answer or "The agent returned an empty answer. Human review required."

    # Deterministic guard (does not trust the LLM): if the classifier is almost guessing
    # AND the agent never verified any code from the note text, there is nothing codable.
    top_confidence = max((s["confidence"] for s in classifier_suggestions), default=0)
    used_lookup = any(s["tool"] in ("lookup_code", "search_codes") for s in steps)
    guard_triggered = top_confidence < NO_DIAGNOSIS_THRESHOLD and not used_lookup
    if guard_triggered:
        answer = ("**No codable diagnosis found in the note.** "
                  "The classifier confidence was very low and no code could be verified from the note text. "
                  "Human review required.")
    # Only check the RECOMMENDED part of the answer, not the "Rejected classifier suggestions" section
    recommended_part = re.split(r"rejected classifier suggestions", answer, flags=re.IGNORECASE)[0]
    code_checks = verify_codes(recommended_part, codes_from_tools)
    problem_codes = [c["code"] for c in code_checks if not (c["valid"] and c["from_tools"])]
    if problem_codes:
        # Deterministic warning banner: the LLM's claims are not trusted over the Python check
        answer = ("**WARNING: the answer below contains code(s) that failed verification: "
                  + ", ".join(problem_codes)
                  + ". Do not use them without human review.**\n\n" + answer)
    return {
        "note": note,
        "answer": answer,
        "code_checks": code_checks,
        "problem_codes": problem_codes,
        "classifier_suggestions": classifier_suggestions,
        "steps": steps,
        "guard_triggered": guard_triggered,
    }
