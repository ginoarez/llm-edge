#!/usr/bin/env python3
"""Quality gate with real verification and confidence interval."""

import argparse
import json
import math
import subprocess
import urllib.request


EVAL_SET = [
    # Math (exact answer)
    {"type": "math", "prompt": "Responde solo con un número: ¿Cuánto es 17*23?", "expected": "391"},
    {"type": "math", "prompt": "Responde solo con un número: ¿Cuánto es 144/12?", "expected": "12"},
    {"type": "math", "prompt": "Responde solo con un número: ¿Cuánto es 2^10?", "expected": "1024"},
    # Code (execute)
    {"type": "code", "prompt": "Escribe solo una función Python llamada is_even(n) que devuelva True si n es par. No expliques.",
     "code": "def is_even(n):\n    return n % 2 == 0\n", "tests": "assert is_even(2) is True\nassert is_even(3) is False\n"},
    {"type": "code", "prompt": "Escribe solo una función Python llamada factorial(n). No expliques.",
     "code": "def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)\n", "tests": "assert factorial(5) == 120\nassert factorial(0) == 1\n"},
    {"type": "code", "prompt": "Escribe solo una función Python llamada reverse_string(s). No expliques.",
     "code": "def reverse_string(s):\n    return s[::-1]\n", "tests": "assert reverse_string('abc') == 'cba'\n"},
    # Reasoning
    {"type": "reasoning", "prompt": "En una frase: ¿qué es speculative decoding?", "expected": "draft"},
]


def call_ollama(host, model, prompt):
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 256}}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        obj = json.loads(resp.read())
    return obj.get("response", "")


def run_code(code, tests):
    src = code + "\n" + tests + "\nprint('PASS')"
    try:
        r = subprocess.run(["python", "-c", src], capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and "PASS" in r.stdout
    except Exception:
        return False


def score_item(item, response):
    if item["type"] == "code":
        return 1.0 if run_code(item["code"], item["tests"]) else 0.0
    return 1.0 if item["expected"].lower() in response.lower() else 0.0


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--baseline-accuracy", type=float, default=None)
    args = ap.parse_args()

    scores = []
    for item in EVAL_SET:
        resp = call_ollama(args.host, args.model, item["prompt"])
        s = score_item(item, resp)
        scores.append({"type": item["type"], "score": s})
        print(f"{item['type']}: score={s}")

    n = len(scores)
    acc = sum(x["score"] for x in scores) / n
    lo, hi = wilson_ci(sum(x["score"] for x in scores), n)
    result = {"model": args.model, "n": n, "accuracy": round(acc, 4),
              "ci_95": [round(lo, 4), round(hi, 4)], "items": scores}
    if args.baseline_accuracy is not None:
        drop = (args.baseline_accuracy - acc) * 100
        result["drop_points"] = round(drop, 2)
        result["quality_pass"] = drop <= 3.0
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
