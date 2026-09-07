"""Tourist agent CLI.

    python demo.py --example taj_mahal
    python demo.py --example kolhapur
    python demo.py --interactive
    python demo.py --example kolhapur --verbose
"""

from __future__ import annotations

import argparse
import json
import sys

from src.agent import format_cli, run_agent

EXAMPLES = {
    "taj_prediction": "How crowded is Taj Mahal expected to be?",
    "taj_instead": "Will Taj Mahal be crowded, and what should I visit instead?",
    "taj_mahal": "Will Taj Mahal be crowded, and what should I visit instead?",
    "hampi": "I want to visit Hampi. How busy will it be, and are there quieter similar heritage sites?",
    "sanchi": "How crowded are the Buddhist monuments at Sanchi likely to be next year?",
    "qutub": "I plan to visit Qutub Minar. Should I expect high crowds, and what else in Delhi is a lower-pressure option?",
    "kolhapur": (
        "I am new to Kolhapur. I am visiting from 1 September 2026 to 5 September 2026. "
        "I like temples, forts and nature. I want to see Mahalakshmi, Jyotiba and Panhala."
    ),
    "kolhapur_interests": (
        "I'm new to Kolhapur. I am visiting Sep 1-5. I like temples, forts and nature."
    ),
    "kolhapur_lesser": "What are some lesser-known places around Kolhapur?",
    "kolhapur_events": "What's happening in Kolhapur during Sep 1-5?",
    "hourly": "What will the crowd be at Taj Mahal on September 4 at 2 PM?",
    "unknown": "Who is Zorblax?",
    "pune": (
        "I'm new to Pune. I am visiting from 10 October 2026 to 12 October 2026. "
        "I like forts and nature."
    ),
    "delhi": "What else can I visit instead of Qutub Minar in Delhi?",
    "satara": "What are some lesser-known places around Satara?",
    "chennai": "What are some lesser-known places around Chennai?",
    "bengaluru": "I'm visiting Bengaluru. Are there quieter similar heritage sites nearby?",
}

ROUTING_SUITE = [
    "taj_prediction",
    "taj_instead",
    "hampi",
    "kolhapur",
    "pune",
    "delhi",
]


def _print_run(request: str, verbose: bool) -> None:
    print(f"USER: {request}\n")
    try:
        state = run_agent(request)
    except Exception as exc:
        print(f"AGENT ERROR: {type(exc).__name__}: {exc}")
        return
    print(format_cli(state, verbose=verbose))
    if verbose:
        print("\nSTRUCTURED STATE (trimmed)")
        slim = {
            "destination": state.get("destination"),
            "start_date": state.get("start_date"),
            "end_date": state.get("end_date"),
            "interests": state.get("interests"),
            "must_visit": state.get("must_visit"),
            "intent": state.get("intent"),
            "needs_web": state.get("needs_web"),
            "wants_hourly": state.get("wants_hourly"),
            "is_multi_day": state.get("is_multi_day"),
            "tool_trace": state.get("tool_trace"),
            "forecasts": state.get("forecasts"),
            "crowd_levels": {
                k: {kk: v[kk] for kk in ("level", "method") if kk in v}
                if isinstance(v, dict)
                else v
                for k, v in (state.get("crowd_levels") or {}).items()
            },
            "ranked_alternatives": [
                {"name": r.get("name"), "final_score": r.get("final_score")}
                for r in (state.get("ranked_alternatives") or [])[:3]
            ],
            "trip_plan": state.get("trip_plan"),
        }
        print(json.dumps(slim, indent=2, default=str))
    print("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Tourism crowd intelligence agent")
    parser.add_argument("--example", choices=sorted(EXAMPLES))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--request", help="Custom natural-language request")
    parser.add_argument("--all-examples", action="store_true")
    parser.add_argument("--routing-suite", action="store_true")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.routing_suite:
        for name in ROUTING_SUITE:
            print(f"\n######## TEST: {name} ########\n")
            _print_run(EXAMPLES[name], verbose=True)
        return 0

    if args.all_examples:
        for name, request in EXAMPLES.items():
            print(f"\n######## EXAMPLE: {name} ########\n")
            _print_run(request, verbose=True)
        return 0

    if args.interactive:
        request = input("Your trip request: ").strip()
    elif args.request:
        request = args.request
    elif args.example:
        request = EXAMPLES[args.example]
    else:
        request = EXAMPLES["taj_mahal"]

    _print_run(request, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
