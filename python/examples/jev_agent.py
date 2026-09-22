"""Let TypeSafe's Jev model play Super Mario Bros and record the run to a movie.

    export TYPESAFE_API_KEY=...          # or put it in .env at the repo root
    python examples/jev_agent.py "Super Mario Bros.nes" jev.nesmovie

    # no key yet? exercise the same harness with the offline rule-based policy:
    python examples/jev_agent.py "Super Mario Bros.nes" jev.nesmovie --policy heuristic

    # Laya (local open-weights model) alone, or Laya first and Jev when Laya is unsure:
    python examples/jev_agent.py "Super Mario Bros.nes" jev.nesmovie --policy laya
    python examples/jev_agent.py "Super Mario Bros.nes" jev.nesmovie --policy duo

Every decision (scene summary, chosen move, probabilities) goes to a .jsonl log next
to the movie, which is the file to read when tuning the questions in nesenv/jev.py.
"""

import argparse

from nesenv.jev import DEFAULT_MODEL, make_policy, play_episode


def main() -> int:
    p = argparse.ArgumentParser(description="Play SMB with Jev making the decisions.")
    p.add_argument("rom")
    p.add_argument("movie", nargs="?", default="jev.nesmovie")
    p.add_argument("--policy", choices=["jev", "heuristic", "laya", "duo"], default="jev")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-decisions", type=int, default=3000, help="hard cap on API calls")
    args = p.parse_args()

    rom = open(args.rom, "rb").read()
    policy = make_policy(args.policy, args.model)
    log_path = args.movie.rsplit(".", 1)[0] + ".jsonl"

    report = play_episode(rom, policy, args.movie, log_path, max_decisions=args.max_decisions)

    for life in report["lives"]:
        print(f"  {life['world']}-{life['stage']}: {life['outcome']} at x={life['distance']}, coins {life['coins']}")
    print(f"{report['outcome']}: {report['levels_cleared']} levels cleared, {report['coins']} coins, "
          f"{report['frames']} frames, {report['decisions']} decisions")
    if hasattr(policy, "usage_line"):
        print(policy.usage_line())
    print(f"wrote {args.movie} (load it with 'Watch Movie' in NES Studio) and {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
