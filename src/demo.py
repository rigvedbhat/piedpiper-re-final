"""Thin wrapper so `python -m src.demo` also works."""

from demo import main

if __name__ == "__main__":
    raise SystemExit(main())
