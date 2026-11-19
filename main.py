"""
CAN CENTRAL BANK SENTIMENT HELP PREDICT GBP/USD VOLATILITY?   - START HERE

Open this file and press Run (VS Code: the ▶ button top-right; PyCharm: right-click ->
Run 'main'). Everything is done from the menu - no terminal commands needed.

  1) Look up a Fed or BoE meeting: the decision, what was said and its sentiment
  2) GBP/USD, volatility and central bank sentiment
  3) How GBP/USD reacted to chosen meetings (first hour + next days)
  4) Central bank sentiment summary -> GBP/USD prediction (next day or any past day)

Tables are printed here and saved in results/, charts open in a window and are saved as
PNG files in results/menu/. To check packages and data sources run check_setup.py.
"""
import sys
import warnings

import config

sys.path.insert(0, str(config.ROOT))
warnings.filterwarnings("ignore")

TITLE = "CAN CENTRAL BANK SENTIMENT HELP PREDICT GBP/USD VOLATILITY?"


def _views():
    import menu
    return menu


OPTIONS = {
    "1": ("Look up a Fed or BoE meeting: decision, what was said, sentiment", lambda: _views().meeting_view()),
    "2": ("GBP/USD, volatility and central bank sentiment", lambda: _views().market_view()),
    "3": ("How GBP/USD reacted to chosen meetings (first hour + next days)", lambda: _views().reaction_view()),
    "4": ("Sentiment summary -> GBP/USD prediction", lambda: _views().decision_view()),
}


def main():
    while True:
        print("\n" + "=" * 64)
        print("  " + TITLE)
        print("=" * 64)
        for k, (label, _) in OPTIONS.items():
            print(f"   {k}) {label}")
        print("   0) Exit")
        choice = input("\n  Choose an option: ").strip()
        if choice in ("0", "q", "exit"):
            print("  Bye!")
            break
        if choice not in OPTIONS:
            print("  Please type a number from 0 to 4.")
            continue
        try:
            OPTIONS[choice][1]()
        except KeyboardInterrupt:
            print("\n  Stopped.")
        except Exception as exc:
            print(f"\n  Something went wrong: {type(exc).__name__}: {exc}")
        input("\n  Press Enter to return to the menu...")


if __name__ == "__main__":
    main()
