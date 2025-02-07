import sys
import math
from rapidfuzz.distance import Levenshtein

def is_subword(subword, word, error_margin=0.3):

    # Compute Levenshtein distance
    distance = Levenshtein.distance(subword, word)
    threshold = math.ceil(error_margin * max(len(subword), len(word)))

    return distance <= threshold


def main():
    print("=== Subword Inference Program (with % Margin) ===")

    # Input subword and main word
    subword = input("Enter the subword to check: ").strip()
    word = input("Enter the main word: ").strip()

    # Error margin for allowable Levenshtein distance (e.g., 0.3 == 30%)
    # Note: You can still allow zero for exact match if desired.
    try:
        error_margin = float(input(
            "Enter the error margin (0.3 for 30%, 0 for exact match): "
        ).strip())
    except ValueError:
        print("Invalid input. Using default 0.3 (30% margin).")
        error_margin = 0.3

    # Check if subword is a valid subword
    result = is_subword(subword, word, error_margin)

    # Build result message
    if result:
        print(
            f"\nYes, '{subword}' is a subword of '{word}' within a "
            f"{int(error_margin*100)}% margin."
        )
    else:
        print(
            f"\nNo, '{subword}' is not a subword of '{word}' within a "
            f"{int(error_margin*100)}% margin."
        )


if __name__ == "__main__":
    main()
