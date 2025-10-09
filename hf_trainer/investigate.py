import argparse
import glob
import os
import re
from typing import Iterable, List, Optional, Tuple


def find_log_files(patterns: Iterable[str]) -> List[str]:
    """Return a sorted list of log file paths matching any of the given glob patterns."""
    matched_files: List[str] = []
    for pattern in patterns:
        matched_files.extend(glob.glob(pattern, recursive=True))
    # Deduplicate and sort for stable output
    return sorted(set(matched_files))


def parse_best_radeval_score(log_path: str, min_epoch: int) -> Tuple[Optional[float], Optional[int]]:
    """Parse a log file and return the best radeval score and its epoch.

    The function looks for patterns like:
      [Metrics] radevalbertscore: 0.2888
      [Metrics] Epoch 3 evaluation complete

    It associates the latest seen radeval*score value before an epoch-complete line
    to that epoch, and tracks the maximum over the entire file.
    """

    # Accept both "radevalscore" and "radevalbertscore" (case-insensitive)
    metric_regex = re.compile(r"\[Metrics\]\s*radeval(?:bert)?score\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
    epoch_done_regex = re.compile(r"\[Metrics\]\s*Epoch\s+(\d+)\s+evaluation complete", re.IGNORECASE)

    best_score: Optional[float] = None
    best_epoch: Optional[int] = None

    last_seen_score: Optional[float] = None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                metric_match = metric_regex.search(line)
                if metric_match:
                    try:
                        last_seen_score = float(metric_match.group(1))
                    except ValueError:
                        # Skip malformed numbers quietly
                        last_seen_score = None
                    continue

                epoch_match = epoch_done_regex.search(line)
                if epoch_match:
                    if last_seen_score is not None:
                        epoch_num = int(epoch_match.group(1))
                        if epoch_num < min_epoch:
                            continue
                        if best_score is None or last_seen_score > best_score:
                            best_score = last_seen_score
                            best_epoch = epoch_num
                    # Reset for next epoch window
                    last_seen_score = None
    except FileNotFoundError:
        return None, None

    return best_score, best_epoch


def model_name_from_log_path(log_path: str) -> str:
    """Derive a model/run name from the log path (parent directory name)."""
    parent_dir = os.path.basename(os.path.dirname(log_path))
    return parent_dir or os.path.basename(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract best radeval*score and epoch per model from logs."
    )
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=[
            # Default pattern matching the user's provided structure
            "ckpt/dino_modern*maira2_tf/*.log",
        ],
        help="Glob pattern(s) for log files (default: ckpt/dino_modern*maira2_tf/*.log)",
    )
    parser.add_argument(
        "--sort",
        choices=["model", "score"],
        default="model",
        help="Sort output by model name or best score (descending for score).",
    )
    parser.add_argument(
        "--min_epoch",
        type=int,
        default=15,
        help="Minimum epoch to consider for best score.",
    )
    args = parser.parse_args()
    print(args)
    log_files = find_log_files(args.patterns)
    if not log_files:
        print("No log files found. Adjust --patterns if needed.")
        return

    results = []
    for log_path in log_files:
        best_score, best_epoch = parse_best_radeval_score(log_path, args.min_epoch)
        model_name = model_name_from_log_path(log_path)
        results.append((model_name, best_score, best_epoch, log_path))

    if args.sort == "model":
        results.sort(key=lambda x: x[0])
    else:  # sort by score descending, None last
        results.sort(key=lambda x: (-1 if x[1] is not None else 0, x[1] if x[1] is not None else -float("inf")), reverse=True)

    print("Model best radeval*score and epoch:")
    for model_name, best_score, best_epoch, log_path in results:
        if best_score is None or best_epoch is None:
            print(f"- {model_name}: N/A (no radeval*score found) [{log_path}]")
        else:
            print(f"- {model_name}: {best_score:.4f} @ epoch {best_epoch} [{log_path}]")


if __name__ == "__main__":
    main()


# Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
# ['valid/patient64544/study1/view1_frontal.png']
# tensor([[ True, False]], device='cuda:0')
# the cardiomediastinal silhouette is normal. the lung parenchyma is clear. there are no pleural or significant bony abnormalities.
# ########################################################
# ['valid/patient64545/study1/view1_frontal.png']
# tensor([[ True, False]], device='cuda:0')
# single portable upright view of the chest demonstrates low lung volumes. there is a small left pleural effusion. there is mild left lower lobe atelectasis. the cardiomediastinal silhouette is unremarkable. the visualized osseous structures are unremarkable.
# ########################################################
# ['valid/patient64548/study1/view1_frontal.png']
# tensor([[ True, False]], device='cuda:0')
# single frontal view of the chest obtained 11 / 1 / 2002 at 4 : 39 p. m. demonstrates stable appearance of the left upper extremity picc line. there is persistent elevation of the left hemidiaphragm with left basilar opacity. there is a small left pleural effusion. no pneumothorax is seen. the cardiomediastinal silhouette is stable. subsequent single frontal radiograph obtained in the upright position on 4 / 17 / 2002 at 5 : 58 am demonstrates interval placement of a right internal jugular central venous catheter with its tip at the cavoatrial junction. no other significant interval change.
# ########################################################
# ['valid/patient64555/study1/view1_frontal.png']
# tensor([[ True, False]], device='cuda:0')
# the cardiomediastinal silhouette is within normal limits. there is a small right pleural effusion. there are patchy opacities in the right upper lung zone, which may represent infection or aspiration. there are no acute osseous abnormalities.
# ########################################################