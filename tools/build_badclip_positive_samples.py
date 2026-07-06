"""

Input train_csv is CC3M format (caption,image) CSV; output is (image,caption).
"""
import argparse
import csv
import os

def parse_args():
    p = argparse.ArgumentParser(description="Generate BadCLIP target class positive samples CSV from CC3M")
    p.add_argument("--train_csv", required=True, help="CC3M training CSV, format (caption,image)")
    p.add_argument("--target_label", required=True, help="Target class word, e.g. banana")
    p.add_argument("--output_csv", required=True)
    p.add_argument("--max_samples", type=int, default=None, help="Max samples to extract, default all")
    p.add_argument("--delimiter", default=",", help="CSV delimiter")
    return p.parse_args()

def main():
    args = parse_args()
    root = os.path.dirname(args.train_csv)

    # Filter samples from CC3M where caption contains target word
    matched_rows = []
    with open(args.train_csv, newline="") as f:
        reader = csv.DictReader(f, delimiter=args.delimiter)
        for row in reader:
            caption = row.get("caption", "")
            if args.target_label.lower() in caption.lower():
                # Convert relative path to absolute path
                image_path = row["image"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(root, image_path)
                matched_rows.append({"image": image_path, "caption": caption})

                if args.max_samples and len(matched_rows) >= args.max_samples:
                    break

    if not matched_rows:
        raise ValueError(f"No samples containing {args.target_label!r} found in captions from {args.train_csv}")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "caption"])
        writer.writeheader()
        writer.writerows(matched_rows)

    print(f"[badclip] Filtered {len(matched_rows)} positive samples for target word {args.target_label!r} from CC3M -> {args.output_csv}")

if __name__ == "__main__":
    main()
