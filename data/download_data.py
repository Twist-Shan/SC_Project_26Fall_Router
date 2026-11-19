"""Download the Chatbot Arena human-preference data (lmarena-ai/arena-human-preference-55k, Apache-2.0).

Usage: python data/download_data.py            -> data/train.csv (about 180 MB)
Then build the derived files (see README.md, "Reproduce").
"""
import os
import sys
import urllib.request

URL = "https://huggingface.co/datasets/lmarena-ai/arena-human-preference-55k/resolve/main/train.csv"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.csv")


def main():
    if os.path.exists(DST):
        print(f"{DST} already exists")
        return
    print(f"downloading {URL}")

    def hook(blocks, bs, total):
        if total > 0:
            sys.stdout.write(f"\r{min(100, 100 * blocks * bs / total):5.1f}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(URL, DST + ".part", hook)
    os.replace(DST + ".part", DST)
    print(f"\nsaved {DST}")


if __name__ == "__main__":
    main()
