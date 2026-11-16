
import json
from pathlib import Path
from collections import defaultdict, Counter

DATA_PATH = Path("data/news_15k.jsonl")


def normalize(text):
    return " ".join(str(text or "").lower().split())


def main():
    groups = defaultdict(list)
    source_category = Counter()

    with DATA_PATH.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            title = normalize(item.get("title"))
            category = item.get("category", "")
            source = item.get("source", "")

            source_category[(source, category)] += 1

            if title:
                groups[title].append({
                    "line": line_number,
                    "title": item.get("title"),
                    "category": category,
                    "source": source,
                    "url": item.get("url"),
                    "body": item.get("body", ""),
                })

    duplicates = {
        title: records
        for title, records in groups.items()
        if len(records) > 1
    }

    print("========== SOURCE / CATEGORY ==========")
    for (source, category), count in sorted(source_category.items()):
        print(f"{source:12} | {category:15} | {count}")

    print("\n========== DUPLICATE TITLES ==========")
    print("Số nhóm:", len(duplicates))

    for index, (title, records) in enumerate(
        duplicates.items(), 1
    ):
        categories = sorted({
            record["category"] for record in records
        })

        print(f"\n--- NHÓM {index} ---")
        print("Tiêu đề:", title)
        print("Số bài:", len(records))
        print("Các nhãn:", categories)

        if len(categories) > 1:
            print("!!! CẢNH BÁO: TIÊU ĐỀ KHÁC NHÃN !!!")

        for record in records:
            print(
                f"  Source: {record['source']} | "
                f"Category: {record['category']} | "
                f"Body length: {len(record['body'])}"
            )
            print("  URL:", record["url"])

            if record["body"]:
                print("  Nội dung:", record["body"][:200])

    print("\n========== HOAN THANH ==========")


if __name__ == "__main__":
    main()
