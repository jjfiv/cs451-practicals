import json, csv

# I have to open: https://docs.python.org/3/library/csv.html every time...

keep = []
with open("data/poetry_id.jsonl") as fp:
    for line in fp:
        data = json.loads(line)
        row = {
            "book": data["book"],
            "page": data["page"],
            "use": data["use"],
            "poetry": data["poetry"],
            **data["features"],
        }
        keep.append(row)

with open("data/poetry.csv", "w") as out:
    writer = csv.DictWriter(out, fieldnames=keep[0].keys())
    writer.writeheader()
    writer.writerows(keep)