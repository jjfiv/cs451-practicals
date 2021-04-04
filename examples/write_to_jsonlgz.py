import json, gzip

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

# Gzip.open with "wt" as write-text:
with gzip.open("data/poetry.jsonl.gz", "wt") as out:
    for entry in keep:
        dict_to_str = json.dumps(entry)
        print(dict_to_str, file=out)