#%%
from bs4 import BeautifulSoup

with open("midd_cs_courses.html") as fp:
    doc = BeautifulSoup(fp.read(), features="html.parser")

# %%
from dataclasses import dataclass, asdict


@dataclass
class CSCourse:
    number: int
    title: str
    description: str


courses = []

for header in doc.find_all("h5", {"class": "coursetitle"}):
    for span in header.find_all("span"):
        span.decompose()
    header_text = header.text.strip()
    # now we have, e.g.: CSCI 1015 - Intro to Rocket Propulsion
    assert header_text.startswith("CSCI ")
    (num, title) = header_text[5:].split(" - ")
    description = header.find_next_sibling("div", {"class": "coursedesc"})
    courses.append(CSCourse(int(num), title, description.text.strip()))

# %%
import json

with open("data/midd_cs_courses.jsonl", "w") as out:
    for course in courses:
        print(json.dumps(asdict(course)), file=out)

# %%
