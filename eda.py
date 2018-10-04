from collections import Counter

from jinja2 import Template
from lxml import etree as ET
from os.path import join
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(style="whitegrid")

tree = ET.parse(join("data", "rest_final.xml"))
root = tree.getroot()

reviews = root.findall("Review")
sentences = root.findall("**/sentence")
print("# Reviews   : ", len(reviews))
print("# Sentences : ", len(sentences))

opinions = root.findall("Review/sentences/sentence/Opinion")
categories = [opinion.attrib["category"] for opinion in opinions]
polarities = [opinion.attrib["polarity"] for opinion in opinions]

print("# Opinions  : ", len(opinions))
df = pd.DataFrame({"categories": categories, "polarities": polarities}).sort_values("categories")
categories = Counter(df.categories)

print(categories)
print("# Categories: ", len(categories))
print(categories)
ambience = categories["REST#AMBIENCE"]
print(ambience)
g = sns.countplot(y="categories", hue="polarities", data=df,
                  palette={"neutral": "yellow", "negative": "#FF1744", "positive": "#00E676"})

plt.tight_layout()
plt.savefig(join("images", "image_rest_final.png"))
plt.show()

template = Template(open(join("report", "README.md.tmp")).read())
data = {}
data["num_reviews"] = len(reviews)
data["num_sentences"] = len(sentences)
data["num_opinions"] = len(opinions)
data["num_categories"] = len(categories)
content = template.render(data=data)
open("README.md", "w").write(content)
