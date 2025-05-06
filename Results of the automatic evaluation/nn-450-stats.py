import csv
import json

# read metanet classes

mn_classes = list()

with open("data/metanet_classes.jsonl", encoding="utf-8") as file:
    for line in file:
        mn_classes.append(json.loads(line)["metaphor"])

mn_classes.append("OTHER")

# read nn-450 sentences

nn450_sentences = set()

with open("data/nn_450.tsv", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    next(reader) # skip header
    for row in reader:
        nn450_sentences.add(row[2])

# read nn-450 combination results

nn450_metcl = None

with open("data/nn_450_classified_dict.json", "r", encoding="utf-8") as jsonfile:
    nn450_metcl = dict(json.load(jsonfile))

classified = dict()

#####################
# Parameter Setting #
#####################

RESULTS_PATH = "prompt-fewshot-out/nn450-llama-3.2-3b.csv"
EXTENDED = False

# read output

with open(RESULTS_PATH, encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        classified[row[0]] = row[1]

# extend mn classes
if EXTENDED:
    for combination in nn450_metcl.values():
        if combination not in mn_classes:
            mn_classes.append(combination)

# check that each example has been elaborated and compute stats

count_all = 0
count_classified = 0
count_metcl_hit = 0
count_other = 0
llm = set()

for sentence in nn450_sentences:
    if sentence in classified:
        count_all += 1
        if classified[sentence] != "OTHER" and classified[sentence] in mn_classes:
            count_classified += 1
            llm.add(sentence)
            if EXTENDED and sentence in nn450_metcl and classified[sentence] == nn450_metcl[sentence]:
                count_metcl_hit += 1
        else:
            count_other += 1
            if classified[sentence] not in mn_classes:
                print("Class does not exist:", classified[sentence])
    else:
        print(f"ERROR: sentence '{sentence}' not found")

print(f"Elaborated {count_all} sentences\n")
print(f"MetaNet-classified sentences: {count_classified} ({count_classified / count_all :.2%})")
print(f">>  METCL hits: {count_metcl_hit} ({count_metcl_hit / count_all :.2%})")
print(f"Other-clasification {count_other} sentences: {count_other / count_all :.2%}\n")

#print(len(nn450_metcl), len(llm))
delta = len(set(nn450_metcl.keys()).difference(llm))
print(f"difference metcl - llm: +{delta} (+{delta / count_all :.2%})")