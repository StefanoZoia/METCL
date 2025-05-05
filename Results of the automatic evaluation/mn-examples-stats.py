import csv
import json

# read metanet classes

mn_classes = list()

with open("data/metanet_classes.jsonl", encoding="utf-8") as file:
    for line in file:
        mn_classes.append(json.loads(line)["metaphor"])

mn_classes.append("OTHER")

# read metanet examples

mn_examples = list()
mn_example_class = dict()

with open("data/mn_examples.csv", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        if row[1] not in mn_examples:
            mn_examples.append(row[1])
            mn_example_class[row[1]] = [row[2]]
        else:
            mn_example_class[row[1]].append(row[2])

# read mn examples combination results

mnex_metcl = None

with open("data/mn_examples_classified_dict.json", "r", encoding="utf-8") as jsonfile:
    mnex_metcl = dict(json.load(jsonfile))

# extend mn classes
for combination in mnex_metcl.values():
    if combination not in mn_classes:
        mn_classes.append(combination)


# read output

classified = dict()

with open("prompt-zs-out/EXTENDED-mnex-deepseek-r1.csv", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter=",")
    for row in reader:
        classified[row[0]] = row[1]


# check that each example has been elaborated and compute stats

count_all = 0
count_null = 0
count_classified = 0
count_metcl_hit = 0
count_hit = 0
count_miss = 0
count_other = 0
llm = set()

for sentence in mn_examples:
    if sentence in classified:
        if classified[sentence] != "NONE":
            count_all += 1
            if classified[sentence] != 'OTHER' and classified[sentence] in mn_classes:
                count_classified += 1
                llm.add(sentence)
                if classified[sentence] in mn_example_class[sentence]:
                    count_hit += 1
                elif sentence in mnex_metcl and classified[sentence] == mnex_metcl[sentence]:
                    count_metcl_hit += 1
                else:
                    count_miss += 1
            else:
                count_other += 1
                if classified[sentence] not in mn_classes:
                    print("Class does not exist:", classified[sentence])
        else:
            count_null += 1
    else:
        print(f"ERROR: sentence '{sentence}' not found")

print(f"Elaborated {count_all} different sentences (void strings excluded)\n")
print(f"MetaNet-classified sentences: {count_classified} ({count_classified / count_all :.2%})")
print(f">>  METCL hits: {count_metcl_hit} ({count_metcl_hit / count_all :.2%})")
print(f">>  hits: {count_hit} ({count_hit / count_all :.2%})")
print(f">>  misses: {count_miss} ({count_miss / count_all :.2%})")
print(f"Other-clasification {count_other} sentences: {count_other / count_all :.2%}\n")

print(f"Recall: {count_hit + count_metcl_hit} ({(count_hit + count_metcl_hit) / count_all :.2%})")
print(f"Precision: {count_hit + count_metcl_hit}/{count_classified} = {(count_hit + count_metcl_hit) / count_classified :.2%}")
delta = len(set(mnex_metcl.keys()).difference(llm))
print(f"difference metcl - llm: +{delta} (+{delta / count_all :.2%})")

