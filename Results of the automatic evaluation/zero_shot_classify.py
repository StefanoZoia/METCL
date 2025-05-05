################
# read dataset #
################

import csv
import json
import time
from statistics import mean
from datetime import timedelta

# read nn-450 sentences

nn450_sentences = list()

with open("data/nn_450.tsv", encoding="utf-8") as file:
    reader = csv.reader(file, delimiter="\t")
    next(reader) # skip header
    for row in reader:
        nn450_sentences.append(row[2])

# read nn-450 combination results

nn450_metcl = None

with open("data/nn_450_classified_dict.json", "r", encoding="utf-8") as jsonfile:
    nn450_metcl = dict(json.load(jsonfile))

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
        mn_example_class[row[1]] = row[2]
        mn_examples.append(row[1])

# read mn examples combination results

mnex_metcl = None

with open("data/mn_examples_classified_dict.json", "r", encoding="utf-8") as jsonfile:
    mnex_metcl = dict(json.load(jsonfile))


#print(f"nn450_senteces length = {len(nn450_sentences)}")
#print(f"mn_classes length = {len(mn_classes)}")
#print(f"mn_examples length = {len(mn_examples)}")

########################
# hypothesis templates #
########################

DEFAULT_TEMPLATE = "This example is {}."

FEW_SHOT_TEMPLATE = """
This example is {}.

Look on the bright side.
This example is GOODNESS IS LIGHT.

They gave me a warm welcome.
This example is AFFECTION IS WARMTH.

The party was really alive.
This example is OTHER.
"""


######################
# Parameters Setting #
######################

MODEL = "microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned"
H_TEMPLATE = DEFAULT_TEMPLATE

OUT_PATH = "zeroshot-out/nn450-llama-3-8b-instruct"

DATASET = "nn450" #"MN" or "NN450"
ENRICH = False    #if True, adds the combined concepts as possible classes

BATCH_START = 0
BATCH_END = 450

#########
# Setup #
#########

# if ENRICH is specified, add the generated combined concepts as classes
if ENRICH:
    if DATASET == "MN":
        for combination in mnex_metcl.values():
            if combination not in mn_classes:
                mn_classes.append(combination)
    elif DATASET == "NN450":
        for combination in nn450_metcl.values():
            if combination not in mn_classes:
                mn_classes.append(combination)
    else:
        raise Exception(f"Invalid dataset specified: {DATASET}")
    

from transformers import pipeline
print(f"mn_classes length during classification = {len(mn_classes)}")

classifier = pipeline("zero-shot-classification", model=MODEL)

# build sentence batch
if DATASET == "MN":
    sentences_list = mn_examples
elif DATASET == "NN450":
    sentences_list = nn450_sentences
else:
    raise Exception(f"Invalid dataset specified: {DATASET}")
batch = sentences_list[BATCH_START:BATCH_END]


##################
# Classification #
##################

last_time = time.perf_counter()
time_delta = 0
time_deltas = []

print(f"mn_classes length during classification = {len(mn_classes)}")
print(f"\n------BATCH: {BATCH_START}-{BATCH_END}--------")
for i, sentence in enumerate(batch):
    #print(f"Working on '{sentence}'")
    if sentence != '':
        result = classifier(sentence, mn_classes, hypothesis_template = H_TEMPLATE)
        with open(f"{OUT_PATH}_{BATCH_START}_{BATCH_END}.csv", "a", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            if DATASET == "MN":
                writer.writerow([sentence, result["labels"][0], mn_example_class[sentence]])
            elif DATASET == "NN450":
                writer.writerow([sentence, result["labels"][0]])
            else:
                raise Exception("Invalid dataset")
    else:
        with open(f"{OUT_PATH}_{BATCH_START}_{BATCH_END}.csv", "a", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            if DATASET == "MN":
                writer.writerow([sentence, "NONE", mn_example_class[sentence]])
            elif DATASET == "NN450":
                writer.writerow([sentence, "NONE"])
            else:
                raise Exception("Invalid dataset")

    curr_time = time.perf_counter()
    time_delta = curr_time - last_time
    time_deltas.append(time_delta)
    expected_remaining_sec = int(mean(time_deltas) * (BATCH_END - BATCH_START - i - 1))
    print(f"Progress: {BATCH_START+i+1}/{BATCH_END}; last sentence took {time_delta:0.1f} sec; finishing in {timedelta(seconds=expected_remaining_sec)}")
    last_time = curr_time