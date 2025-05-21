################
# read dataset #
################

import csv
import json
import time
from statistics import mean
from datetime import timedelta
import torch
from transformers import pipeline, AutoTokenizer

# read demo

demo_sentences = list()
with open("data/demo.txt", encoding="utf-8") as demofile:
    demo_sentences = demofile.readlines()

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

#mn_classes.append("OTHER")

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


####################
# Prompt templates #
####################

FEW_SHOT_TEMPLATE = """Consider the following examples.

Example: Look on the bright side
Class: GOODNESS IS LIGHT

Example: They gave me a warm welcome
Class: AFFECTION IS WARMTH

Example: The party was really alive
Class: OTHER

Assign a class between {} to the following example.
Example: {}
Class:"""

ZERO_SHOT_TEMPLATE = """Assign a class between {} to the following example.
Example: {}
Class:"""


######################
# Parameters Setting #
######################

MODEL = "tiiuae/falcon-7b-instruct"
TEMPLATE = FEW_SHOT_TEMPLATE

OUT_PATH = "prompt-fewshot-out/demo"

DATASET = "DEMO" #"MN" or "NN450"
EXTEND = True    #if True, adds the combined concepts as possible classes
BATCH_START = 0
BATCH_END = 1

TOKEN = "your huggingface token"


#########
# Setup #
#########

# if EXTEND is True, add the generated combined concepts as classes
if EXTEND:
    if DATASET == "MN":
        for combination in mnex_metcl.values():
            if combination not in mn_classes:
                mn_classes.append(combination)
    elif DATASET == "NN450":
        for combination in nn450_metcl.values():
            if combination not in mn_classes:
                mn_classes.append(combination)
    elif DATASET == "DEMO":
        pass
    else:
        raise Exception(f"Invalid dataset specified: {DATASET}")

TOKENIZER = AutoTokenizer.from_pretrained(MODEL, token=TOKEN)

# instantiate pipeline
torch.manual_seed(0)
PIPE = pipeline(
    "text-generation",
    model=MODEL,
    tokenizer=TOKENIZER,
    torch_dtype=torch.bfloat16,
)

# build sentence batch
if DATASET == "MN":
    sentences_list = mn_examples
elif DATASET == "NN450":
    sentences_list = nn450_sentences
elif DATASET == "DEMO":
    sentences_list = demo_sentences
else:
    raise Exception(f"Invalid dataset specified: {DATASET}")
batch = sentences_list[BATCH_START:BATCH_END]

# returns the string length in tokens
def token_length(string):
    string_tokens = TOKENIZER(string, return_tensors="pt")
    string_length = string_tokens.input_ids.shape[1]
    return string_length

# find the 50 token-longest classes in mn_classes
longest_mn_classes = sorted(mn_classes, key=token_length, reverse=True)[:50]

# find the token-longest sentence in sentences_list
longest_sentence = max(sentences_list, key=token_length)

# account space for the generation of the longest class
MAX_NEW_TOKENS = token_length(longest_mn_classes[0])
MAX_PROMPT_LENGTH = TOKENIZER.model_max_length - MAX_NEW_TOKENS

# define the MAX_CLASSES to have in one prompt, trying from 50 down to 2
MAX_CLASSES = 50
too_long = True
while too_long and MAX_CLASSES > 2:
    # check if the formatted prompt exceeds MAX_PROMPT_LENGTH
    prompt = TEMPLATE.format(", ".join(longest_mn_classes[:MAX_CLASSES]), longest_sentence)
    if token_length(prompt) > MAX_PROMPT_LENGTH:
        MAX_CLASSES -= 2
        too_long = True
    else:
        too_long = False
if too_long:
    raise Exception("Model's max length too short to fit current prompt")
else:
    print(f"Max classes in each prompt: {MAX_CLASSES}")

########################
# Auxilliary functions #
########################

def partial_classification(prompt_template, example, classes):

    # build prompt
    prompt = prompt_template.format(", ".join(classes), example)

    # run text generation pipeline
    sequences = PIPE(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_return_sequences = 1,
        return_full_text = False,
        pad_token_id=TOKENIZER.eos_token_id
    )

    generated = sequences[0]['generated_text'].strip()

    # extract selected class from generated text
    result = None
    result_position = None

    for c in classes:
        position = generated.find(c)
        if position != -1:
            if result is None:
                result = c
                result_position = position
            elif position < result_position:
                result = c
                result_position = position
            elif position == result_position and len(c) > len(result):
                result = c
                result_position = position

    return result

# a classification step goes through all available classes and returns a reduced version of the list
def classification_step(prompt_template, example, classes, max_classes):
    selected_classes = list()

    # split the classestiiuae/falcon-7b-instructtiiuae/falcon-7b-instruct
    classes_splits = [classes[i:i+max_classes] for i in range(0, len(classes), max_classes)]

    for split in classes_splits:
        selected = partial_classification(prompt_template, example, split)
        #print(">", selected, "\n")
        if selected is not None:
            selected_classes.append(selected)

    return selected_classes

def classify_example(prompt_template, example, classes, max_classes):
    # the classification starts with all classes
    selected_classes = classes

    # we go step by step until the selected classes are less than max_classes

    while len(selected_classes) >= max_classes:
        #print(len(selected_classes))
        selected_classes = classification_step(prompt_template, example, selected_classes, max_classes)

    # then run the last step
    selected_classes.append("OTHER")
    #print(len(selected_classes))
    selected_classes = classification_step(prompt_template, example, selected_classes, max_classes)
    #print(len(selected_classes))
    return selected_classes[0] if len(selected_classes) > 0 else "OTHER"


###################
# classification  #
###################

last_time = time.perf_counter()
time_delta = 0
time_deltas = []

print(f"mn_classes length during classification = {len(mn_classes)}")
print(f"\n------{DATASET}: {BATCH_START}-{BATCH_END}--------")
for i, sentence in enumerate(batch):
    #print(f"Working on '{sentence}'")
    if sentence != '':
        result = classify_example(TEMPLATE, sentence, mn_classes, MAX_CLASSES)
        with open(f"{OUT_PATH}_{BATCH_START}_{BATCH_END}.csv", "a", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            if DATASET == "MN":
                writer.writerow([sentence, result, mn_example_class[sentence]])
            elif DATASET == "NN450" or DATASET == "DEMO":
                writer.writerow([sentence, result])
            else:
                raise Exception("Invalid dataset")
    else:
        with open(f"{OUT_PATH}_{BATCH_START}_{BATCH_END}.csv", "a", encoding="utf-8", newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            if DATASET == "MN":
                writer.writerow([sentence, "NONE", mn_example_class[sentence]])
            elif DATASET == "NN450" or DATASET == "DEMO":
                writer.writerow([sentence, "NONE"])
            else:
                raise Exception("Invalid dataset")

    curr_time = time.perf_counter()
    time_delta = curr_time - last_time
    time_deltas.append(time_delta)
    expected_remaining_sec = mean(time_deltas) * (BATCH_END - BATCH_START - i - 1)
    print(f"Progress: {BATCH_START+i+1}/{BATCH_END}; last sent {time_delta:0.1f} sec; finishing in {timedelta(seconds=expected_remaining_sec)}")
    last_time = curr_time

