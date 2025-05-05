import csv
import json
from nltk.stem import WordNetLemmatizer

# CHANGE HERE to choose how many sentences from what resource are to be classified
CORPUS = "metanet_examples"
# CORPUS = "nn450"
BATCH_START = 0
BATCH_END = 100

OUTPUT_FILE_NAME = f"output/{CORPUS}_frame_based_{BATCH_START}_{BATCH_END}"


# Data retrieval
def retrieve_metanet_annotations():
    annotations = []
    mn_example_classes = []
    with open("data/metanet_annotations.csv", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=",")
        header = next(reader)
        for row in reader:
            if row[4] == "-":
                continue
            annotation = {
                "source": row[2],
                "target": row[3],
                "sentence": row[1]
            }
            annotations.append(annotation)
            mn_example_classes.append(row[0])
    return mn_example_classes, annotations

def retrieve_nn450_annotations():
    annotations = []
    with open("data/nn_450.tsv", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        header = next(reader)
        for row in reader:
            annotation = {
                "source": row[0],
                "target": row[1],
                "sentence": row[2]
            }
            annotations.append(annotation)
    return annotations

def retrieve_lus(frame, max_depth=1):
    with open("data/mn_lexical_units.jsonl", "r", encoding='utf8') as f:
        metanet_frames = [json.loads(line) for line in f.readlines()]

    metanet_frame = None

    for mf in metanet_frames:
        if mf["frame"] == frame.replace("_", " "):
            metanet_frame = mf
    
    lex_units_set = set()

    if metanet_frame is None:
        return lex_units_set
    
    lus = metanet_frame['lus']
    if max_depth > 0:    
        for ancestor_name, depth in metanet_frame['ancestors']:
            if depth > max_depth:
                continue
            try:
                ancestor_frame = next(x for x in metanet_frames if x['frame'] == ancestor_name)
            except StopIteration:
                continue
            for key in ancestor_frame['lus'].keys():
                lus[key].extend(ancestor_frame['lus'][key])

    # make a set out of the 4 lists
    for key in lus.keys():
        lex_units_set.update(lus[key])

    return lex_units_set

def load_mn_data():
    frames_evoked_by = dict()    #TODO: trattare correttamente parentesi quadre/tonde e LUs con più termini (e quindi più POS tags)
    metaphors_with_frame = {
        "source": dict(),
        "target": dict()
    }
    
    with open("data/metanet_classes.jsonl", encoding="utf-8") as scraped_data:
        for line in scraped_data:
            metanet_class = json.loads(line.strip())

            for key in ["source", "target"]:
                frame = metanet_class[key + " frame"]
                if frame is None or frame == "-":
                    continue
                # add frame to dict
                if frame not in metaphors_with_frame[key]:
                    metaphors_with_frame[key][frame] = set()
                metaphors_with_frame[key][frame].add(metanet_class["metaphor"])

                # add LUs to dict
                frame_lus = retrieve_lus(frame)
                for lu in frame_lus:
                    lemma = lu.split("[")[0].split(".")[0].strip().lower()
                    if lemma not in frames_evoked_by:
                        frames_evoked_by[lemma] = set()
                    frames_evoked_by[lemma].add(frame)

    return frames_evoked_by, metaphors_with_frame


# Classification
def get_evoked_frames(annotation, frames_evoked_by):
    evoked_frames = {
        "source": [],
        "target": []
    }
    wnl = WordNetLemmatizer()
    for key in ["source", "target"]:
        lemma = wnl.lemmatize(annotation[key].lower()).lower()
        if lemma in frames_evoked_by:
            evoked_frames[key] = list(frames_evoked_by[lemma]) 
    return evoked_frames

def assign_class(evoked_frames, metaphors_with_frame):
    evoked_mets = {
        "source": set(),
        "target": set(),
    }
    for key in ["source", "target"]:
        for frame in evoked_frames[key]:
            if frame in metaphors_with_frame[key]:
                evoked_mets[key].update(metaphors_with_frame[key][frame])

    assigned_classes = evoked_mets["source"].intersection(evoked_mets["target"])
    return assigned_classes

def save_to_file(row):
    with open(F"{OUTPUT_FILE_NAME}.tsv", "a", encoding='utf-8', newline="") as tsvfile:
        output_writer = csv.writer(tsvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(row)

def classify_metanet_examples(frames_evoked_by, metaphors_with_frame):
    mn_example_classes, annotations = retrieve_metanet_annotations()
    batch = annotations[BATCH_START:BATCH_END]
    for i, annotation in enumerate(batch):
        evoked_frames = get_evoked_frames(annotation, frames_evoked_by)
        candidate_classes = assign_class(evoked_frames, metaphors_with_frame)
        row = [annotation["sentence"], list(candidate_classes), mn_example_classes[BATCH_START+i]]
        save_to_file(row)

def classify_nn450(frames_evoked_by, metaphors_with_frame):
    annotations = retrieve_nn450_annotations()
    batch = annotations[BATCH_START:BATCH_END]
    for annotation in batch:
        evoked_frames = get_evoked_frames(annotation, frames_evoked_by)
        candidate_classes = assign_class(evoked_frames, metaphors_with_frame)
        row = [annotation["sentence"], list(candidate_classes)]
        save_to_file(row)

def main():
    frames_evoked_by, metaphors_with_frame = load_mn_data()
    open(f"{OUTPUT_FILE_NAME}.tsv", "w").close() # clear output file

    if CORPUS == "metanet_examples":
        classify_metanet_examples(frames_evoked_by, metaphors_with_frame)
    elif CORPUS == "nn450":
        classify_nn450(frames_evoked_by, metaphors_with_frame)
    print(f"Classified sentences from {BATCH_START} to {BATCH_END} from {CORPUS}")

if __name__ == "__main__":
    main()