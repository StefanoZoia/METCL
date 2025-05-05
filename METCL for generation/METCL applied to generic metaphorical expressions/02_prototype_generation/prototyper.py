# here we want to build a prototype for each concept involved in a metaphor from cn-represented.tsv
#   - rank its related concepts with a scoring function: (rel, rel_weight, relatedness) --> [0, 1]
#   - select the N best ranked related concepts and build the prototype based on their score (> .5)

import prototyper_config as cfg
import csv
import cn_rel_getter as rel
import math
import os
from nltk.stem import WordNetLemmatizer


# compute a score for the relation
def compute_score(main_concept, start, rel, end, weight, relatedness):
    score = 0

    # compute score due to the relation type (for ignored relations, directly returns 0)
    if main_concept == start:
        if rel in cfg.outgoing_rel_score:
            score += cfg.outgoing_rel_score[rel] * cfg.feature_weight['rel']
        else:
            return 0
    else:
        if rel in cfg.incoming_rel_score:
            score += cfg.incoming_rel_score[rel] * cfg.feature_weight['rel']
        else:
            return 0
    
    # compute score due to rel_weight

    weight = min(2, weight) / 2     # normalization step, cutting outliers (> 2.0)
    score += weight * cfg.feature_weight['rel_weight'] 

    # compute score due to relatedness
    score += relatedness * cfg.feature_weight['relatedness']

    #      dovuti al fatto che le relazioni hanno score quantizzati e bassi, e che la similaità semantica è alta per i sinonimi (esclusi)
    if score > 0:
        score = math.sqrt(math.sqrt(score))

    return round(score, 3)

# inserts the property into the dic, checking if it's already into it
def conditionally_update_dict(dict, property, score):
    if score > 0.5:     # se lo score è troppo basso, non è una proprietà tipica
        if property in dict:
            dict[property] = max(score, dict[property])
        else:
            dict[property] = score

# returns the dictionaries of positive and negative properties for concept
def get_properties(concept, expand=True):
    name = concept.replace(' ', '_').replace('-', '_')
    REL_FILE = f'data/relations/{name}.tsv'
    
    # define the relations dict
    pos_dict = dict()
    neg_dict = dict()

    # contains the list of concepts to explore if too few relations can be found
    root_concepts = [WordNetLemmatizer().lemmatize(concept)]

    with open(REL_FILE, newline='', encoding='utf-8') as rel_file:
        reader = csv.reader(rel_file, delimiter='\t')
        for row in reader:
            start_concept = row[0].split('/')[-1]   #remove cn prefix
            end_concept = row[2].split('/')[-1]     #remove cn prefix
            relation = row[1]

            # add to root concepts FormOF and DerivedFrom related concepts
            if (relation in cfg.root_rel) and (start_concept == concept):
                root_concepts.append(end_concept)

            # add the other concept as a property, in the appropriate dict
            other = end_concept if start_concept == concept else start_concept
            # some relations have no weight, so they are not considered
            if row[4] == "":
                print(start_concept, relation, end_concept)
                continue
            score = compute_score(concept, start_concept, relation, end_concept, float(row[3]), float(row[4]))
            if relation in cfg.neg_rel:
                conditionally_update_dict(neg_dict, other, score)
            else:
                conditionally_update_dict(pos_dict, other, score)
                
    # if less than MIN_PROP properties have been retrieved, expand to root concepts
    # NOTE: even if a root concept has few properties, the expansion stops
    if expand and (len(pos_dict) + len(neg_dict)) < cfg.MIN_PROP:
        # get each root concept's properties
        for root in root_concepts:
            if rel.isRepresented(root) or rel.get_cn_rel_with_score(root):
                pos_expansion, neg_expansion = get_properties(root, expand=False)
                # add root properties, applying a penalty
                for p in pos_expansion:
                    conditionally_update_dict(pos_dict, p, (cfg.expansion_weight * pos_expansion[p]))
                for p in neg_expansion:
                    conditionally_update_dict(neg_dict, p, (cfg.expansion_weight * neg_expansion[p]))

    return pos_dict, neg_dict


# checks for inconsistencies and returns a merged dict of properties
def merge_properties(pos_dict, neg_dict):
    merged = pos_dict.copy()
    for prop in neg_dict:
        # if inconsistent, remove from merged
        if prop in merged:
            merged.pop(prop)
        else:
            # prepose '-' and append
            neg = f'-{prop}'
            merged[neg] = neg_dict[prop]
    return merged


# if at least MIN_PROP typical properties can be attributed to concept, write its prototype
def write_prototype(concept):
    # get and score concept's properties
    pos_dict, neg_dict = get_properties(concept)
    p_dict = merge_properties(pos_dict, neg_dict)
    # if at least MIN_PROP properties have been retrieved:
    if len(p_dict) >= cfg.MIN_PROP:
        # select the best MAX_PROP properties
        best_list = list()
        for p in sorted(p_dict, key=p_dict.get, reverse=True):
            best_list.append(p)
        # write prototype
        with open(f'{cfg.PROTOTYPE_DIR}/{concept}.txt', 'w', newline='', encoding='utf-8') as prot_file:
            for p in best_list[:cfg.MAX_PROP]:
                prot_file.write(f'{p}: {p_dict[p]}\n')
        return True
    else:
        print(f'WARNING: Could not generate a prototype for concept "{concept}"')
        return False



def main():
    # remove old prototype files
    for filename in os.listdir(cfg.PROTOTYPE_DIR):
        file_path = os.path.join(cfg.PROTOTYPE_DIR, filename)
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


    with open(cfg.OUT_FILE, 'w', newline='', encoding='utf-8') as outputfile:
        with open(cfg.INTERMEDIATE_FILE, newline='', encoding='utf-8') as feed:
            writer = csv.writer(outputfile, delimiter='\t', quotechar='"')
            reader = csv.reader(feed, delimiter='\t', quotechar='"')

            # for each metaphor that has source's and target's relations represented
            for row in reader:
                source = row[0].lower()
                target = row[1].lower()

                if write_prototype(source):
                    if write_prototype(target):
                        # copy row on the output file
                        writer.writerow(row)


if __name__ == "__main__": main()



