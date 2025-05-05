import json
import csv
import requests
import time
import pprint

# MetaNet Filter:
# - input: MetaNet conceptual metaphors and frames scraped from website
# - output: MetaNet conceptual metaphor corpus with source and target represented by cadidate concepts
#           as json lists, ordered by distance in terms of MetaNet "subcase" relations

# For each conceptual metaphor m, the candidate src and tgt concepts are looked for following these rules:
# 1) src/tgt frame of m
# 2) derive src/tgt from m's name, assuming that it has the form "TARGET [BE] SOURCE"
# 3) use the src/tgt frames "relevant framenet frames" as src/tgt
# 4) breadth-first search of “is subcase of” related frames, starting from src/tgt frames of m
# 5) breadth-first search of "both source and target subcase of" and "src/tgt subcase of" related metaphors:
#    \-> apply rules 1 to 4 to each related metaphor

# rel_dict is a dict of dicts
# each internal dict contains related nodes and their distance
# the expansion adds to the external dictionary all reachable nodes and their distance
def relation_expansion(rel_dict):
    # for each key
    for cur_node in rel_dict.keys():
        # build a FIFO queue
        # add each node in the main value list to the queue
        queue = list(rel_dict[cur_node].keys())

        # while the queue is not void
        while queue:
            # n = first node in the queue
            # remove n from the queue
            n = queue.pop(0)

            # expand on n if possiblle
            if n in rel_dict.keys():

                # for each node related to n
                for new_node in rel_dict[n].keys():
                    # add it to the queue
                    queue.append(new_node)

                    # if new_node is not in the cur_node related list, add it
                    if new_node not in (rel_dict[cur_node].keys()):
                        # the distance to new_node is (the distance to n) + 1 
                        rel_dict[cur_node][new_node] = rel_dict[cur_node][n] + 1
                    else:
                        # if a shorter path is found, update the distance
                        rel_dict[cur_node][new_node] = min(rel_dict[cur_node][new_node], rel_dict[cur_node][n] + 1)


# auxiliary function to split the name of the conceptual metaphor around the verb TO BE,
# assuming that the conceptual metaphor name is called TARGET [BE] SOURCE
def split_conceptual_metaphor(metaphor):
    splitters = [" are the ", " are an ", " are a ", " are ",
                  " is the ",  " is an ",  " is a ",  " is "]
    # case-insensitive search (not all conceptual metaphor names are uppercase)
    metaphor = metaphor.lower()
    for be in splitters:
        # split the name on the first applicable splitter
        if be in metaphor:
            metaphor = metaphor.split(be)
            return (metaphor[0].strip(), metaphor[1].strip())
SPLITTED_SOURCE_INDEX = 1
SPLITTED_TARGET_INDEX = 0


# auxiliary function to update a candidate concepts distance dictionary
def update_distance_dict(distance_dict, to_node, new_distance):
    # if to_node is not in the dict, add it
    if to_node not in distance_dict.keys():
        distance_dict[to_node] = new_distance
    # if a shorter path is found, update the distance
    else:
        distance_dict[to_node] = min(distance_dict[to_node], new_distance)


def main():

    ###########################################
    # Read metanet_classes and metanet_frames #
    ###########################################

    # save source and target of each metaphor in a dictionary
    metaphors_dict = dict()
    # save "source subcase of" and "target subcase of" relations in two dictionaries
    super_src_of = dict()
    super_tgt_of = dict()

    with open("data/metanet_classes.jsonl", encoding="utf-8") as scraped_data:
        for line in scraped_data:
            scraped = json.loads(line.strip())
            
            metaphor = scraped["metaphor"]
            src = scraped["source frame"]
            tgt = scraped["target frame"]

            metaphors_dict[metaphor] = {"source": src, "target": tgt}

            super_st = scraped["both s and t subcase of"]

            super_source = scraped["source subcase of"]
            super_src_list = super_st + super_source
            super_src_of[metaphor] = {el: 1 for el in super_src_list}

            super_target = scraped["target subcase of"]
            super_tgt_list = super_st + super_target
            super_tgt_of[metaphor] = {el: 1 for el in super_tgt_list}

    # save relevant_fn_frames for each frame
    fn_frames_for = dict()
    # save "subcase of" frame relation in a dictionary
    super_frames_of = dict()

    with open("data/metanet_frames.jsonl", encoding="utf-8") as scraped_data:
        for line in scraped_data:
            scraped = json.loads(line.strip())
            
            frame = scraped["frame"]
            super_frames = scraped["subcase of"]
            fn = scraped["relevant_fn_frames"]

            super_frames_list = super_frames
            super_frames_of[frame] = {el: 1 for el in super_frames_list}
            fn_frames_for[frame] = fn

    # extend the value lists of the subcase dictionaries with a breadth-first search of the relation
    relation_expansion(super_src_of)
    relation_expansion(super_tgt_of)
    relation_expansion(super_frames_of)
    
    ###############################################
    # build MetaNet corpus using the dictionaries #
    ###############################################
    with open("output/metanet_corpus.tsv", "w", encoding='utf-8', newline="") as  tsvfile:
        output_writer = csv.writer(tsvfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        output_writer.writerow(["#Source", "#Target", "#ConceptualMetaphor"])

        lost_count = 0

        # for each conceptual metaphor in MetaNet
        for met in metaphors_dict.keys():

            ##########################
            # SOURCE CONCEPT SEARCH  #
            ##########################

            # list containing this metaphor and the subsumers found in a breadth-first search of "source subcase of" relations
            met_subsumers_list = [met] + list(super_src_of[met].keys())
            # source concept candidates
            source_candidates = dict()

            # RULE 5) apply rules 1 to 4 to each metaphor in met_subsumers_list
            while met_subsumers_list:
                
                # retrieve next metaphor
                cur_met = met_subsumers_list.pop(0)

                # RULE 1) src/tgt frame of cur_met
                if cur_met in metaphors_dict.keys():
                    update_distance_dict(source_candidates,
                                         metaphors_dict[cur_met]["source"],
                                         super_src_of[met][cur_met] if cur_met != met else 0)
            
                # RULE 2) derive src/tgt from cur_met's name, assuming that it has the form "TARGET [BE] SOURCE"
                update_distance_dict(source_candidates,
                                     split_conceptual_metaphor(cur_met)[SPLITTED_SOURCE_INDEX],
                                     super_src_of[met][cur_met] if cur_met != met else 0)
                    
                # RULE 3) use the src/tgt frames "relevant framenet frames" as src/tgt
                if cur_met in metaphors_dict.keys() \
                            and metaphors_dict[cur_met]["source"] in fn_frames_for.keys():
                    fn_list = fn_frames_for[metaphors_dict[cur_met]["source"]]
                    while fn_list:
                        update_distance_dict(source_candidates,
                                             fn_list.pop(0),
                                             super_src_of[met][cur_met] if cur_met != met else 0)
                
                # RULE 4) breadth-first search of “is subcase of” related frames, starting from src/tgt frame of cur_met
                if cur_met in metaphors_dict.keys():
                    curmet_src = metaphors_dict[cur_met]["source"]

                    if curmet_src in super_frames_of.keys():
                        frame_subsumers_list = list(super_frames_of[curmet_src].keys())
                        while frame_subsumers_list:
                            cur_frame = frame_subsumers_list.pop(0)
                            update_distance_dict(source_candidates,
                                                 cur_frame,
                                                 (super_src_of[met][cur_met] if cur_met != met else 0)
                                                  + super_frames_of[curmet_src][cur_frame])

            ##########################
            # TARGET CONCEPT SEARCH  #
            ##########################

            # list containing this metaphor and the subsumers found in a breadth-first search of "target subcase of" relations
            met_subsumers_list = [met] + list(super_tgt_of[met].keys())
            # target concept candidates
            target_candidates = dict()

            # RULE 5) apply rules 1 to 4 to each metaphor in met_subsumers_list
            while met_subsumers_list:
                
                # retrieve next metaphor
                cur_met = met_subsumers_list.pop(0)

                # RULE 1) src/tgt frame of cur_met
                if cur_met in metaphors_dict.keys():
                    update_distance_dict(target_candidates,
                                         metaphors_dict[cur_met]["target"],
                                         super_tgt_of[met][cur_met] if cur_met != met else 0)
            
                # RULE 2) derive src/tgt from cur_met's name, assuming that it has the form "TARGET [BE] SOURCE"
                update_distance_dict(target_candidates,
                                     split_conceptual_metaphor(cur_met)[SPLITTED_TARGET_INDEX],
                                     super_tgt_of[met][cur_met] if cur_met != met else 0)
                    
                # RULE 3) use the src/tgt frames "relevant framenet frames" as src/tgt
                if cur_met in metaphors_dict.keys() \
                            and metaphors_dict[cur_met]["target"] in fn_frames_for.keys():
                    fn_list = fn_frames_for[metaphors_dict[cur_met]["target"]]
                    while fn_list:
                        update_distance_dict(target_candidates,
                                             fn_list.pop(0),
                                             super_tgt_of[met][cur_met] if cur_met != met else 0)
                
                # RULE 4) breadth-first search of “is subcase of” related frames, starting from src/tgt frame of cur_met
                if cur_met in metaphors_dict.keys():
                    curmet_tgt = metaphors_dict[cur_met]["target"]

                    if curmet_tgt in super_frames_of.keys():
                        frame_subsumers_list = list(super_frames_of[curmet_tgt].keys())
                        while frame_subsumers_list:
                            cur_frame = frame_subsumers_list.pop(0)
                            update_distance_dict(target_candidates,
                                                 cur_frame,
                                                 (super_tgt_of[met][cur_met] if cur_met != met else 0)
                                                  + super_frames_of[curmet_tgt][cur_frame])
            
            ###############################
            # OUTPUT FOR CURRENT METAPHOR #
            ###############################
                    
            # if there are both some source and some target candidates              
            if source_candidates and target_candidates:
                # build a list of source candidates sorted by relational distance
                source_cand_list = [x[0] for x in sorted(source_candidates.items(), key = lambda kv : kv[1])]
                # build a list of target candidates sorted by relational distance
                target_cand_list = [x[0] for x in sorted(target_candidates.items(), key = lambda kv : kv[1])]

                # output the two candidate concepts lists for met
                output_writer.writerow([json.dumps(source_cand_list), json.dumps(target_cand_list), met])

            # else print a warning
            else:
                lost_count += 1
                print(f"WARNING: Conceptual Metaphor {met} cannot be represented")




if __name__ == '__main__': main()



