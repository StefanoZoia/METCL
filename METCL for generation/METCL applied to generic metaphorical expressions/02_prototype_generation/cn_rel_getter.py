# This script saves the relevant information from ConceptNet
# for both the source and target concepts in the corpus
# in a file under the 'relations' directory

import requests
import csv
import prototyper_config as cfg
import os
import time


REQ_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/117.0',
'Accept': 'application/json'}

# if the response is not a json, waits and retries
def get_json_or_retry(query):
    json = None #returned variable
    limit = 10  #max n of retries
    delay = 1   #n of seconds to wait between retries
    retried = 0
    got_it = False
    while not got_it and retried < limit:
        try:
            json = requests.get(query, REQ_HEADERS).json()
            got_it = True
        except:
            time.sleep(delay)
            got_it = False
            retried += 1
            print(f'>>>>WARNING: failed attempt {retried}/{limit}')
    return json
    


# return the most related concepts according to CN API
def get_related_dict(concept):
    related_dict = dict()
    obj = get_json_or_retry(f'https://api.conceptnet.io/related{concept}?filter=/c/en&limit=1000')
    for related_concept in obj['related']:
        related_dict[related_concept['@id']] = related_concept['weight']
    return related_dict


# interface with the ConceptNet Web API to get the necessary information
# and should 
def get_cn_rel_with_score(word):
    this_concept = f'/c/{cfg.lang_code}/{word}'
    print(this_concept)
    obj = get_json_or_retry(f'http://api.conceptnet.io/query?node={this_concept}&other=/c/{cfg.lang_code}')

    # if there is an error, return False (e.g. the word might not be present in ConceptNet)
    if ('error' in obj.keys()):
        return False

    # use the 'related' concepts as a cache
    relatedness_score = get_related_dict(this_concept)
    to_be_continued = True
    # save the retrieved information in a csv file
    with open(f'data/relations/{word}.tsv', 'w', newline='', encoding='utf-8') as rel_file:
        writer = csv.writer(rel_file, delimiter='\t', quotechar='"')
        while to_be_continued:

            # save relation and target in the relations dict
            for edge in obj['edges']:
                start = edge["start"]["term"]
                rel = edge["rel"]["label"]
                end = edge["end"]["term"]
                rel_weight = edge["weight"]
                # get ConceptNet Numberbatch relatedness score
                nb_weight = 0
                other = start if end == this_concept else end
                if other in relatedness_score:
                    nb_weight = relatedness_score[other]
                else:
                    # if the relatedness to the other concept is not known, request it
                    print(this_concept, other)
                    nb_weight = get_json_or_retry(f'http://api.conceptnet.io/relatedness?node1={this_concept}&node2={other}')
                    nb_weight = nb_weight['value'] if nb_weight is not None else None
                    relatedness_score[other] = nb_weight    # cache the score in case the concept has more than one relation
                # write the relation on the file
                writer.writerow([start, rel, end, rel_weight, nb_weight])

            # if the output is paginated by edges and it's not ended, continue on the next page
            to_be_continued = 'view' in obj.keys() and (obj['view']['paginatedProperty'] == 'edges') and ('nextPage' in obj['view'].keys())
            if to_be_continued:
                nextURL = f"http://api.conceptnet.io{obj['view']['nextPage']}"
                print(nextURL)
                obj = get_json_or_retry(nextURL)
    # if there is no error, return True
    return True


# check if a word is already represented in /relations
def isRepresented(word):
    filename = f'{word}.tsv'
    path = 'data/relations'
    return filename in os.listdir(path)


def main():
    with open(cfg.INTERMEDIATE_FILE, 'w', newline='', encoding='utf-8') as outputfile:
        with open(cfg.CORPUS_FILE, newline='', encoding='utf-8') as feed:
            writer = csv.writer(outputfile, delimiter='\t', quotechar='"')
            reader = csv.reader(feed, delimiter='\t', quotechar='"')
            next(reader, None)  # skip the headers

            # for each metaphor in the corpus file
            for row in reader:
                source = row[0].lower().replace(" ", "_").replace("-", "_")
                target = row[1].lower().replace(" ", "_").replace("-", "_")

                # if source was already represented or can be represented now
                if isRepresented(source) or get_cn_rel_with_score(source):
                    # if target was already represented or can be represented now
                    if isRepresented(target) or get_cn_rel_with_score(target):
                        # copy row on the output file
                        writer.writerow(row)


if __name__ == "__main__": main()