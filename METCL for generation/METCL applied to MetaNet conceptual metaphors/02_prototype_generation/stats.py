import os
import csv
import statistics
import pprint

REL_DIR = f'relations'

def main():
    with open('prototyped.tsv', encoding='utf-8', newline='') as corpusfile:
        reader = csv.reader(corpusfile, delimiter='\t', quotechar='"')
        source_target = set()
        for row in reader:
            source_target.add(f'{row[0]}_{row[1]}'.lower())
        print(f'total distinct source-target pairs = {len(source_target)}')

    rels = []
    rel_counters = []
    rel_weights = []
    relatedness = []
    for filename in os.listdir(REL_DIR):
        with open(f'{REL_DIR}/{filename}', encoding='utf-8', newline='') as tsv_file:
            rel_counter = 0
            for row in csv.reader(tsv_file, delimiter='\t'):
                rel_counter += 1
                if row[1] not in rels:
                    rels.append(row[1])
                rel_weights.append(row[3])
                relatedness.append(float(row[4]))
            rel_counters.append(rel_counter)
        
    print(f'total files = {len(rel_counters)}')
    print(f'avg rel per file = {statistics.mean(rel_counters)}')
    print(f'min rel per file = {min(rel_counters)}')
    print(f'max rel per file = {max(rel_counters)}')
    print(f'min rel_weight = {min(rel_weights)}')
    print(f'max rel_weight = {max(rel_weights)}')
    print(f'min relatedness = {min(relatedness)}')
    print(f'max relatedness = {max(relatedness)}')
    
    
    with open('cn-represented.tsv', encoding='utf-8', newline='') as corpusfile:
        reader = csv.reader(corpusfile, delimiter='\t', quotechar='"')
        for row in reader:
            with open('prototyped.tsv', encoding='utf-8', newline='') as outfile:
                inner = csv.reader(outfile, delimiter='\t', quotechar='"')
                found = False
                for line in inner:
                    if row[0] == line[0] and row[1] == line[1] and row[2] == line[2]:
                        found = True
                if not found:
                    print(row)



if __name__ == "__main__": main()