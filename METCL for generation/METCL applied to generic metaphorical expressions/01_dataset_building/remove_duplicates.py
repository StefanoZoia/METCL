import pprint
import csv
import os

# using different sources may lead to duplicates: this script removes them

def rm_dup(filename):
    print(f'\nremoving duplicates from {filename}')
    dup_count = 0

    # store the list of examples for each source-target pair
    source_target = dict()

    with open(filename, encoding='utf-8', newline='') as corpusfile:
        reader = csv.reader(corpusfile, delimiter='\t', quotechar='"')

        for row in reader:
            key = f'{row[0]}_{row[1]}'.lower()
            if key not in source_target:
                # new source-target pair
                source_target[key] = [row]
                #writer.writerow(row)
            else:
                if row not in source_target[key]:
                    # new example
                    source_target[key].append(row)
                    #writer.writerow(row)
                else:
                    # duplicate
                    dup_count += 1

    # use a tmp file to store the non duplicated rows 
    tmpfile = 'tmp.txt'
    met_count = 0
    with open(tmpfile, 'w', encoding='utf-8', newline='') as tmp:
        writer = csv.writer(tmp, delimiter='\t', quotechar='"')
        for couple in sorted(source_target):
            for row in source_target[couple]:
                writer.writerow(row)
                met_count += 1

    print(f'>>> removed {dup_count} duplicated rows')
    print(f'>>> dataset contains {len(source_target)} distinct source-target pairs')
    print(f'>>> dataset contains {met_count} metaphorical sentences')

    # replace the input file with the tmp file
    os.remove(filename)
    os.rename(tmpfile, filename)

if __name__ == '__main__':
    #rm_dup('all_nn.tsv') 
    #rm_dup('all_na.tsv') 
    #rm_dup('all_an.tsv') 
    rm_dup('metanet_corpus.tsv') 