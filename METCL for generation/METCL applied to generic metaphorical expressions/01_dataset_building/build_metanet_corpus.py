import csv     

def build_metanet_corpus():
    data_folder = "data/metanet"
    mn_classes_path = f"{data_folder}/metanet_annotation.csv"

    metanet_sentences = []
    with open(mn_classes_path, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if line[4] == "-":
                continue
            target = line[3]
            source = line[2]
            sentence = line[1]
            metanet_sentences.append([source, target, sentence])
    
    output_file = "output/metanet_corpus.tsv"
    with open(output_file, 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f, delimiter="\t", quotechar='"')
        metanet_sentences.insert(0, ["#source", "#target", "#sentence"])
        writer.writerows(metanet_sentences)
        
def main():
    build_metanet_corpus()

if __name__ == "__main__":
    main()
