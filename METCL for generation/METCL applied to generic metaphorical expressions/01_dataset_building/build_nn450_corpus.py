from remove_duplicates import rm_dup

def build_nn450_corpus():
    data_folder = "data/nn450"
    gordon_filepath = f"{data_folder}/gordon_nn.tsv"
    vuamc_filepath = f"{data_folder}/vuamc_nn.tsv"
    mensa_filepath = f"{data_folder}/mensa_nn.tsv"

    # assembling the nn450 corpus by merging the three resources
    corpus = []
    for path in [gordon_filepath, vuamc_filepath, mensa_filepath]:
        with open(path, "r", encoding="utf8") as f:
            corpus.extend(f.readlines())

    # storing results in the output file
    output_path = f"output/nn450_corpus.tsv"
    with open(output_path, "w", encoding="utf8") as f:
        f.writelines(corpus)

    # remove duplicates from output file
    rm_dup(output_path)

def main():
    build_nn450_corpus()

if __name__ == "__main__":
    main()