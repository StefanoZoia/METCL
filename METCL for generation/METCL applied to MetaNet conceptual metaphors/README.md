CoCoS system applied to Metaphors

A Python tool for dynamic generation of knowledge in Description Logics of
Typicality tested in the contexts of RaiPlay, WikiArt Emotions, and ArsMeteo
here applied to metaphor interpretation.

The tool is still under development, so no guarantee is provided on its functioning.

This work includes data from ConceptNet 5, which was compiled by the Commonsense Computing Initiative. ConceptNet 5 is freely available under the Creative Commons Attribution-ShareAlike license (CC BY SA 4.0) from https://conceptnet.io. The included data was created by contributors to Commonsense Computing projects, contributors to Wikimedia projects, Games with a Purpose, Princeton University's WordNet, DBPedia, OpenCyc, and Umbel.

---

# Module 1 - Dataset building

### Description


Our work relies on the conceptual metaphors in the MetaNet Wiki, available at this link until november 2024: https://metaphor.icsi.berkeley.edu/pub/en/index.php/MetaNet_Metaphor_Wiki. Web Archive version: https://web.archive.org/web/20220401075504/https://metaphor.icsi.berkeley.edu/pub/en/index.php/MetaNet_Metaphor_Wiki. In addition, it contains the manual annotations with "source" and "target" word.

The folder "01_dataset building" contains:
- metanet_classes.jsonl contains web-scraped informations about the MetaNet conceptual metaphor classes.
- metanet_annotations.csv contains manual annotations of the metaphorical expressions provided by MetaNet as lexicalization examples for the conceptual metaphors outlined in the wiki.
- metanet_filter.py generates the metanet_corpus.tsv file, containing the representations of the conceptual metaphors as triples, where the source and target are represented by the source and target frames (or the first subsuming frame that can be found in ConceptNet).

### Libraries

nltk library is used for tokenization and POS tagging

---

# Module 2 - Prototype generation

### Description

Generation of the prototypes involved in metaphors.

1. A first step involves the retrieval of relations from ConceptNet
2. The second step builds the prototypes based on the extracted relations

### Input

The tsv representation of the metaphor corpus, represented as triples (source, target, sentence).

### Output

Prototypes of each concept involved in some metaphor

### How to run

To run the retrieval of relation from ConceptNet: python3 metanet_cn_rel.py

- please, note that the execution may require some time, needed to query the ConceptNet Knowledge Base

To run the generation of prototypes of concepts: python3 metanet_prototyper.py

### Configuration

The configuration for both steps is specified in prototyper_config.py.
In particular, the configuration for metanet_prototyper.py specifies how to filter ConceptNet relations

### Libraries

nltk is used for lemmatization

---

# Module 3 - Conceptual combination

### Description

Generation of the prototypes of metaphorical concepts by conceptual combination.

1. A first step preprocesses the input
2. The second step performs the conceptual combination

### Input

1. The representation of rigid and typical properties of the concepts involved in the combinatio (we not consider all the proprerties as typical)
2. The tsv representation of the metaphors

### Output

Prototypes of the combined metaphorical concepts.

### How to run

To run the generation of the input file for METCL: python3 cocos_preprocessing.py

- the preprocessing step writes the prototypes of each couple of concepts involved in a metaphor into a suitable file

To run the conceptual combination with METCL: python3 cocos.py

- This runs the combination on all the files in a specified folder.
- To run on a single file, simply call the cocos() function. The function takes the file name as the first parameter and the maximum number of properties in the resulting concept as a second (optional) parameter.

### Configuration

The configuration for both steps is specified in cocos_config.py.
In particular, the configuration specifies where to search for the input

