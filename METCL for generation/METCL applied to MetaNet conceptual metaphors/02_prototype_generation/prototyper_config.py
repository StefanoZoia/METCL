######################################
# configuration for cn_rel_getter.py #
######################################

# path to the file containing the corpus
CORPUS_FILE = '../01_dataset_building/output/metanet_corpus.tsv'

# path to the output file of cn_rel_getter - input for prototyper
INTERMEDIATE_FILE = 'data/cn-represented.tsv'

# Language code for the data to be processed
lang_code = 'en'


###################################
# configuration for prototyper.py #
###################################

# path to the output file containing only the prototyped rows of the original corpus
OUT_FILE = 'output/prototyped.tsv'

# path to directory where prototypes of concepts will be saved
PROTOTYPE_DIR = 'output/prototypes'

# minimum and maximum number of typical properties in a prototype
MIN_PROP = 2
MAX_PROP = 8

# the weight of each feature (their sum must be 1)
feature_weight = {'rel': 0.4, 'rel_weight': 0.2, 'relatedness': 0.4}

# [0.5, 1] weight of properties inherited from root concepts: if 1, they are treated as direct properties
expansion_weight = 0.9

# list of relations identifying a root concept
root_rel = ['FormOf', 'DerivedFrom', 'IsA', 'PartOf', "MannerOf", "Synonym", "InstanceOf"]

# list of relations that imply a negation
neg_rel = ['NotIsA', 'NotUsedFor', 'NotCapableOf', 'NotHasProperty', 'ObstructedBy', 'Antonym', 'DistinctFrom']

# the fraction of the score to give to outgoing relations
outgoing_rel_score = {
    'IsA': 0.5,
    'PartOf': 0.5,
    'HasA': 0.25,
    'UsedFor': 0.75,
    'CapableOf': 0.75,
    'AtLocation': 0.5,
    'Causes': 0.5,
    'HasSubevent': 0.25,
    'HasFirstSubevent': 0.5,
    'HasLastSubevent': 0.5,
    'HasPrerequisite': 0.9,
    'HasProperty': 0.9,
    'MotivatedByGoal': 0.75,
    'ObstructedBy': 0.9,
    'CreatedBy': 0.9,
    'Antonym': 0.5,
    'DistinctFrom': 0.25,
    'SymbolOf': 1,
    'MannerOf': 0.5,
    'HasContext': 0.5,
    'CausesDesire': 0.25,
    'MadeOf': 0.25,
    'ReceivesAction': 0.25,
    'NotIsA': 0.5,
    'NotUsedFor': 0.75,
    'NotCapableOf': 0.75,
    'NotHasProperty': 0.9,
    'Entails': 0.75,
    'InstanceOf': 0.5
}

# the fraction of the score to give to incoming relations
incoming_rel_score = {
    'PartOf': 0.25,
    'HasA': 0.25,
    'UsedFor': 0.25,
    'CapableOf': 0.25,
    'HasPrerequisite': 0.25,
    'HasProperty': 0.25,
    'MotivatedByGoal': 0.25,
    'ObstructedBy': 0.9,
    'Antonym': 0.5,
    'DistinctFrom': 0.25,
    'CausesDesire': 0.25,
    'NotUsedFor': 0.25,
    'NotCapableOf': 0.25,
    'NotHasProperty': 0.25,
}
