DESCRIPTION

This folder contains the code and the results of the automatic evaluation for the classification task briefly described below.

METAPHOR CLASSIFICATION TASK

Given a metaphorical sentence and the annotation of the source and target concepts inside the sentence, 
find the conceptual metaphor in MetaNet (or MetaNet Extended) that best fits the sentence, or return the residual class (OTHER) 
if none of the conceptual metaphors fit.

In this directory, you find the code to run the metaphor classification with the LLMs freely available on HuggingFace.
The experiments with GPT4o, DeepSeek-R1 and Qwen2.5-Max were executed by manually prompting the models.

CONTENTS OF THIS DIRECTORY

data - contains the datasets, the annotations and the results of METCL generation used to perform the metaphor classification task
zero_shot_classify.py - runs the classification task using the zero-shot classification pipeline from HuggingFace
zeroshot-out - contains the output of the classification using the zero-shot classification pipeline from HuggingFace
prompt_classify.py - runs the classification task using the text generation pipeline from HuggingFace
prompt-zeroshot-out - contains the output of the classification using the text generation pipeline from HuggingFace in a zero shot setting
prompt-fewshot-out - contains the output of the classification using the text generation pipeline from HuggingFace in a few shot setting
nn-450-stats.py - computes the performance measures of a classification of the sentences in the NN450 dataset
mn-examples-stats.py - computes the performance measures of a classification of the sentences in the MetaNet dataset

HOW TO RUN THE LLM CLASSIFICATION

You can decide to run the classification in a zero-shot setting using the "zero_shot_classify.py" script, or in a few-shot setting
using the "prompt_classify.py" script.

For example, for the zero-shot classification, you can find the configuration of the dedicated script in the code,
under the "Parameters Setting" comment. The parameters defined there specify:
- MODEL: the language model to run from the HuggingFace library (https://huggingface.co/models?pipeline_tag=zero-shot-classification)
- H_TEMPLATE: the instruction given to the model. Leave unchanged if you want to repeat the experiment.
- OUT_PATH: the relative path for the output file
- DATASET: if you want to classify the sentences from NN450 or from MetaNet
- ENRICH: True if you want to use the classes from MetaNet Extended, False if you want to use the original MetaNet classes
- BATCH_START and BATCH_END: the indexes of the sentences to classify. The default values are 0 and 450 for NN450; 0 and 853 for MetaNet examples.

Run using the command:
python3 zero_shot_classify.py

For the few-shot classification, the H_TEMPLATE parameter id substituted by:
- TEMPLATE: the template of the prompt given to the model. Leave unchanged if you want to repeat the experiment.
The other parameters are used as in the zero-shot setting, except for:
- MODEL: The suitable language models tu use can be found in the HuggingFace library (https://huggingface.co/models?pipeline_tag=text-generation).
- TOKEN: your HuggingFace token (required to access some models)

In this case, run using the command:
python3 prompt_classify.py




