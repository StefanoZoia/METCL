import csv
import statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats
import os
import json

ANSWERS_FOLDER = "data/lab_session_answers"
FIRST_EVAL_INDEX = 6
LAST_EVAL_INDEX = 15

OUTPUT_FILE_NAME = "output/eval_conceptual"

# reads a google form csv answers
def read_form(answer_file):
    overall_eval = list()       # list of overall points assigned in this form
    with open(answer_file, encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        questions = next(reader)    # read the questions separately

        # the rows of the csv have shape (timestamp, [answer for question N]*)
        for row in reader:
            for i in range(FIRST_EVAL_INDEX, LAST_EVAL_INDEX+1):
                if row[i] != '':
                    overall_eval.append(int(row[i]))
    return overall_eval

# read the answers to the forms and compute relevant statistics
def main():
    overall_eval = list()       # list of overall points assigned in all forms
    
    for file in os.listdir(ANSWERS_FOLDER):
        file_path = os.path.join(ANSWERS_FOLDER, file)
        overall_eval.extend(read_form(file_path))
    
    with open('data/webapp_evaluation.csv', encoding='utf-8', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)       # skip the header

        # the rows of the csv have shape (metaphor_id, session_id, value)
        for row in reader:
            overall_eval.append(int(row[2]))

    print('Evaluation results:\n')

    print(f'N. of evaluation: {len(overall_eval)}')
    mean = statistics.mean(overall_eval)
    print(f'Overall mean: {mean:.2f}')
    median = statistics.median(overall_eval)
    print(f'Overall median: {median}')
    mode = statistics.mode(overall_eval)
    print(f'Overall mode: {mode}')
    std_dev = statistics.stdev(overall_eval)
    print(f'Overall st. dev.: {std_dev:.2f}')
    print(f'Overall skewness: {stats.skew(overall_eval):.2f}')

    freq, bins, patches = plt.hist(overall_eval, bins=np.arange(0.5, 11.5, 1))
    for p in patches:
        p.set_edgecolor('black')

    # add to each patch a label with its height
    bin_centers = np.diff(bins)*0.5 + bins[:-1]

    n = 0
    for fr, x, patch in zip(freq, bin_centers, patches):
        height = int(freq[n])
        plt.annotate("{}".format(height),
                    xy = (x, height),             # top left corner of the histogram bar
                    xytext = (0,0.2),             # offsetting label position above its bar
                    textcoords = "offset points", # Offset (in points) from the *xy* value
                    ha = 'center', va = 'bottom'
                    )
        n = n+1

    plt.title('Overall evaluation')

    plt.yticks([])
    plt.xlabel('evaluation score')
    plt.xticks(range(1, 11))
    plt.ylabel('n. of answers')

    line = plt.axvline(mean, color='red')
    extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
    plt.legend([line, extra, extra], [f'mean = {mean:.2f}', f'std. dev. = {std_dev:.2f}', f'median = {int(median)}'])
    plt.savefig(f"{OUTPUT_FILE_NAME}.pdf")
    plt.show()

    # Serializing json
    json_object = json.dumps(overall_eval)
    # Writing to overall.json
    with open("output/overall.json", "w") as outfile:
        outfile.write(json_object)

if __name__ == '__main__' : main()