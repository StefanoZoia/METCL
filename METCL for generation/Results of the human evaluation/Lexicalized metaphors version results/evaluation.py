import csv
import statistics
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy import stats
import json

# CHANGE HERE to choose the data to analyze
CORPORA = ["metanet_results", "nn450_results"]
# CORPORA = ["metanet_results"]
# CORPORA = ["nn450_results"]

OUTPUT_FILE_NAME = "expressions" if len(CORPORA) == 2 else CORPORA[0]

# reads a google form csv answers and computes relevant statistics
def main():
    overall_eval = list()       # list of overall points assigned to the semantic extraction
    for corpus in CORPORA:
        with open(f"data/{corpus}.jsonl", encoding='utf-8', newline='') as file:
            # the rows of the csv have shape (timestamp, [answer for question N]*)
            for line in file:
                row = json.loads(line)
                for eval in row["metaphors"].values():
                    overall_eval.append(int(eval))

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
    plt.savefig(f"output/eval_{OUTPUT_FILE_NAME}.pdf")
    plt.show()

    # Serializing json
    json_object = json.dumps(overall_eval)
    # Writing to overall.json
    with open("output/overall.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__' : main()