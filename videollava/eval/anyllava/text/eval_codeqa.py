import os
import argparse
import json
import re


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()


def eval_single(result_file):
    results = [json.loads(line) for line in open(result_file)]

    total = len(results)
    right = 0
    for result in results:
        pred = result['pred']
        if pred == "" or pred == " ":
            continue
        pred = pred[1:] if len(pred) > 0 and pred[0] == ' ' else pred
        ground_truth = result['label']
        if pred.split(".")[0].upper() == ground_truth.split(".")[0].upper() or pred.split(".")[0].upper() in ground_truth.split(".")[0].upper() or ground_truth.split(".")[0].upper() in pred.split(".")[0].upper():
            right += 1
        elif pred.split(".")[0].lower == "zero" and ground_truth.split(".")[0] =="0" or pred.split(".")[0] == "0" and ground_truth.split(".")[0].lower == "zero":
            right += 1
        elif pred.split(".")[0].lower == "one" and ground_truth.split(".")[0] =="1" or pred.split(".")[0] == "1" and ground_truth.split(".")[0].lower == "one":
            right += 1
        elif pred.split(".")[0].lower == "two" and ground_truth.split(".")[0] =="2" or pred.split(".")[0] == "2" and ground_truth.split(".")[0].lower == "two":
            right += 1
        elif pred.split(".")[0].lower == "three" and ground_truth.split(".")[0] =="3" or pred.split(".")[0] == "3" and ground_truth.split(".")[0].lower == "three":
            right += 1
        elif pred.split(".")[0].lower == "four" and ground_truth.split(".")[0] =="4" or pred.split(".")[0] == "4" and ground_truth.split(".")[0].lower == "four":
            right += 1
        elif pred.split(".")[0].lower == "five" and ground_truth.split(".")[0] =="5" or pred.split(".")[0] == "5" and ground_truth.split(".")[0].lower == "five":
            right += 1
        elif pred.split(".")[0].lower == "six" and ground_truth.split(".")[0] =="6" or pred.split(".")[0] == "6" and ground_truth.split(".")[0].lower == "six":
            right += 1
        elif pred.split(".")[0].lower == "seven" and ground_truth.split(".")[0] =="7" or pred.split(".")[0] == "7" and ground_truth.split(".")[0].lower == "seven":
            right += 1
        elif pred.split(".")[0].lower == "eight" and ground_truth.split(".")[0] =="8" or pred.split(".")[0] == "8" and ground_truth.split(".")[0].lower == "eight":
            right += 1
        elif pred.split(".")[0].lower == "nine" and ground_truth.split(".")[0] =="9" or pred.split(".")[0] == "9" and ground_truth.split(".")[0].lower == "nine":
            right += 1
        else:
            continue

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.result_file)