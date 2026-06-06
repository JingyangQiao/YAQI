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
        if pred.split(".")[0] == ground_truth.split(".")[0] or pred.split(".")[0] in ground_truth.split(".")[0] or ground_truth.split(".")[0] in pred.split(".")[0]:
            right += 1

    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total))

if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.result_file)