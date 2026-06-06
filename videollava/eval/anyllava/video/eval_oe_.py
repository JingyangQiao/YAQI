import os
import argparse
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, default='merge.jsonl')
    parser.add_argument('--output-dir', type=str)
    return parser.parse_args()

def eval_single(result_file, output_dir=None):
    results = [json.loads(line) for line in open(result_file)]
    total = len(results)
    all_preds = [r['pred'].lstrip() for r in results]
    all_labels = [r['label'] for r in results]

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device=device)

    pred_emb = model.encode(all_preds, batch_size=64, device=device, show_progress_bar=True)
    label_emb = model.encode(all_labels, batch_size=64, device=device, show_progress_bar=True)
    sims = cosine_similarity(pred_emb, label_emb).diagonal()
    right = np.clip(sims, 0, None).sum()

    result_str = 'Samples: {}\nAccuracy: {:.2f}%\n'.format(total, 100. * right / total)
    print(result_str)
    if output_dir is not None:
        output_file = os.path.join(output_dir, 'Result.text')
        with open(output_file, 'w') as f:
            f.write(result_str)

if __name__ == "__main__":
    args = get_args()
    if args.result_file is not None:
        eval_single(args.result_file, args.output_dir)