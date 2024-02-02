import torch as t
import pandas as pd
import os
from generate_acts import load_llama
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))


t.set_grad_enabled(False)

def get_few_shot_accuracy(datasets, model_size, n_shots=5, calibrated=True, device='cpu', layers=[]):
    """Compute the few-shot accuracy of the model on the given datasets.
    Returns a list of dictionaries with experimental results, namely:
    * The dataset used.
    * The number of shots in the few shot prompt.
    * The few shot prompt used.
    * The accuracy of the model.
    * The calibration constant, if calibrated=True.
    """

    tokenizer, model = load_llama(model_size, device)

    outs = []
    for dataset in datasets:
        out = {
            'n_shots' : n_shots,
            'dataset': dataset
            }

        # prepare data and prompt
        data_directory = os.path.join(ROOT, 'datasets', f"{dataset}.csv")
        df = pd.read_csv(data_directory)
        shots = df.sample(n_shots)
        queries = df.drop(shots.index)

        prompt = ''
        for _, shot in tqdm(shots.iterrows(), desc=f'Processing {dataset}'):
            prompt += f'{shot["statement"]} '
            if bool(shot['label']):
                prompt += 'TRUE\n'
            else:
                prompt += 'FALSE\n'

        out['shots'] = shots['statement'].tolist()
        out['prompt'] = prompt

        # cache activations over the prompt for reuse
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        out['acts'] = defaultdict(dict)
        out['prob_true'] = {}
        out['prob_false'] = {}
        # get completions and evaluate accuracy
        true_idx, false_idx = tokenizer.encode('TRUE')[1], tokenizer.encode('FALSE')[1]
        p_trues, p_falses = [], []
        for query_ind, query in tqdm(queries.iterrows()):
            input_ids = tokenizer.encode(prompt + query['statement'], return_tensors='pt').to(device)
            outputs = model(input_ids, past_key_values=past_key_values, output_hidden_states=True)
            probs = outputs.logits[0, -1, :].softmax(-1)
            prob_true = probs[true_idx].item()
            prob_false = probs[false_idx].item()
            p_trues.append(prob_true), p_falses.append(prob_false)
            out['prob_true'][query_ind] = prob_true
            out['prob_false'][query_ind] = prob_false
            for layer in layers:
                # hidden states: layer (+1 bc 1st is emb), batch, token (last), dim
                out['acts'][layer][query_ind] = outputs.hidden_states[layer+1][0][-1][:].to(device='cpu').tolist() 

        # if calibrated, compute calibration constant
        if calibrated:
            diffs = [p_true - p_false for p_true, p_false in zip(p_trues, p_falses)]
            gamma = sorted(diffs)[len(diffs) // 2]
            out['gamma'] = gamma
        else:
            gamma = 0

        # get predicted labels
        predicted_labels = [p_true - p_false > gamma for p_true, p_false in zip(p_trues, p_falses)]
        predicted_labels = t.Tensor(predicted_labels).to(device).float()
        true_labels = t.Tensor(queries['label'].values).to(device)

        acc = (predicted_labels == true_labels).float().mean().item()
        out['acc'] = acc

        outs.append(out)

    return outs

if __name__ == '__main__':
    """
    Compute the few-shot accuracy of the model on the given datasets and save results.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', help='datasets to evaluate on')
    parser.add_argument('--model_size', type=str, default='13B', help='model size to evaluate')
    parser.add_argument('--n_shots', type=int, default=5, help='number of shots to use')
    parser.add_argument('--uncalibrated', action='store_true', default=False, help='set flag if using uncalibrated few shot')
    parser.add_argument('--device', default='cuda:0', help='device to use')
    parser.add_argument('--layers', nargs='*', type=int, default=[], help='layers to save acts for')
    args = parser.parse_args()

    out = get_few_shot_accuracy(args.datasets, args.model_size, args.n_shots, not args.uncalibrated, args.device, args.layers)

    # save results
    fp = os.path.join(ROOT, 'experimental_outputs', f'few_shot_results_{args.model_size}.json')
    if os.path.exists(fp):
        with open(fp, 'r') as f:
            data = json.load(f)
    else:
        data =[]
    data.extend(out)
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)
        
    # # save results
    # fp = os.path.join(ROOT, 'experimental_outputs', 'few_shot_results.json')
    # if os.path.exists(fp):
    #     with open(fp) as f:
    #         data = json.load(f)
    # else:
    #     data = {}
    # data.update(out) # update any existing data with data from this run
    # with open(fp, 'w') as f:
    #     json.dump(data, f, indent=4)