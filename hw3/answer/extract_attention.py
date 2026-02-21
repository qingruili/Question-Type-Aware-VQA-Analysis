# -*- coding: utf-8 -*-
# extract_attention.py
#
# Runs greedy decoding on a single sentence and saves the attention
# weight matrix (alpha) plus the source/target token lists to a .npz
# file, ready for heatmap visualisation in the notebook.
#
# Usage:
#   python extract_attention.py \
#       -m data/seq2seq_E049.pt \
#       -s "Also lassen Sie uns einen Blick auf die Zahlen werfen ." \
#       -o data/analysis/attention_example.npz

import os
import sys
import optparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
from neuralmt import (
    Seq2Seq, hp,
    tokenise_de, tokenise_en,
    greedyDecoder,
)


def extract(model, german_sentence, max_len=50):
    """
    Run greedy decoding on a single German sentence and return:
      - src_tokens  : list of German tokens (with <sos>/<eos>)
      - tgt_tokens  : list of generated English tokens
      - alpha_matrix: numpy array of shape (n_tgt, n_src)
                      alpha_matrix[t, i] = attention weight on source
                      word i when generating target word t
    """
    model.eval()
    device = hp.device

    # ── Tokenise & encode source ───────────────────────────────
    src_lex  = model.params['srcLex']
    tgt_lex  = model.params['tgtLex']
    tgt_itos = tgt_lex.get_itos()          # index → string

    tokens = tokenise_de(german_sentence.lower())
    indices = [src_lex['<sos>']] + [src_lex[t] for t in tokens] + [src_lex['<eos>']]
    src_tensor = torch.tensor(indices).unsqueeze(1).to(device)  # (seq, 1)

    # ── Encode ────────────────────────────────────────────────
    with torch.no_grad():
        encoder_out, encoder_hidden = model.encoder(src_tensor)
        outputs, alphas = greedyDecoder(
            model.decoder, encoder_out, encoder_hidden, max_len
        )

    # ── Decode output tokens ───────────────────────────────────
    pred_indices = outputs.topk(1)[1][:, 0, 0].tolist()   # (tgt_len,)
    tgt_tokens = []
    for idx in pred_indices:
        word = tgt_itos[int(idx)]
        if word == '<eos>':
            break
        tgt_tokens.append(word)

    # ── Trim alpha to actual output length ────────────────────
    # alphas shape from greedyDecoder: (batch=1, src_len, max_len)
    # We want (n_tgt, n_src)
    n_tgt = len(tgt_tokens)
    n_src = encoder_out.shape[0]           # includes <sos> and <eos>
    alpha_matrix = alphas[0, :n_src, :n_tgt].cpu().numpy()  # (n_src, n_tgt)
    alpha_matrix = alpha_matrix.T                            # (n_tgt, n_src)

    # ── Source tokens (include <sos> and <eos> to match encoder) ──
    src_tokens = ['<sos>'] + tokens + ['<eos>']

    return src_tokens, tgt_tokens, alpha_matrix


if __name__ == '__main__':
    parser = optparse.OptionParser(
        usage="usage: %prog -m MODEL -s SENTENCE -o OUTPUT"
    )
    parser.add_option(
        '-m', '--model', dest='model',
        default=os.path.join('data', 'seq2seq_E049.pt'),
        help='Path to model checkpoint [default: data/seq2seq_E049.pt]'
    )
    parser.add_option(
        '-s', '--sentence', dest='sentence',
        default="Also lassen Sie uns einen Blick auf die Zahlen werfen .",
        help='German sentence to translate and extract attention from'
    )
    parser.add_option(
        '-o', '--output', dest='output',
        default=os.path.join('data', 'analysis', 'attention_example.npz'),
        help='Output .npz file path'
    )
    (opts, _) = parser.parse_args()

    # Load model
    print(f"Loading model from: {opts.model}")
    model = Seq2Seq(build=False)
    model.load(opts.model)
    model.to(hp.device)
    model.eval()

    # Extract attention
    print(f"Source sentence: {opts.sentence}")
    src_tokens, tgt_tokens, alpha_matrix = extract(model, opts.sentence)

    print(f"Source tokens ({len(src_tokens)}): {src_tokens}")
    print(f"Target tokens ({len(tgt_tokens)}): {tgt_tokens}")
    print(f"Alpha matrix shape: {alpha_matrix.shape}  (tgt × src)")
    print(f"\nAlpha matrix (rows=target, cols=source):")
    for t, row in zip(tgt_tokens, alpha_matrix):
        weights = '  '.join(f'{w:.2f}' for w in row)
        print(f"  {t:12s} | {weights}")

    # Save
    np.savez(
        opts.output,
        src_tokens=np.array(src_tokens),
        tgt_tokens=np.array(tgt_tokens),
        alpha=alpha_matrix,
    )
    print(f"\nSaved to: {opts.output}")
