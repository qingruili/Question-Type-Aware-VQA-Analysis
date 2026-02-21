# -*- coding: utf-8 -*-
# Analysis script for HW3: Neural Machine Translation
#
# Usage:
#   python analysis.py -i <hypothesis.out> \
#                      -r <reference.out>  \
#                      -s <source.txt>     \
#                      -o <output.csv>
#
# Example:
#   python analysis.py -i data/analysis/default_out.out \
#                      -r data/reference/dev.out         \
#                      -s data/input/dev.txt             \
#                      -o data/analysis/default_scores.csv
#
# CSV columns:
#   id        : sentence index, starting from 1
#   src_len   : number of words in the German source sentence
#   source    : German source sentence
#   reference : human reference English translation
#   hypothesis: model output (hypothesis being scored)
#   bleu      : sentence-level BLEU score (0-100)

import os
import sys
import csv
import optparse

import sacrebleu


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_file(path):
    """Load a text file and return a list of stripped lines."""
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]


def clean(sentence):
    """Remove stray special tokens that beam search sometimes outputs."""
    return sentence.replace('<sos>', '').replace('<eos>', '').strip()


def sentence_bleu(hypothesis, reference):
    """Compute sentence-level BLEU score. Returns a float 0-100."""
    result = sacrebleu.sentence_bleu(
        hypothesis, [reference], smooth_method='exp'
    )
    return round(result.score, 2)


# ─────────────────────────────────────────────
# Main: build and write CSV
# ─────────────────────────────────────────────

def build_csv(src_path, ref_path, hyp_path, out_path):
    """
    Load source, reference and hypothesis files, compute per-sentence
    BLEU, and write everything to a CSV file.
    """
    src = load_file(src_path)
    ref = load_file(ref_path)
    hyp = [clean(s) for s in load_file(hyp_path)]

    if not (len(src) == len(ref) == len(hyp)):
        raise ValueError(
            f"File length mismatch: src={len(src)}, "
            f"ref={len(ref)}, hyp={len(hyp)}"
        )

    n = len(src)
    rows = []
    for i in range(n):
        rows.append({
            'id':         i + 1,                   # 1-indexed
            'src_len':    len(src[i].split()),      # word count of German input
            'source':     src[i],                   # German source sentence
            'reference':  ref[i],                   # human reference translation
            'hypothesis': hyp[i],                   # model output
            'bleu':       sentence_bleu(hyp[i], ref[i]),
        })

    fieldnames = ['id', 'src_len', 'source', 'reference', 'hypothesis', 'bleu']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {n} rows to: {out_path}")

    # Also print a quick corpus-level BLEU as a sanity check
    corpus = sacrebleu.corpus_bleu(
        hyp, [ref], force=True, lowercase=True, tokenize='none'
    )
    print(f"Corpus BLEU (sanity check): {corpus.score:.2f}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = optparse.OptionParser(
        usage="usage: %prog -i HYP -r REF -s SRC -o OUT"
    )
    parser.add_option(
        '-i', '--input', dest='hyp',
        help='Hypothesis (model output) .out file'
    )
    parser.add_option(
        '-r', '--reference', dest='ref',
        default=os.path.join('data', 'reference', 'dev.out'),
        help='Reference .out file [default: data/reference/dev.out]'
    )
    parser.add_option(
        '-s', '--source', dest='src',
        default=os.path.join('data', 'input', 'dev.txt'),
        help='Source (German) .txt file [default: data/input/dev.txt]'
    )
    parser.add_option(
        '-o', '--output', dest='out',
        help='Output CSV file path'
    )

    (opts, _) = parser.parse_args()

    if not opts.hyp:
        parser.error("Please provide a hypothesis file with -i")
    if not opts.out:
        # Default: same location as hypothesis file, .csv extension
        opts.out = os.path.splitext(opts.hyp)[0] + '_scores.csv'
        print(f"No output path given, defaulting to: {opts.out}")

    build_csv(opts.src, opts.ref, opts.hyp, opts.out)
