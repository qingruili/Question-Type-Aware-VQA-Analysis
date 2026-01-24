from transformers import pipeline
import logging, os, csv
from difflib import SequenceMatcher

fill_mask = pipeline('fill-mask', model='distilbert-base-uncased')
mask = fill_mask.tokenizer.mask_token

def get_typo_locations(fh):
    tsv_f = csv.reader(fh, delimiter='\t')
    for line in tsv_f:
        yield (
            # line[0] contains the comma separated indices of typo words
            [int(i) for i in line[0].split(',')],
            # line[1] contains the space separated tokens of the sentence
            line[1].split()
        )

def edit_distance(a, b):
    """Return the exact Levenshtein edit distance between a and b."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    
    # Ensure a is the shorter string for space optimization
    if len(a) > len(b):
        a, b = b, a

    m, n = len(a), len(b)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,       # delete
                cur[j - 1] + 1,    # insert
                prev[j - 1] + cost # substitute
            )
        prev = cur

    return prev[n]


def select_correction(typo, predict, k=2, ed_weight=0.45, sim_weight=0.45, lm_weight=0.1):
    """
    Select the best correction using a weighted combination of:
    - SequenceMatcher similarity (sim_weight)
    - Transformer/LM score (lm_weight)
    
    Args:
        typo: The misspelled word
        predict: List of predictions with 'token_str' and 'score'
        k: Maximum edit distance allowed
        sim_weight: Weight for SequenceMatcher similarity (0-1)
        lm_weight: Weight for transformer score (0-1)
    
    Returns:
        Best correction string
    """
    typo_l = typo.lower()

    # Normalize candidates: (cand_str, lm_score)
    cands = []
    for p in predict:
        cand = p["token_str"].strip()
        if cand:
            cands.append((cand, float(p.get("score", 0.0))))

    if not cands:
        return typo

    # Helper: SequenceMatcher ratio
    def sm_ratio(c):
        return SequenceMatcher(None, typo_l, c.lower()).ratio()

    # Filter candidates: edit distance >= 1 (must be different) and <= k
    valid_cands = []
    for c, s in cands:
        ed = edit_distance(typo_l, c.lower())
        if 1 <= ed <= k:
            valid_cands.append((c, s, ed))
    
    # Fallback: if no valid candidates, use all candidates with ed >= 1
    if not valid_cands:
        for c, s in cands:
            ed = edit_distance(typo_l, c.lower())
            if ed >= 1:
                valid_cands.append((c, s, ed))
    
    # If still no candidates (all are identical to typo), return typo
    if not valid_cands:
        return typo

    # Calculate combined score for each candidate
    def combined_score(cand_tuple):
        c, lm_score, ed = cand_tuple
        similarity = sm_ratio(c)
        
        # Normalize edit distance to 0-1 scale (closer = higher score)
        # Using 1 / (1 + ed) so smaller distance = higher score
        ed_score = 1 / ed
        
        # Combine similarity and edit distance into one similarity measure
        
        # Final weighted score
        return (ed_weight * ed_score)+(sim_weight * similarity ) + (lm_weight * lm_score)

    # Select best candidate by combined score
    best = max(valid_cands, key=combined_score)[0]

    # Preserve capitalization
    if typo and typo[0].isupper():
        best = best.capitalize()

    return best


def spellchk(fh):
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent
        for i in locations:
            # predict top_k replacements only for the typo word at index i
            predict = fill_mask(
                " ".join([ sent[j] if j != i else mask for j in range(len(sent)) ]), 
                top_k=200
            )
            logging.info(predict)
            spellchk_sent[i] = select_correction(sent[i], predict)
        yield(locations, spellchk_sent)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", 
                            dest="input", 
                            default=os.path.join('data', 'input', 'dev.tsv'), 
                            help="file to segment")
    argparser.add_argument("-l", "--logfile", 
                            dest="logfile", 
                            default=None, 
                            help="log file for debugging")
    opts = argparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))
