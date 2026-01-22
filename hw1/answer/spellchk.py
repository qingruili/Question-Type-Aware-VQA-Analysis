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

#def select_correction(typo, predict):
    # return the most likely prediction for the mask token
    #return predict[0]['token_str']
    #return max(predict, key=lambda p: SequenceMatcher(None, typo.lower(), p['token_str']).ratio())['token_str']
'''
def select_correction(typo, predict):
    # Score by similarity minus length penalty
    def score(p):
        sim = SequenceMatcher(None, typo.lower(), p['token_str']).ratio()
        len_penalty = abs(len(typo) - len(p['token_str'])) * 0.05
        return sim - len_penalty
    
    best = max(predict, key=score)['token_str']
    
    # Preserve capitalization (for start of sentence, after quotes "", after dashes --)
    if typo[0].isupper():
        best = best.capitalize()
    
    return best
'''

def edit_distance_leq_k(a, b, k=2):
    """Return True if Levenshtein edit distance(a,b) <= k."""
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if abs(len(a) - len(b)) > k:
        return False

    if len(a) > len(b):
        a, b = b, a

    m, n = len(a), len(b)
    prev = list(range(n + 1))

    for i in range(1, m + 1):
        cur = [i] + [0] * n
        # Track row minimum for early exit
        row_min = cur[0]
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # delete
                cur[j - 1] + 1,   # insert
                prev[j - 1] + cost# substitute
            )
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > k:
            return False
        prev = cur

    return prev[n] <= k

def select_correction(typo, predict, k=2):
    """
    A: best_sim = max SequenceMatcher similarity to typo
    B: best_edlm = among candidates with edit distance <= k, pick highest LM score
                 (fallback: if none within k, pick highest LM score overall)

    Final choice: whichever of (best_sim, best_edlm) is MORE similar to typo
                  using (1) smaller edit distance, then (2) higher SequenceMatcher.
    """
    typo_l = typo.lower()

    # normalize candidates: (cand_str, lm_score)
    cands = []
    for p in predict:
        cand = p["token_str"].strip()
        if cand:
            cands.append((cand, float(p.get("score", 0.0))))

    if not cands:
        return typo

    # A) Most similar by SequenceMatcher
    def sm_ratio(c):
        return SequenceMatcher(None, typo_l, c.lower()).ratio()

    best_sim = max(cands, key=lambda x: sm_ratio(x[0]))[0]

    # B) Best LM among candidates within edit distance <= k
    close = [(c, s) for (c, s) in cands if edit_distance_leq_k(typo_l, c.lower(), k=k)]
    if close:
        best_edlm = max(close, key=lambda x: x[1])[0]
    else:
        best_edlm = max(cands, key=lambda x: x[1])[0]  # fallback to most plausible

    # Compare which is closer to original typo
    # Primary: edit distance (we compute exact small distances with DP if you want,
    # but for simplicity we reuse the <=k checker by trying k=0..maxK.)
    def exact_lev(a, b):
        # full Levenshtein (only used twice, cheap)
        a, b = a.lower(), b.lower()
        m, n = len(a), len(b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev_diag = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                tmp = dp[j]
                cost = 0 if a[i-1] == b[j-1] else 1
                dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev_diag + cost)
                prev_diag = tmp
        return dp[n]

    ed_sim  = exact_lev(typo_l, best_sim.lower())
    ed_edlm = exact_lev(typo_l, best_edlm.lower())

    if ed_edlm < ed_sim:
        best = best_edlm
    elif ed_sim < ed_edlm:
        best = best_sim
    else:
        # tie -> higher SequenceMatcher
        best = best_edlm if sm_ratio(best_edlm) > sm_ratio(best_sim) else best_sim

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
                top_k=100
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
