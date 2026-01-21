from transformers import pipeline
import logging, os, csv
import textdistance

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

def select_correction(typo, predict):
    # return the most likely prediction for the mask token
    return predict[0]['token_str']

def analyze_predictions(typo, predict, analysis_file, sent, typo_index, ground_truth_word):
    """
    Analyze predictions and write to file (only used in analysis mode).
    """
    # Create sentence with typo word in brackets
    sent_with_brackets = []
    for j, word in enumerate(sent):
        if j == typo_index:
            sent_with_brackets.append(f"[{word}]")
        else:
            sent_with_brackets.append(word)
    
    analysis_file.write(f"\nOriginal sentence: {' '.join(sent_with_brackets)}\n")
    analysis_file.write(f"Original typo: '{typo}'\n")
    analysis_file.write(f"Ground truth: '{ground_truth_word}'\n")
    analysis_file.write(f"{'Rank':<6}{'Word':<15}{'Score':<12}{'Edit Dist':<12}\n")
    analysis_file.write("-" * 50 + "\n")
    
    for i, pred in enumerate(predict):
        word = pred['token_str']
        score = pred['score']
        edit_dist = textdistance.damerau_levenshtein(typo.lower(), word.lower())
        # Mark if this is the ground truth
        marker = " <-- GROUND TRUTH" if word.lower() == ground_truth_word.lower() else ""
        analysis_file.write(
            f"{i+1:<6}{word:<15}{score:<12.6f}{edit_dist:<12}{marker}\n"
        )

def spellchk(fh, analysis_mode=None):
    # If analysis mode is enabled, open analysis file and read ground truth
    if analysis_mode:
        analysis_file = open('analysis.txt', 'w')
        reference_file = os.path.join('data', 'reference', 'dev.out')
        with open(reference_file) as ref_f:
            ground_truth_sentences = [line.strip().split() for line in ref_f]
        sentence_index = 0
    
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent
        
        for i in locations:
            # predict top_k replacements only for the typo word at index i
            predict = fill_mask(
                " ".join([ sent[j] if j != i else mask for j in range(len(sent)) ]), 
                top_k=20
            )
            logging.info(predict)
            
            # If analysis mode, analyze predictions
            if analysis_mode:
                ground_truth_word = ground_truth_sentences[sentence_index][i]
                analyze_predictions(sent[i], predict, analysis_file, sent, i, ground_truth_word)
            
            spellchk_sent[i] = select_correction(sent[i], predict)
        
        if analysis_mode:
            sentence_index += 1
        
        yield(locations, spellchk_sent)
    
    # Close analysis file if it was opened
    if analysis_mode:
        analysis_file.close()

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

    analysis_mode = False

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f, analysis_mode):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))
