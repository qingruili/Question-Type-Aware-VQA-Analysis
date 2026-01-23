from transformers import pipeline
import logging, os, csv
import textdistance  # ADDED: For calculating edit distance
from difflib import SequenceMatcher  # ADDED: For calculating similarity ratio

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


# ADDED: Function to calculate edit distance between two words
def edit_distance(a, b):
    """
    Calculate the Damerau-Levenshtein edit distance
    TODO: Implement edit distance manually if textdistance is not allowed
    """
    a, b = a.lower(), b.lower()
    return textdistance.damerau_levenshtein(a, b)


# ADDED: Function to calculate similarity ratio between two words
def similarity_ratio(a, b):
    """Calculate SequenceMatcher similarity ratio (0 to 1, higher = more similar)."""
    return SequenceMatcher(None, a, b).ratio()


# MODIFIED: Enhanced from default.py to use configurable correction methods
def select_correction(typo, predict):
    """
    Select the best correction from predictions using one of two methods:
    Method 1: Edit distance (lower is better)
    Method 2: Similarity ratio (higher is better)
    
    Uncomment one of the methods below to use it.
    """
    
    # ===== METHOD 1: Edit Distance (Minimum edit distance) =====
    min_edit_dist = float('inf')
    best_word = predict[0]['token_str']  # fallback to top prediction
    
    for pred in predict:
        word = pred['token_str']
        edit_dist = edit_distance(typo, word)
        
        if edit_dist < min_edit_dist:
            min_edit_dist = edit_dist
            best_word = word
    
    # ===== METHOD 2: Similarity Ratio (Maximum similarity) =====
    # Uncomment this section to use similarity ratio instead
    # max_similarity = -1
    # best_word = predict[0]['token_str']  # fallback to top prediction
    # 
    # for pred in predict:
    #     word = pred['token_str']
    #     sim = similarity_ratio(typo, word)
    #     
    #     if sim > max_similarity:
    #         max_similarity = sim
    #         best_word = word
    
    # Preserve capitalization if original typo was capitalized
    if typo and typo[0].isupper():
        best_word = best_word.capitalize()
    
    return best_word


# ADDED: Function to analyze predictions and write detailed analysis
def analyze_predictions(typo, predict, analysis_file, sent, typo_index, ground_truth_word, selected_word):
    """
    Analyze predictions and write to file (only used in analysis mode).
    Shows all predictions with their scores and edit distances.
    Only writes to file if the selected word is incorrect.
    """
    # Only write to file if the correction is wrong
    if selected_word.lower() == ground_truth_word.lower():
        return  # Skip writing for correct answers
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
    analysis_file.write(f"Selected word: '{selected_word}' *** INCORRECT ***\n")
    analysis_file.write(f"{'Rank':<6}{'Word':<15}{'Score':<12}{'Edit Dist':<12}{'Similarity':<12}\n")
    analysis_file.write("-" * 60 + "\n")
    
    for i, pred in enumerate(predict):
        word = pred['token_str']
        score = pred['score']
        edit_dist = edit_distance(typo, word)
        sim = similarity_ratio(typo, word)
        # Mark if this is the ground truth
        marker = " <-- GROUND TRUTH" if word.lower() == ground_truth_word.lower() else ""
        analysis_file.write(
            f"{i+1:<6}{word:<15}{score:<12.6f}{edit_dist:<12}{sim:<12.6f}{marker}\n"
        )


# ADDED: Function to write ground truth scores and selected word to CSV
def write_ground_truth_scores(typo, predict, scores_writer, ground_truth_word, 
                               sentence_num, typo_index, selected_word):
    """
    Write ground truth word's scores and selected word to scores.csv file.
    Includes: typo, ground_truth, transformer_score, rank, edit_distance, 
              similarity_score, dissimilarity_score, selected_word, is_correct
    """
    # Find ground truth in predictions
    transformer_score = "N/A"
    rank = "N/A"
    
    for i, pred in enumerate(predict):
        if pred['token_str'].lower() == ground_truth_word.lower():
            transformer_score = f"{pred['score']:.6f}"
            rank = str(i + 1)
            break
    
    # Calculate edit distance between typo and ground truth
    edit_dist = edit_distance(typo, ground_truth_word)
    
    # Calculate similarity score between typo and ground truth
    sim_score = similarity_ratio(typo, ground_truth_word)
    
    # Calculate dissimilarity (1 - similarity) - sometimes easier to interpret
    dissim_score = 1 - sim_score
    
    # Check if selected word matches ground truth
    is_correct = "YES" if selected_word.lower() == ground_truth_word.lower() else "NO"
    
    # Write to scores file using csv writer (handles commas properly)
    scores_writer.writerow([
        typo, 
        ground_truth_word, 
        transformer_score, 
        rank, 
        edit_dist,
        f"{sim_score:.6f}",
        f"{dissim_score:.6f}",
        selected_word,
        is_correct
    ])


# MODIFIED: Enhanced from default.py to support analysis mode
def spellchk(fh, analysis_mode=None):
    # ADDED: If analysis mode is enabled, open analysis files and read ground truth
    if analysis_mode:
        analysis_file = open('output/dev_error_details.txt', 'w')
        scores_file = open('output/dev_score_details.csv', 'w', newline='')
        scores_writer = csv.writer(scores_file)
        # MODIFIED: Added more columns to CSV header
        scores_writer.writerow([
            'Typo', 'Ground_Truth', 'Transformer_Score', 'Rank', 'Edit_Distance',
            'Similarity_Score', 'Dissimilarity_Score', 'Selected_Word', 'Is_Correct'
        ])
        
        reference_file = os.path.join('data', 'reference', 'dev.out')
        with open(reference_file) as ref_f:
            ground_truth_sentences = [line.strip().split() for line in ref_f]
        sentence_index = 0
    
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent
        
        for i in locations:
            # MODIFIED: Increased top_k from 20 to 100 to capture more candidates
            predict = fill_mask(
                " ".join([ sent[j] if j != i else mask for j in range(len(sent)) ]), 
                top_k=100
            )
            logging.info(predict)
            
            # MODIFIED: Select correction before analysis so we can record it
            selected_word = select_correction(sent[i], predict)
            
            # ADDED: If analysis mode, analyze predictions
            if analysis_mode:
                ground_truth_word = ground_truth_sentences[sentence_index][i]
                analyze_predictions(sent[i], predict, analysis_file, sent, i, ground_truth_word, selected_word)
                # MODIFIED: Now includes selected_word parameter
                write_ground_truth_scores(sent[i], predict, scores_writer, ground_truth_word, 
                                        sentence_index, i, selected_word)
            
            spellchk_sent[i] = selected_word
        
        if analysis_mode:
            sentence_index += 1
        
        yield(locations, spellchk_sent)
    
    # ADDED: Close analysis files if they were opened
    if analysis_mode:
        analysis_file.close()
        scores_file.close()


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

    # ADDED: Analysis mode flag - set to True to enable detailed analysis
    analysis_mode = True  # Set to True to enable analysis mode

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f, analysis_mode):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))