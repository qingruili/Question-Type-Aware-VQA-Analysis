from transformers import pipeline
import logging, os, csv
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
def calculate_edit_distance(a, b):
    """
    Calculate Damerau-Levenshtein edit distance between two words.
    Considers four operations: insertion, deletion, substitution, and transposition.
    Transposition (swap of adjacent characters) counts as 1 operation, not 2.
    
    Args:
        a: First word
        b: Second word
        
    Returns:
        int: Minimum number of edits needed
    """
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    
    len_a, len_b = len(a), len(b)
    
    # Create distance matrix with extra row/column for empty string
    # dp[i][j] represents distance between a[0:i] and b[0:j]
    max_dist = len_a + len_b
    dp = [[max_dist for _ in range(len_b + 2)] for _ in range(len_a + 2)]
    
    # Initialize base cases
    dp[0][0] = max_dist
    for i in range(0, len_a + 1):
        dp[i + 1][0] = max_dist
        dp[i + 1][1] = i
    for j in range(0, len_b + 1):
        dp[0][j + 1] = max_dist
        dp[1][j + 1] = j
    
    # Fill in the distance matrix
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            
            dp[i + 1][j + 1] = min(
                dp[i][j] + cost,        # substitution (or match if cost=0)
                dp[i + 1][j] + 1,       # insertion
                dp[i][j + 1] + 1,       # deletion
            )
            
            # Check for transposition (swap of adjacent characters)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                dp[i + 1][j + 1] = min(dp[i + 1][j + 1], dp[i - 1][j - 1] + 1)
    
    return dp[len_a + 1][len_b + 1]


def calculate_similarity_score(a, b):
    """
    Calculate SequenceMatcher similarity ratio between two words.      
    Returns:
        float: Similarity ratio between 0 and 1 (higher = more similar)
    """
    a, b = a.lower(), b.lower()
    return SequenceMatcher(None, a, b).ratio()


def calculate_edit_distance_score(ed, max_distance=7):
    """
    Convert edit distance to a normalized score (higher is better).
    Uses linear normalization: score = 1 - (distance / max_distance)
    
    Args:
        ed: Edit distance value
        max_distance: Maximum expected edit distance (default: 7, from data analysis)
        
    Returns:
        float: Normalized score between 0 and 1 (higher = closer match)
    """
    if ed >= max_distance:
        return 0.0
    return 1.0 - (ed / max_distance)


def select_correction(typo, predict, ed_weight=0.45, sim_weight=0.45, transformer_weight=0.1):
    """
    Select the best spelling correction using weighted combination of multiple signals.
    
    Strategy:
    1. Filter candidates (must be different from typo, edit_distance > 0)
    2. For each candidate, calculate weighted score:
       - Edit distance score (1 - distance/7): closer = higher score
       - Similarity score (SequenceMatcher): higher = more similar
       - Transformer score: higher = more contextually appropriate
    3. Select candidate with highest weighted combined score
    4. Preserve original capitalization
    
    Args:
        typo: The misspelled word
        predict: List of predictions from transformer, each with 'token_str' and 'score'
        ed_weight: Weight for edit distance score
        sim_weight: Weight for similarity score
        transformer_weight: Weight for transformer score
    
    Returns:
        str: Best correction with capitalization preserved
    """
    typo_lower = typo.lower()

    # Extract candidates with their scores
    candidates = []
    for p in predict:
        cand = p["token_str"].strip()
        if cand:
            transformer_score = float(p.get("score", 0.0))
            edit_dist = calculate_edit_distance(typo_lower, cand.lower())
            
            # Only consider candidates that are different from the typo (edit_distance > 0)
            if edit_dist > 0:
                candidates.append((cand, transformer_score, edit_dist))
    
    # If no valid candidates found, return original typo
    if not candidates:
        return typo
    
    # Find best candidate by computing weighted score for each
    best_score = -1
    best_candidate = None
    
    for cand, transformer_score, edit_dist in candidates:
        # Calculate edit distance score (normalized 0-1, higher is better)
        edit_distance_score = calculate_edit_distance_score(edit_dist)
        
        # Calculate similarity score using SequenceMatcher (0-1, higher is better)
        similarity_score = calculate_similarity_score(typo_lower, cand.lower())
        
        # Compute weighted combination of all three scores
        # All scores are already in [0, 1] range
        combined_score = (ed_weight * edit_distance_score) + \
                        (sim_weight * similarity_score) + \
                        (transformer_weight * transformer_score)
        
        # Track the best candidate
        if combined_score > best_score:
            best_score = combined_score
            best_candidate = cand
    
    # Preserve capitalization from original typo
    if typo and typo[0].isupper():
        best_candidate = best_candidate.capitalize()
    
    return best_candidate


def analyze_predictions(typo, predict, analysis_file, sent, typo_index, 
                       ground_truth_word, selected_word):
    """
    Write detailed prediction analysis to file for incorrect corrections.
    Shows all top predictions with their scores and metrics.
    Only writes entries where the selected correction is wrong.
    
    Args:
        typo: Original misspelled word
        predict: List of transformer predictions
        analysis_file: File handle to write analysis
        sent: Complete sentence tokens
        typo_index: Index of typo in sentence
        ground_truth_word: Correct word from reference
        selected_word: Word selected by our algorithm
    """
    # Only analyze incorrect predictions
    if selected_word == ground_truth_word:
        return
    
    # Format sentence with typo in brackets for context
    sent_with_brackets = []
    for j, word in enumerate(sent):
        if j == typo_index:
            sent_with_brackets.append(f"[{word}]")
        else:
            sent_with_brackets.append(word)
    
    # Write analysis header
    analysis_file.write(f"\nOriginal sentence: {' '.join(sent_with_brackets)}\n")
    analysis_file.write(f"Original typo: '{typo}'\n")
    analysis_file.write(f"Ground truth: '{ground_truth_word}'\n")
    analysis_file.write(f"Selected word: '{selected_word}' *** INCORRECT ***\n")
    analysis_file.write(f"{'Rank':<6}{'Word':<15}{'Score':<12}{'Edit Dist':<12}{'Similarity':<12}\n")
    analysis_file.write("-" * 60 + "\n")
    
    # Write all predictions with metrics
    for i, pred in enumerate(predict):
        word = pred['token_str']
        score = pred['score']
        edit_dist = calculate_edit_distance(typo, word)
        sim = calculate_similarity_score(typo, word)
        
        # Mark ground truth and selected word
        marker = ""
        if word.lower() == ground_truth_word.lower():
            marker = " <-- GROUND TRUTH"
        elif word.lower() == selected_word.lower():
            marker = " <-- SELECTED (WRONG)"
        
        analysis_file.write(
            f"{i+1:<6}{word:<15}{score:<12.6f}{edit_dist:<12}{sim:<12.6f}{marker}\n"
        )


def write_detailed_scores(typo, predict, scores_writer, ground_truth_word, 
                         sentence_num, typo_index, selected_word):
    """
    Write side-by-side comparison of ground truth vs selected word metrics to CSV.
    Allows analysis of why algorithm chose selected word over ground truth.
    
    Args:
        typo: Original misspelled word
        predict: List of transformer predictions
        scores_writer: CSV writer object
        ground_truth_word: Correct word from reference
        sentence_num: Sentence index (unused but kept for compatibility)
        typo_index: Word index in sentence (unused but kept for compatibility)
        selected_word: Word selected by our algorithm
    """
    # Find ground truth in predictions
    gt_transformer_score = "N/A"
    gt_rank = "N/A"
    for i, pred in enumerate(predict):
        if pred['token_str'].lower() == ground_truth_word.lower():
            gt_transformer_score = f"{pred['score']:.6f}"
            gt_rank = str(i + 1)
            break
    
    # Calculate ground truth metrics
    gt_edit_distance = calculate_edit_distance(typo, ground_truth_word)
    gt_edit_score = calculate_edit_distance_score(gt_edit_distance) if gt_edit_distance > 0 else "N/A"
    gt_similarity = calculate_similarity_score(typo, ground_truth_word)
    
    # Find selected word in predictions
    selected_transformer_score = "N/A"
    selected_rank = "N/A"
    for i, pred in enumerate(predict):
        if pred['token_str'].lower() == selected_word.lower():
            selected_transformer_score = f"{pred['score']:.6f}"
            selected_rank = str(i + 1)
            break
    
    # Calculate selected word metrics
    selected_edit_distance = calculate_edit_distance(typo, selected_word)
    selected_edit_score = calculate_edit_distance_score(selected_edit_distance) if selected_edit_distance > 0 else "N/A"
    selected_similarity = calculate_similarity_score(typo, selected_word)
    
    # Write row with all metrics for comparison
    scores_writer.writerow([
        typo,
        ground_truth_word,
        gt_transformer_score,
        gt_rank,
        gt_edit_distance,
        f"{gt_edit_score:.6f}" if isinstance(gt_edit_score, float) else gt_edit_score,
        f"{gt_similarity:.6f}",
        selected_word,
        selected_transformer_score,
        selected_rank,
        selected_edit_distance,
        f"{selected_edit_score:.6f}" if isinstance(selected_edit_score, float) else selected_edit_score,
        f"{selected_similarity:.6f}"
    ])


def spellchk(fh, analysis_mode=None):
    """
    Main spell checking function.
    Processes sentences with typos and generates corrections.
    
    Args:
        fh: File handle for input TSV file
        analysis_mode: If True, generates detailed analysis files
        
    Yields:
        tuple: (typo locations, corrected sentence tokens)
    """
    # Initialize analysis files if in analysis mode
    if analysis_mode:
        analysis_file = open('output/dev_error_details.txt', 'w')
        scores_file = open('output/dev_score_details.csv', 'w', newline='')
        scores_writer = csv.writer(scores_file)
        
        # Write CSV header
        scores_writer.writerow([
            'typo',
            'ground_truth',
            'ground_truth_transformer_score',
            'ground_truth_transformer_rank',
            'ground_truth_edit_distance',
            'ground_truth_edit_score',
            'ground_truth_similarity_score',
            'selected',
            'selected_transformer_score',
            'selected_transformer_rank',
            'selected_edit_distance',
            'selected_edit_score',
            'selected_similarity_score'
        ])
        
        # Load ground truth reference
        reference_file = os.path.join('data', 'reference', 'dev.out')
        with open(reference_file) as ref_f:
            ground_truth_sentences = [line.strip().split() for line in ref_f]
        sentence_index = 0
    
    # Process each sentence
    for (locations, sent) in get_typo_locations(fh):
        spellchk_sent = sent.copy()
        
        # Correct each typo in the sentence
        for i in locations:
            # Get top 200 predictions from transformer
            predict = fill_mask(
                " ".join([sent[j] if j != i else mask for j in range(len(sent))]), 
                top_k=200
            )
            logging.info(predict)
            
            # Select best correction
            selected_word = select_correction(sent[i], predict)
            
            # Write analysis if in analysis mode
            if analysis_mode:
                ground_truth_word = ground_truth_sentences[sentence_index][i]
                analyze_predictions(sent[i], predict, analysis_file, sent, i, 
                                  ground_truth_word, selected_word)
                write_detailed_scores(sent[i], predict, scores_writer, ground_truth_word, 
                                    sentence_index, i, selected_word)
            
            # Apply correction
            spellchk_sent[i] = selected_word
        
        if analysis_mode:
            sentence_index += 1
        
        yield (locations, spellchk_sent)
    
    # Clean up analysis files
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
    analysis_mode = False  # Set to True to enable analysis mode

    with open(opts.input) as f:
        for (locations, spellchk_sent) in spellchk(f, analysis_mode):
            print("{locs}\t{sent}".format(
                locs=",".join([str(i) for i in locations]),
                sent=" ".join(spellchk_sent)
            ))