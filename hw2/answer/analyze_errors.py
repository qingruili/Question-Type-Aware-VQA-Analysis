#!/usr/bin/env python3
"""
Analyze prediction errors by comparing model output with ground truth.
Uses the same F1 calculation as conlleval.py (chunk-based, not token-based).

Usage: 
    python analyze_errors.py                          # Analyze single model
    python analyze_errors.py -o errors.txt            # Save to file
    python analyze_errors.py --compare old.json new.json  # Compare two runs
"""

import sys
import json
from collections import defaultdict

# ============================================================
# F1 calculation functions (from conlleval.py)
# ============================================================

def split_tag(chunk_tag):
    """Split chunk tag into IOBES prefix and chunk_type"""
    if chunk_tag == 'O':
        return ('O', None)
    return chunk_tag.split('-', maxsplit=1)

def is_chunk_end(prev_tag, tag):
    """Check if the previous chunk ended between the previous and current word"""
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'
    if chunk_type1 != chunk_type2:
        return True
    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def is_chunk_start(prev_tag, tag):
    """Check if a new chunk started between the previous and current word"""
    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'
    if chunk_type1 != chunk_type2:
        return True
    return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

def calc_metrics(tp, p, t, percent=True):
    """Compute precision, recall and F1"""
    precision = tp / p if p else 0
    recall = tp / t if t else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    if percent:
        return 100 * precision, 100 * recall, 100 * f1
    return precision, recall, f1

def count_chunks(true_seqs, pred_seqs):
    """Count correct, true, and predicted chunks"""
    correct_chunks = defaultdict(int)
    true_chunks = defaultdict(int)
    pred_chunks = defaultdict(int)

    correct_counts = defaultdict(int)
    true_counts = defaultdict(int)
    pred_counts = defaultdict(int)

    prev_true_tag, prev_pred_tag = 'O', 'O'
    correct_chunk = None

    for true_tag, pred_tag in zip(true_seqs, pred_seqs):
        if true_tag == pred_tag:
            correct_counts[true_tag] += 1
        true_counts[true_tag] += 1
        pred_counts[pred_tag] += 1

        _, true_type = split_tag(true_tag)
        _, pred_type = split_tag(pred_tag)

        if correct_chunk is not None:
            true_end = is_chunk_end(prev_true_tag, true_tag)
            pred_end = is_chunk_end(prev_pred_tag, pred_tag)

            if pred_end and true_end:
                correct_chunks[correct_chunk] += 1
                correct_chunk = None
            elif pred_end != true_end or true_type != pred_type:
                correct_chunk = None

        true_start = is_chunk_start(prev_true_tag, true_tag)
        pred_start = is_chunk_start(prev_pred_tag, pred_tag)

        if true_start and pred_start and true_type == pred_type:
            correct_chunk = true_type
        if true_start:
            true_chunks[true_type] += 1
        if pred_start:
            pred_chunks[pred_type] += 1

        prev_true_tag, prev_pred_tag = true_tag, pred_tag

    if correct_chunk is not None:
        correct_chunks[correct_chunk] += 1

    return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)

def compute_f1_metrics(refs_flat, preds_flat):
    """Compute overall and per-type F1 metrics"""
    (correct_chunks, true_chunks, pred_chunks,
     correct_counts, true_counts, pred_counts) = count_chunks(refs_flat, preds_flat)
    
    # Overall metrics
    sum_correct_chunks = sum(correct_chunks.values())
    sum_true_chunks = sum(true_chunks.values())
    sum_pred_chunks = sum(pred_chunks.values())
    sum_correct_counts = sum(correct_counts.values())
    sum_true_counts = sum(true_counts.values())
    
    overall_prec, overall_rec, overall_f1 = calc_metrics(
        sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
    
    token_accuracy = 100 * sum_correct_counts / sum_true_counts if sum_true_counts else 0
    
    # Per-type metrics
    chunk_types = sorted(set(list(true_chunks) + list(pred_chunks)))
    per_type_metrics = {}
    for t in chunk_types:
        prec, rec, f1 = calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
        per_type_metrics[t] = {
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'true_count': true_chunks[t],
            'pred_count': pred_chunks[t],
            'correct_count': correct_chunks[t]
        }
    
    return {
        'overall': {
            'precision': overall_prec,
            'recall': overall_rec,
            'f1': overall_f1,
            'token_accuracy': token_accuracy,
            'total_tokens': sum_true_counts,
            'total_phrases': sum_true_chunks,
            'found_phrases': sum_pred_chunks,
            'correct_phrases': sum_correct_chunks
        },
        'per_type': per_type_metrics
    }

# ============================================================
# File loading functions
# ============================================================

def load_conll_words(filepath):
    """Load words from CoNLL format file (word is first column)"""
    sentences = []
    current_sentence = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                parts = line.split()
                current_sentence.append(parts[0])
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

def load_tags(filepath):
    """Load tags from output file (one tag per line, blank lines separate sentences)"""
    sentences = []
    current_sentence = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                current_sentence.append(line)
    if current_sentence:
        sentences.append(current_sentence)
    return sentences

# ============================================================
# Main analysis function
# ============================================================

def analyze_errors(input_file, pred_file, ref_file, output_file=None, save_json=None):
    """Compare predictions with reference and report errors using F1 metrics"""
    
    words = load_conll_words(input_file)
    preds = load_tags(pred_file)
    refs = load_tags(ref_file)
    
    # Flatten for F1 calculation
    refs_flat = [tag for sent in refs for tag in sent]
    preds_flat = [tag for sent in preds for tag in sent]
    
    # Compute F1 metrics (same as conlleval.py)
    metrics = compute_f1_metrics(refs_flat, preds_flat)
    
    # Collect per-sentence error info
    total_token_errors = 0
    error_types = {}
    all_sentences = []
    
    for sent_idx, (sent_words, sent_preds, sent_refs) in enumerate(zip(words, preds, refs)):
        if len(sent_words) != len(sent_preds) or len(sent_words) != len(sent_refs):
            print(f"Warning: Length mismatch in sentence {sent_idx}", file=sys.stderr)
            continue
        
        sentence_errors = []
        for word_idx, (word, pred, ref) in enumerate(zip(sent_words, sent_preds, sent_refs)):
            if pred != ref:
                total_token_errors += 1
                error_key = (ref, pred)
                error_types[error_key] = error_types.get(error_key, 0) + 1
                sentence_errors.append({
                    'word_idx': word_idx,
                    'word': word,
                    'gold': ref,
                    'pred': pred
                })
        
        all_sentences.append({
            'sent_idx': sent_idx,
            'words': sent_words,
            'refs': sent_refs,
            'preds': sent_preds,
            'errors': sentence_errors,
            'error_count': len(sentence_errors)
        })
    
    # Save JSON for comparison
    if save_json:
        json_data = {
            'metrics': metrics,
            'total_token_errors': total_token_errors,
            'error_types': {f"{k[0]}->{k[1]}": v for k, v in error_types.items()},
            'sentences': all_sentences
        }
        with open(save_json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON data saved to: {save_json}", file=sys.stderr)
    
    # Output report
    out = open(output_file, 'w') if output_file else sys.stderr
    
    print("=" * 70, file=out)
    print("ERROR ANALYSIS REPORT (using conlleval F1 metrics)", file=out)
    print("=" * 70, file=out)
    
    # Overall F1 metrics (same as conlleval.py)
    om = metrics['overall']
    error_sentences = [s for s in all_sentences if s['error_count'] > 0]
    
    print(f"\n{'─' * 70}", file=out)
    print("OVERALL METRICS (chunk-based, same as conlleval.py)", file=out)
    print(f"{'─' * 70}", file=out)
    print(f"  Processed {om['total_tokens']} tokens with {om['total_phrases']} phrases", file=out)
    print(f"  Found: {om['found_phrases']} phrases; Correct: {om['correct_phrases']}", file=out)
    print(f"", file=out)
    print(f"  Precision:      {om['precision']:6.2f}%", file=out)
    print(f"  Recall:         {om['recall']:6.2f}%", file=out)
    print(f"  F1 Score:       {om['f1']:6.2f}%  ← Main metric", file=out)
    print(f"  Token Accuracy: {om['token_accuracy']:6.2f}%", file=out)
    print(f"", file=out)
    print(f"  Token errors: {total_token_errors}", file=out)
    print(f"  Sentences with errors: {len(error_sentences)} / {len(all_sentences)}", file=out)
    
    # Per-type F1 metrics
    print(f"\n{'─' * 70}", file=out)
    print("PER-TYPE METRICS", file=out)
    print(f"{'─' * 70}", file=out)
    print(f"  {'Type':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Found':>8} {'Gold':>8}", file=out)
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}", file=out)
    
    for chunk_type in sorted(metrics['per_type'].keys()):
        m = metrics['per_type'][chunk_type]
        print(f"  {chunk_type:<10} {m['precision']:>9.2f}% {m['recall']:>9.2f}% {m['f1']:>9.2f}% {m['pred_count']:>8} {m['true_count']:>8}", file=out)
    
    # Top error types
    print(f"\n{'─' * 70}", file=out)
    print("TOP 15 TOKEN ERROR TYPES (Gold -> Predicted : Count)", file=out)
    print(f"{'─' * 70}", file=out)
    sorted_errors = sorted(error_types.items(), key=lambda x: -x[1])[:15]
    for (gold, pred), count in sorted_errors:
        print(f"  {gold:8} -> {pred:8} : {count:4} times", file=out)
    
    # All error sentences in original order
    print(f"\n{'=' * 70}", file=out)
    print(f"ALL ERROR SENTENCES ({len(error_sentences)} sentences with errors)", file=out)
    print("=" * 70, file=out)
    
    for sent_data in all_sentences:
        if sent_data['error_count'] == 0:
            continue
            
        sent_idx = sent_data['sent_idx']
        sent_words = sent_data['words']
        sent_refs = sent_data['refs']
        sent_preds = sent_data['preds']
        errors = sent_data['errors']
        
        print(f"\n{'─' * 70}", file=out)
        print(f"[Sentence {sent_idx:04d}] ({len(errors)} error{'s' if len(errors) > 1 else ''})", file=out)
        print(f"{'─' * 70}", file=out)
        
        # Print full sentence
        print(f"Text: {' '.join(sent_words)}", file=out)
        
        # Print word-by-word comparison
        print(f"\nWord-by-word (errors marked with ✗):", file=out)
        print(f"{'Idx':<4} {'Word':<20} {'Gold':<10} {'Pred':<10} {'Match'}", file=out)
        print(f"{'-'*4} {'-'*20} {'-'*10} {'-'*10} {'-'*5}", file=out)
        
        for word_idx, (word, ref, pred) in enumerate(zip(sent_words, sent_refs, sent_preds)):
            match = "✓" if ref == pred else "✗"
            word_display = word[:18] + ".." if len(word) > 20 else word
            if ref != pred:
                print(f"{word_idx:<4} {word_display:<20} {ref:<10} {pred:<10} {match}  <<<", file=out)
            else:
                print(f"{word_idx:<4} {word_display:<20} {ref:<10} {pred:<10} {match}", file=out)
    
    if output_file:
        out.close()
        print(f"\nError analysis written to: {output_file}", file=sys.stderr)
    
    return all_sentences, metrics

def compare_runs(file1, file2, output_file=None):
    """Compare two JSON error analysis files"""
    
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    
    out = open(output_file, 'w') if output_file else sys.stderr
    
    print("=" * 70, file=out)
    print("COMPARISON REPORT", file=out)
    print(f"  Run 1 (OLD): {file1}", file=out)
    print(f"  Run 2 (NEW): {file2}", file=out)
    print("=" * 70, file=out)
    
    # Overall F1 comparison
    m1, m2 = data1['metrics']['overall'], data2['metrics']['overall']
    
    print(f"\n{'─' * 70}", file=out)
    print("OVERALL METRICS COMPARISON", file=out)
    print(f"{'─' * 70}", file=out)
    print(f"  {'Metric':<20} {'OLD':>12} {'NEW':>12} {'Change':>12}", file=out)
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}", file=out)
    
    for metric in ['f1', 'precision', 'recall', 'token_accuracy']:
        v1, v2 = m1[metric], m2[metric]
        label = metric.replace('_', ' ').title()
        print(f"  {label:<20} {v1:>11.2f}% {v2:>11.2f}% {v2-v1:>+11.2f}%", file=out)
    
    print(f"  {'Token Errors':<20} {data1['total_token_errors']:>12} {data2['total_token_errors']:>12} {data2['total_token_errors']-data1['total_token_errors']:>+12}", file=out)
    
    # Per-type F1 comparison
    print(f"\n{'─' * 70}", file=out)
    print("PER-TYPE F1 COMPARISON", file=out)
    print(f"{'─' * 70}", file=out)
    print(f"  {'Type':<10} {'OLD F1':>10} {'NEW F1':>10} {'Change':>10}", file=out)
    print(f"  {'-'*10} {'-'*10} {'-'*10} {'-'*10}", file=out)
    
    all_types = set(data1['metrics']['per_type'].keys()) | set(data2['metrics']['per_type'].keys())
    for chunk_type in sorted(all_types):
        f1_old = data1['metrics']['per_type'].get(chunk_type, {}).get('f1', 0)
        f1_new = data2['metrics']['per_type'].get(chunk_type, {}).get('f1', 0)
        change = f1_new - f1_old
        indicator = "↑" if change > 0.5 else ("↓" if change < -0.5 else " ")
        print(f"  {chunk_type:<10} {f1_old:>9.2f}% {f1_new:>9.2f}% {change:>+9.2f}% {indicator}", file=out)
    
    # Per-sentence comparison
    sentences1 = {s['sent_idx']: s for s in data1['sentences']}
    sentences2 = {s['sent_idx']: s for s in data2['sentences']}
    
    improved = []
    regressed = []
    fixed = []
    broken = []
    
    for sent_idx in sentences1:
        if sent_idx not in sentences2:
            continue
        
        err1 = sentences1[sent_idx]['error_count']
        err2 = sentences2[sent_idx]['error_count']
        
        if err2 < err1:
            improved.append((sent_idx, err1, err2, sentences1[sent_idx], sentences2[sent_idx]))
            if err2 == 0:
                fixed.append((sent_idx, err1, sentences1[sent_idx]))
        elif err2 > err1:
            regressed.append((sent_idx, err1, err2, sentences1[sent_idx], sentences2[sent_idx]))
            if err1 == 0:
                broken.append((sent_idx, err2, sentences2[sent_idx]))
    
    print(f"\n{'─' * 70}", file=out)
    print("PER-SENTENCE CHANGES", file=out)
    print(f"{'─' * 70}", file=out)
    print(f"  Improved sentences:  {len(improved):4} (fewer errors in NEW)", file=out)
    print(f"  Regressed sentences: {len(regressed):4} (more errors in NEW)", file=out)
    print(f"  Fully fixed:         {len(fixed):4} (had errors -> now perfect)", file=out)
    print(f"  Newly broken:        {len(broken):4} (was perfect -> now has errors)", file=out)
    
    # Show improved sentences
    if improved:
        print(f"\n{'=' * 70}", file=out)
        print(f"IMPROVED SENTENCES ({len(improved)} total, showing first 30)", file=out)
        print("=" * 70, file=out)
        
        for sent_idx, err1, err2, s1, s2 in improved[:30]:
            print(f"\n[Sentence {sent_idx:04d}] Errors: {err1} -> {err2} (improved by {err1-err2})", file=out)
            print(f"  Text: {' '.join(s1['words'][:15])}{'...' if len(s1['words']) > 15 else ''}", file=out)
            
            old_error_words = {e['word_idx']: e for e in s1['errors']}
            new_error_words = {e['word_idx']: e for e in s2['errors']}
            
            fixed_positions = set(old_error_words.keys()) - set(new_error_words.keys())
            for pos in sorted(fixed_positions):
                e = old_error_words[pos]
                print(f"    ✓ FIXED: '{e['word']}' was {e['gold']}->{e['pred']}, now correct", file=out)
    
    # Show regressed sentences
    if regressed:
        print(f"\n{'=' * 70}", file=out)
        print(f"REGRESSED SENTENCES ({len(regressed)} total, showing first 30)", file=out)
        print("=" * 70, file=out)
        
        for sent_idx, err1, err2, s1, s2 in regressed[:30]:
            print(f"\n[Sentence {sent_idx:04d}] Errors: {err1} -> {err2} (worse by {err2-err1})", file=out)
            print(f"  Text: {' '.join(s1['words'][:15])}{'...' if len(s1['words']) > 15 else ''}", file=out)
            
            old_error_words = {e['word_idx']: e for e in s1['errors']}
            new_error_words = {e['word_idx']: e for e in s2['errors']}
            
            new_positions = set(new_error_words.keys()) - set(old_error_words.keys())
            for pos in sorted(new_positions):
                e = new_error_words[pos]
                print(f"    ✗ NEW ERROR: '{e['word']}' now {e['gold']}->{e['pred']}", file=out)
    
    if output_file:
        out.close()
        print(f"\nComparison written to: {output_file}", file=sys.stderr)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze prediction errors with F1 metrics')
    parser.add_argument('-i', '--input', default='data/input/dev.txt',
                        help='Input file with words (CoNLL format)')
    parser.add_argument('-p', '--pred', default='output/dev.out',
                        help='Prediction file')
    parser.add_argument('-r', '--ref', default='data/reference/dev.out',
                        help='Reference/ground truth file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output file (default: print to stderr)')
    parser.add_argument('-j', '--json', default=None,
                        help='Save JSON data for later comparison')
    parser.add_argument('--compare', nargs=2, metavar=('OLD_JSON', 'NEW_JSON'),
                        help='Compare two JSON files from previous runs')
    args = parser.parse_args()
    
    if args.compare:
        compare_runs(args.compare[0], args.compare[1], args.output)
    else:
        analyze_errors(args.input, args.pred, args.ref, args.output, args.json)