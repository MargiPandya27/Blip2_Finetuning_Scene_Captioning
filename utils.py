from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_bleu_scores(references, hypotheses):


    weights_list = [
        (1.0, 0, 0, 0),     # BLEU-1
        (0.5, 0.5, 0, 0),   # BLEU-2
        (0.33, 0.33, 0.33, 0),  # BLEU-3
        (0.25, 0.25, 0.25, 0.25)  # BLEU-4
    ]


    for i, weights in enumerate(weights_list, 1):
        score = sum(
            sentence_bleu(ref, hyp, weights=weights, smoothing_function=SmoothingFunction().method4)
            for ref, hyp in zip(references, hypotheses)
        ) / len(references)
        print(f"BLEU-{i}: {score:.4f}")

    return score
