import math

def compute_new_answer_span(translated_context, orig_answer):
    """
    Takes in a sentence corresponding to a translated context and the original
    answer and returns a tuple of (index of answer, translated answer). If no
    answer has sufficient jaccard similarity, (-1, '') is returned.
    """
    def compute_bigrams(token):
        return {token[i:i+2].lower() for i in range(len(token) - 1)}

    def set_union(sets):
        result = set()
        for s in sets:
            result |= s
        return result

    def compute_jaccard(a, b):
        return len(a & b) / len(a | b)
        answer_index = translated_context.find(orig_answer)
        if answer_index != -1: # Got a complete match
            return answer_index, orig_answer
    
    tokenized_answer = orig_answer.split(' ')
    tokenized_context = translated_context.split(' ')
    
    answer_bigrams = set_union([compute_bigrams(token) for token in tokenized_answer])
    context_bigrams = [compute_bigrams(token) for token in tokenized_context]
    
    best_jaccard = -1
    best_answer_index = -1
    best_span_length = -1
    tolerance = math.ceil(len(tokenized_answer) / 3)
    for span_length in range(max(len(tokenized_answer) - tolerance, 1), len(tokenized_answer) + tolerance + 1):
        for i in range(len(tokenized_context) - span_length + 1):
            span = context_bigrams[i:i+span_length]
            span_bigrams = set_union(span)
            
            jaccard_score = compute_jaccard(span_bigrams, answer_bigrams)
            if jaccard_score > best_jaccard: # By default prefers shorter sequences that have equivalent scores
                best_jaccard = jaccard_score
                best_answer_index = i
                best_span_length = span_length
                
    translated_answer = ' '.join(tokenized_context[best_answer_index:best_answer_index+best_span_length])
#     print(best_span_length)
#     print(tokenized_context)
#     print(tokenized_answer)
#     print(best_jaccard)
    
    if best_jaccard < 0.45:
        return -1, ''
    return translated_context.find(translated_answer), translated_answer