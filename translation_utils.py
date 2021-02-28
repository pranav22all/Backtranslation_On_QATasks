import math
import random
import nltk
from BackTranslation import BackTranslation


def backtranslate_dataset(data_dict, languages, prob)
    """
    Takes in data_dict and list of languages, and performs backtranslation 
    on the questions and contexts. Returns new_data_dict with additional
    questions, contexts, and answers. 
    data_dict -> {'question': [], 'context': [], 'id': [], 'answer': []}
    languages -> list of strings containing languages for backtranslation 
    prob -> probability a given input example is backtranslated on
    """

    #TODO: Determine handling of ids
    new_data_dict = {'question': [], 'context': [], 'id': [], 'answer': []}
    num_questions = len(data_dict['question'])
    trans = BackTranslation()
    nltk.download('punkt') #Make sure this line works as expected (sentence splitting)

    for curr_index in range(num_questions):
        curr_question, curr_context = new_data_dict['question'][curr_index], new_data_dict['context'][curr_index]
        curr_id, curr_answer = new_data_dict['id'][curr_index], new_data_dict['answer'][curr_index]

        # Do backtranslation on this example:
        if random.random() <= prob:
            sentences = nltk.tokenize.sent_tokenize(curr_context) #will remove spaces at sentence start

            #Find which sentence the current answer appears in:
            curr_answer_start_index = curr_answer["answer_start"]
            answer_sent_index = None
            word_ind_in_sent = None

            curr_total_word_index = 0
            for sent_index, curr_sentence in enumerate(sentences):
                curr_total_word_index += len(curr_sentence) + 1 #Account for leading space
                if curr_total_word_index >= curr_answer_start_index:
                    answer_sent_index = sent_index
                    break

            #Can handle multiple answers in answer sentence utilizing something like this:
            #word_ind_in_sent = sentences[answer_sent_index].find(curr_answer["text"], curr_total_word_index - curr_answer_start_index - 1)

            #At this point, we know the sentence with the answer, now backtranslate:
            for curr_lang in languages:
                translated_answer_sentence = None
                translated_context = ""
                translated_question = trans.translate(curr_question, src = 'en', tmp = curr_lang)
                word_count_before_answer_sentence = 0
                for sent_index, curr_sentence in enumerate(sentences):
                    translated_sentence = trans.translate(curr_sentence, src = 'en', tmp = curr_lang)
                    if sent_index == answer_sent_index:
                        translated_answer_sentence = translated_sentence
                        word_count_before_answer_sentence = len(translated_context) + 1 #Account for space
                    translated_context += " " + translated_sentence

                #Pass only a single sentence (context) and original answer into compute_new_answer_span:
                new_answer_index, new_answer_text = compute_new_answer_span(translated_answer_sentence, curr_answer["text"])
                new_answer = {"answer_start": word_count_before_answer_sentence + new_answer_index, "text": new_answer_text}

                #Update new_data_dict accordingly:
                new_data_dict['question'].append(translated_question)
                new_data_dict['context'].append(translated_context)
                #new_data_dict['id'].append(curr_id)  #Determine how to handle id properly
                new_data_dict['answer'].append(new_answer)

        #Don't backtranslate here:
        else:
            new_data_dict['question'].append(curr_question)
            new_data_dict['context'].append(curr_context)
            new_data_dict['id'].append(curr_id)
            new_data_dict['answer'].append(curr_answer)

    return new_data_dict



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