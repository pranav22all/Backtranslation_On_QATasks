import math
import random
import nltk
from BackTranslation import BackTranslation
import time
import json

trans = BackTranslation()

def get_easy_backtranslate_codes():
    print(trans.searchLanguage('French'))
    print(trans.searchLanguage('German'))
    print(trans.searchLanguage('Spanish'))
    print(trans.searchLanguage('Dutch'))
    print(trans.searchLanguage('Italian'))
    print(trans.searchLanguage('Russian'))
    print(trans.searchLanguage('Swedish'))
    print(trans.searchLanguage('Norwegian'))

def get_hard_backtranslate_codes():
    print(trans.searchLanguage('Chinese'))
    print(trans.searchLanguage('Japanese'))
    print(trans.searchLanguage('Arabic'))
    print(trans.searchLanguage('Turkish'))
    print(trans.searchLanguage('Hungarian'))
    print(trans.searchLanguage('Korean'))
    print(trans.searchLanguage('Hebrew'))
    print(trans.searchLanguage('Tamil'))

#Modification #2: Utilize Harder Languages
#languages = ['zh-cn', 'ja', 'ar', 'tr', 'hu', 'ko', 'he', 'ta'] as possibilities (only keep 8)
def backtranslate_dataset(data_dict, languages = ['fr', 'de', 'es', 'nl', 'it', 'ru', 'sv', 'no'], prob=0.9, multiply_factor=10):
    """
    Takes in data_dict and list of languages, and performs backtranslation 
    on the questions and contexts. Returns new_data_dict with additional
    questions, contexts, and answers. 
    data_dict -> {'question': [], 'context': [], 'id': [], 'answer': []}
    languages -> list of strings containing languages for backtranslation 
    prob -> probability a given input example is backtranslated on
    """
    sleeping_time = 0.5
    
    def translate_excerpt(excerpt, languages, sleeping_time):
        try:
            #print("Except is:")
            #print(excerpt)
            translated = trans.translate(excerpt, src = 'en', tmp = random.choice(languages), sleeping=sleeping_time).result_text
            #Modification #3: Multi-step Backtranslation
            final_translated = trans.translate(translated, src = 'en', tmp = random.choice(languages), sleeping=sleeping_time).result_text
            print("Translated successfully", sleeping_time)
            #print("Translation is:")
            #print(translated)
            #print("*******")
            return final_translated, sleeping_time #Change translated to final_translated for Mod #3
        except Exception:
            sleeping_time = 1
            print("There was an exception while translating")
            time.sleep(10)
            return excerpt, sleeping_time

    new_data_dict = data_dict.copy() # Keep all original, non-backtranslated data
    num_questions = len(data_dict['question'])
    nltk.download('punkt')
    #start = time.time()

    for i in range(multiply_factor):
        for curr_index in range(num_questions):
            curr_question, curr_context, curr_answer = data_dict['question'][curr_index], data_dict['context'][curr_index], data_dict['answer'][curr_index]

            # Do backtranslation on this example:
            sentences = nltk.tokenize.sent_tokenize(curr_context) #will remove spaces at sentence start
            #print(sentences)

            #Find which sentence the current answer appears in:
            curr_answer_start_index = curr_answer["answer_start"][0]
            answer_sent_index = None

            curr_total_word_index = 0
            for sent_index, curr_sentence in enumerate(sentences):
                curr_total_word_index += len(curr_sentence) + 1 #Account for leading space
                if curr_total_word_index > curr_answer_start_index:
                    answer_sent_index = sent_index
                    break

            #At this point, we know the sentence with the answer, now backtranslate:
            before_answer = ' '.join(sentences[:answer_sent_index])
            after_answer = ' '.join(sentences[answer_sent_index + 1:])
            answer_sentence = sentences[answer_sent_index]

            if before_answer and random.random() < prob:
                before_answer, sleeping_time = translate_excerpt(before_answer, languages, sleeping_time)
            if random.random() < prob:
                answer_sentence, sleeping_time = translate_excerpt(answer_sentence, languages, sleeping_time)
            if after_answer and random.random() < prob:
                after_answer, sleeping_time = translate_excerpt(after_answer, languages, sleeping_time)
            
            word_count_before_answer_sentence = 0
            translated_context = ''
            if before_answer:
                translated_context += before_answer + ' '
                word_count_before_answer_sentence = len(translated_context)
            translated_context += answer_sentence
            if after_answer:
                translated_context += ' ' + after_answer
                
            #Pass only a single sentence (context) and original answer into compute_new_answer_span:
            new_answer_index, new_answer_text = compute_new_answer_span(answer_sentence, curr_answer["text"][0])
            if new_answer_index == -1: # No answer was found with a reasonable jaccard score
                print("Skipping this question")
                continue
            new_answer = {"answer_start": [word_count_before_answer_sentence + new_answer_index], "text": [new_answer_text]}

            #Modification #1: Translate Question
            #trans_question, sleeping_time = translate_excerpt(curr_question, languages, sleeping_time)

            #Update new_data_dict accordingly:
            new_data_dict['question'].append(curr_question) #Change this line for Mod #1
            new_data_dict['context'].append(translated_context)
            new_id = str(hash(translated_context + curr_question + str(random.random())))
            new_data_dict['id'].append(new_id)
            new_data_dict['answer'].append(new_answer)
        print("Finished round", i)
        print("Current size of new_data_dict is: ", len(new_data_dict["question"]))
        print("*******")
        sleeping_time = 0.5
        time.sleep(60) # Avoid upsetting Google translate
    #print(time.time() - start)

    print("Generated", len(new_data_dict["question"]), "total examples")
    print("Backtranslate function complete")
    # with open("translated_output.json", "w+") as f:
    #     json.dump(new_data_dict, f, indent=4)

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


if __name__ == '__main__':
    get_easy_backtranslate_codes()
