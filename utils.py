from nltk.corpus import wordnet as wn
import torch
from transformers import AutoTokenizer, AutoModel
import re
import numpy as np
from scipy.stats import ttest_ind
import nltk


def get_word_idx(tokenizer, encoded_context, word) -> list[int]:
        
        """Function to return the indices of the tokens from a given word, given a word and an encoded context containing the word.#
        :param tokenizer: the tokenizer used to encode the context
        :param encoded_context: the encoded context for the word (phrase or sentence)
        :param word: the word to look for
        :return: a list of the indices of the word's tokens within the encoded context
        """#
        # include all tokenization styles, e.g. with whitespace at word beginning (gpt-like models)
        word_pattern = re.compile("^\W?" + word + "\W?$", flags=re.IGNORECASE)
        
        # unique word IDs in the context
        unique_word_ids = list(set([word_id for word_id in encoded_context.word_ids() if word_id is not None]))#

        # retrieve word idx by looping over candidate word ids
        try_sequence_length = 5 # iteratively try longer token sequences to find the word, in case the word was split into subwords
        for i in range(1, try_sequence_length + 1):
            for word_id in unique_word_ids:
                word_id_sequence = [word_id + j for j in range(i)]
                # get the indices of the candidate sequence
                idx = np.where(np.isin(encoded_context.word_ids(), word_id_sequence))[0].tolist()
                # decode words at the indices
                decoded = tokenizer.decode(encoded_context["input_ids"][0][idx])
                alt_decoded = "".join(decoded.split(" "))  # in case whitespace was introduced during decoding
                # check if the word is found in the decoded context, return indices if there is a match
                if (re.search(word_pattern, decoded)) or (re.search(word_pattern, alt_decoded)):
                    return idx#
        raise ValueError(f"\"{word}\" not found in \"{tokenizer.decode(encoded_context['input_ids'][0][1:])}\"")
       


def get_word_embedding_by_layer(tokenizer, embedding_model, context: str, word: str, layers: list[int]) -> torch.Tensor:

    """ Function to extract the embedding of a word in its context for specified layers.
    
    :param tokenizer: the tokenizer for the model
    :param embedding_model: the model to use for embedding the input, should be set to eval mode
    :param context: the context for the word (phrase/sentence)
    :param word: the word to look for
    :param layers: the list of layers for extracting the embedding
    :return: the layerwise stacked word embedding
    """
    
    # encoding the context
    encoded_context = tokenizer.encode_plus(context, return_tensors="pt", truncation = True).to(embedding_model.device)

    # retrieving the indices of the term in the encoded context
    word_idx = get_word_idx(tokenizer, encoded_context, word)
            
    # passing the context to the model embedding and retrieving the hidden_states
    embedding_model.eval()
    with torch.no_grad():
        hidden_states = embedding_model(**encoded_context).hidden_states

    # retrieving the hidden states for selected layers
    embeddings_by_layer = [hidden_states[layer][0] for layer in layers]
    
    # retrieving the word embedding by indexing for the word 
    word_embeddings_by_layer = [layer[word_idx] for layer in embeddings_by_layer]
    
    # and averaging over subword embeddings if applicable
    word_embeddings_by_layer =[layer.mean(dim=0).to("cpu") for layer in word_embeddings_by_layer]
    
    return torch.stack(word_embeddings_by_layer)
    
    
    
def load_model_for_embedding_retrieval(embedding_model_name, device, hf_token=None):

    # function to load model in eval mode and to output all hidden states

    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name, use_fast = True, token = hf_token)
    
    try:
        embedding_model = AutoModel.from_pretrained(embedding_model_name, output_hidden_states=True, device_map="auto", token = hf_token)
    except:
        # for models where device map has not been implemented (and also not necessary, because these older models are small)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding_model = AutoModel.from_pretrained(embedding_model_name, output_hidden_states=True, token = hf_token).to(device)
    
    embedding_model.eval();

    return tokenizer, embedding_model
    
    

def get_number_of_hidden_states(tokenizer, embedding_model): 

    # function to get the number of hidden states
    
    # test run of the model
    context = "test"
    encoded = tokenizer.encode_plus(context, return_tensors="pt").to(embedding_model.device)
    with torch.no_grad():
        hidden_states = embedding_model(**encoded).hidden_states
    n_hidden_states  = len(hidden_states)
        
    return n_hidden_states
    


def get_number_of_hidden_states_from_name(embedding_model_name, device):
    # function to get the number of hidden states from embedding_model_name

    tokenizer, embedding_model = load_model_for_embedding_retrieval(embedding_model_name, device)
    n_hidden_states = get_number_of_hidden_states(tokenizer, embedding_model)

    return n_hidden_states   



def get_examples_with_term(term, examples):
    """ Filter examples which contain the term of interested """
    
    return([example for example in examples if term in example.split(" ")])


### Wordnet Helper Functions


def get_wn_synset(term, pos, sense):
    """Get synset for a given term and a given sense.
    Return synset's name
    E.g., term="polite", sense=0 => synset="polite.a.01", name="polite"
    """
    wn_pos = None
    if pos == 'ADJECTIVE':
        wn_pos = wn.ADJ
    elif pos == 'NOUN':
        wn_pos = wn.NOUN

    wn_synset = wn.synsets(term, pos=wn_pos)[sense - 1]

    return wn_synset.name()
    
    
def get_wn_antonym(wn_synset):
    """Get the antonym of a given synset.
    """
    wn_synset = wn.synset(wn_synset)
    antonyms_lemmas = wn_synset.lemmas()[0].antonyms()

    antonym = np.NaN
    if len(antonyms_lemmas) > 0:
        antonym = antonyms_lemmas[0].synset().name()
    
    return antonym
    
    
def get_wn_definition(wn_synset):
    """Get WordNet's definition of a given synset.
    """
    wn_synset = wn.synset(wn_synset)
    return wn_synset.definition()


def get_wn_examples(wn_synset):
    """Get WordNet's sentence examples of a given synset in lower case.
    """
    wn_synset = wn.synset(wn_synset)
    lower_case_examples = list(map(str.lower, wn_synset.examples()))
    
    return lower_case_examples
    

def replace_wn_terms(term, example_list, synset):
    
    """
    Function to replace terms in example sentences.
    
    At times wordnet examples do not contain the term of interest but a synonym. 
    If the example contains the synset name, we replace it by the term. (e.g. “the right answer” for “Correct.a.01”)
    If term, synset name and the word in the example are three different tokens, we flag the term to change the example manually. (e.g. sentence = 'a very unsure young man', synset = diffident.a.02, term = timid)
    """

    are_3_diff_tokens = False
    altered_examples = []
    synset_word = synset.split('.')[0]

    for example in example_list:
        
        
        if term not in example and synset_word in example:
            # 1. Case where *term* NOT in *example* sentence, but its synset
            # e.g., 'stood apart with aloof dignity', synset = aloof.s.01, term = distant
            # => replace *synset word* in example sentence with given *term*
            new_sent = example.replace(synset_word, term)
            altered_examples.append(new_sent)
        
        else:
            # 2. Case for 3 different tokens: sentence = 'a very unsure young man', synset = diffident.a.02, term = timid
            # => add to list & manually change after
            if term not in example and synset_word not in example:
                are_3_diff_tokens = True

            # 3. Case for matches
            altered_examples.append(example)

    return altered_examples, are_3_diff_tokens
    

def get_all_noun_and_adjective_synsets(term):
    """ function to retrieve all synset names, definitions, examples for a term when we do not know the specific word sense.
    """

    synsets = [wn_synset.name() for wn_synset in wn.synsets(term) if wn_synset.pos() in ["n", "a", "s"]]
    definitions = []
    examples = []

    for synset in synsets:
    
        synset_examples = list(map(str.lower, wn.synset(synset).examples()))
        synset_examples, diff = replace_wn_terms(term, synset_examples, synset)
        synset_examples = [example for example in synset_examples if term in example]
        examples = examples + synset_examples
    
        definitions.append(wn.synset(synset).definition())
    
    return(synsets, definitions, examples)


# Defining the templates for gendered terms or names

TEMPLATES = ['this is ', 'that is ', 'there is ', 'the person is ', 'here is ', ' is here', ' is there']

def fill_template(gendered_term, TEMPLATE, isNNP=False):
    """ Function to fill a template with a gendered term/name.
    """
    # if it is a noun
    if TEMPLATE.startswith(" is"):
        # add article "the" before noun
        if nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
            return('the ' + gendered_term + TEMPLATE)
        else:
            return(gendered_term + TEMPLATE)
    elif nltk.pos_tag([gendered_term])[0][1] in ['NN', 'JJ'] and isNNP is False:
        # add article "the" before noun
        TEMPLATE += 'the '
        return(TEMPLATE + gendered_term)
    else:
        return(TEMPLATE + gendered_term)
    

### simple function for statistical analysis

def get_stats(values):

    values = list(values)
    mean = np.mean(values)
    std = np.std(values)
    return mean, std

def standardize(values):

    values = list(values)
    mean, std = get_stats(values)
    return (values - mean)/std

def standardize_new_values(values, mean, std):

    values = list(values)
    return [(value- mean)/std for value in values]


def get_diff(values1,values2):
    
    values1 = list(values1)
    values2 = list(values2)
    
    mean1, std1 = get_stats(values1)
    mean2, std2 = get_stats(values2)
    diff = mean1 - mean2
    test_stats, diff_p = ttest_ind(values1, values2)
    diff_abs = abs(diff)
    
    return mean1, std1, mean2, std2, diff, diff_p, diff_abs