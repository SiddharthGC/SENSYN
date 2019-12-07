import numpy as np
import spacy
spacy_nlp = spacy.load("en_core_web_sm")
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from dataset import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from ioutil.corpus_reader import CorpusReader

stopwords = set(stopwords.words('english'))

# data
TRAIN_DATA_X_FILE = '../data/tmp/train-set-X'
TRAIN_DATA_y_FILE = '../data/tmp/train-set-y'
VALID_DATA_X_FILE = '../data/tmp/dev-set-X'
VALID_DATA_y_FILE = '../data/tmp/dev-set-y'

TRAIN_DATA_FILE = '../data/train-set.txt'
VALID_DATA_FILE = '../data/dev-set.txt'

TRAIN_DATA_FEATURES_CSV = "../data/csv/train-data.csv"
VALID_DATA_FEATURES_CSV = "../data/csv/dev-data.csv"

wordnetLemmatizer = WordNetLemmatizer()


def abs_diff_similarity(doc_pos_tags, doc_pos_tags_to_compare):
    # all tokens
    f1 = abs(len(doc_pos_tags) - len(doc_pos_tags_to_compare)) / float(len(doc_pos_tags) + len(doc_pos_tags_to_compare))

    # all nouns
    c1 = len([1 for token_pos_tag in doc_pos_tags if token_pos_tag[1].startswith('N')])
    c2 = len([1 for token_pos_tag in doc_pos_tags_to_compare if token_pos_tag[1].startswith('N')])
    if c1 == 0 and c2 == 0:
        f2 = 0
    else:
        f2 = abs(c1 - c2) / float(c1 + c2)

    # all verbs
    c1 = len([1 for token_pos_tag in doc_pos_tags if token_pos_tag[1].startswith('V')])
    c2 = len([1 for token_pos_tag in doc_pos_tags_to_compare if token_pos_tag[1].startswith('V')])
    if c1 == 0 and c2 == 0:
        f3 = 0
    else:
        f3 = abs(c1 - c2) / float(c1 + c2)

    # all adverbs
    c1 = len([1 for token_pos_tag in doc_pos_tags if token_pos_tag[1].startswith('R')])
    c2 = len([1 for token_pos_tag in doc_pos_tags_to_compare if token_pos_tag[1].startswith('R')])
    if c1 == 0 and c2 == 0:
        f4 = 0
    else:
        f4 = abs(c1 - c2) / float(c1 + c2)

    # all adjectives
    c1 = len([1 for token_pos_tag in doc_pos_tags if token_pos_tag[1].startswith('J')])
    c2 = len([1 for token_pos_tag in doc_pos_tags_to_compare if token_pos_tag[1].startswith('J')])
    if c1 == 0 and c2 == 0:
        f5 = 0
    else:
        f5 = abs(c1 - c2) / float(c1 + c2)

    return f1, f2, f3, f4, f5


def compute_path_sim_score(subj_doc, subj_doc_compare):
    count = 0
    score = 0
    for syn in wn.synsets(subj_doc):
        result = []
        for syn1 in wn.synsets(subj_doc_compare):
            if syn.path_similarity(syn1) == None:
                result.append(0)
            else:
                result.append(syn.path_similarity(syn1))
        if result:
            best_score = max(result)
            count += 1
            score += best_score
    if count > 0:
        score = score / count
    return score


def comparesubjects(subjects_doc,subjects_doc_compare):
    len_doc_subjects = len(subjects_doc)
    len_doc_subjects_compare = len(subjects_doc_compare)
    if len_doc_subjects_compare !=0  and len_doc_subjects!=0:
        scores = []
        for subj_doc in subjects_doc:
            for subj_doc_compare in subjects_doc_compare:
                scores.append(compute_path_sim_score(subj_doc,subj_doc_compare))
        return max(scores)/len_doc_subjects
    else:
        return 1


def compareobjects(objects_doc, objects_doc_compare):
    len_doc_objects = len(objects_doc)
    len_doc_objects_compare = len(objects_doc_compare)
    if len_doc_objects_compare != 0 and len_doc_objects != 0:
        scores = []
        for obj_doc in objects_doc:
            for obj_doc_compare in objects_doc_compare:
                scores.append(compute_path_sim_score(obj_doc, obj_doc_compare))
        return max(scores) / len_doc_objects
    else:
        return 1


def compare_verbs_docs(verbs_doc, verbs_doc_compare):
    len_doc_verbs = len(verbs_doc)
    len_doc_verbs_compare = len(verbs_doc_compare)
    if len_doc_verbs_compare != 0 and len_doc_verbs != 0:
        scores = []
        for verb_doc in verbs_doc:
            for verb_doc_compare in verbs_doc_compare:
                scores.append(compute_path_sim_score(verb_doc, verb_doc_compare))
        return max(scores) / len_doc_verbs
    else:
        return 1


def compare_nouns_docs(nouns_doc, nouns_doc_compare):
    len_doc_nouns = len(nouns_doc)
    len_doc_nouns_compare = len(nouns_doc_compare)
    if len_doc_nouns_compare != 0 and len_doc_nouns != 0:
        scores = []
        for noun_doc in nouns_doc:
            for noun_doc_compare in nouns_doc_compare:
                scores.append(compute_path_sim_score(noun_doc, noun_doc_compare))
        return max(scores) / len_doc_nouns
    else:
        return 1


def compareheads(head_doc_lemma,head_doc_to_compare_lemma,head_doc_pos,head_doc_to_compare_pos):
    count = 0
    score = 0
    map_tag = {'NN':'n','JJ':'a','VB':'v','RB':'r'}
    doc_pos = None
    doc_compare_pos = None
    if head_doc_pos in map_tag:
        doc_pos = map_tag[head_doc_pos]

    if head_doc_to_compare_pos in map_tag:
        doc_compare_pos = map_tag[head_doc_to_compare_pos]

    for syn in wn.synsets(head_doc_lemma, doc_pos):
        result = []
        for syn1 in wn.synsets(head_doc_to_compare_lemma, doc_compare_pos):
            if syn.path_similarity(syn1) == None:
                result.append(0)
            else:
                result.append(syn.path_similarity(syn1))
        if result:
            best_score = max(result)
            count += 1
            score += best_score
    if count>0:
        score = score/count
    return score


def calculate_tfidf(sentence1, sentence2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([sentence1, sentence2])
    dense = vectors.todense()
    dense_list = dense.tolist()
    dot = np.dot(dense_list[0], dense_list[1])
    norm1 = np.linalg.norm(dense_list[0])
    norm2 = np.linalg.norm(dense_list[1])
    sim = dot / (norm1 * norm2)
    return sim


def generate_features(dataset):
    X, y = None, dataset.get_y()
    idx = 0
    csv_rows = []
    for sample in dataset.get_samples():
        tokenized_doc_with_pos_tags = list()
        tokenized_doc_with_pos_tags_to_compare = list()
        head_doc_lemma = None
        head_doc_to_compare_lemma = None
        head_doc_pos = None
        head_doc_to_compare_pos = None
        subjects_doc = []
        subjects_doc_compare = []
        objects_doc = []
        objects_doc_compare = []
        token_heads = {}
        count_head = 0
        count_tokens_compare = 0
        count_tokens = 0
        sentence1 = sample.get_document(in_lower_case=True)
        sentence2 = sample.get_document_to_compare(in_lower_case=True)
        doc = spacy_nlp(sentence1)
        verbs_doc = []
        verbs_doc_compare = []
        nouns_doc = []
        nouns_doc_compare = []

        for token in doc:
            tokenized_doc_with_pos_tags.append([token.text, token.tag_])
            if(token.dep_ == "ROOT"):
                head_doc_lemma = token.text
                head_doc_pos = token.tag_
            if(token.pos_ == "VERB"):
                verbs_doc.append(token.lemma_)
            if (token.pos_ == "NOUN"):
                nouns_doc.append(token.lemma_)
            if(token.dep_ == "nsubj"):
                subjects_doc.append(token.text)
            if(token.dep_ == "nobj" or token.dep_ == "dobj"):
                objects_doc.append(token.text)
            if(token.text not in stopwords and len(token.text) > 1):
                token_heads[token.text] = token.head.text
        count_tokens += 1


        doc_to_compare = spacy_nlp(sentence2)
        for token in doc_to_compare:
            tokenized_doc_with_pos_tags_to_compare.append([token.text, token.tag_])
            if(token.dep_ == "ROOT"):
                head_doc_to_compare_lemma = token.text
                head_doc_to_compare_pos = token.tag_
            if (token.pos_ == "VERB"):
                verbs_doc_compare.append(token.lemma_)
            if (token.pos_ == "NOUN"):
                nouns_doc_compare.append(token.lemma_)
            if(token.dep_ == "nsubj"):
                subjects_doc_compare.append(token.text)
            if (token.dep_ == "nobj" or token.dep_ == "dobj"):
                objects_doc_compare.append(token.text)
            if (token.text not in stopwords and len(token.text) > 1):
                if(token.text in token_heads and token_heads[token.text] == token.head.text):
                    count_head+=1
            count_tokens_compare+=1

        common_tokens = set([token_pos_tag[0] for token_pos_tag in tokenized_doc_with_pos_tags if token_pos_tag[0] not in stopwords and len(token_pos_tag[0]) > 1])\
            .intersection(set([token_pos_tag[0] for token_pos_tag in tokenized_doc_with_pos_tags_to_compare if token_pos_tag[0] not in stopwords and len(token_pos_tag[0]) > 1]))
        num_common_tokens = len(common_tokens)
        # 1. unigram overlap
        unigram_overlap = (2 * num_common_tokens) / float(len(set([token_pos_tag[0] for token_pos_tag in tokenized_doc_with_pos_tags if token_pos_tag[0] not in stopwords and len(token_pos_tag[0]) > 1])) + len(set([token_pos_tag[0] for token_pos_tag in tokenized_doc_with_pos_tags_to_compare if token_pos_tag[0] not in stopwords and len(token_pos_tag[0]) > 1])))
        # 2. abs diff.
        abs_diff = abs_diff_similarity(tokenized_doc_with_pos_tags, tokenized_doc_with_pos_tags_to_compare)
        # 3. compare heads of two sentences
        heads_are_same_measure = compareheads(head_doc_lemma,head_doc_to_compare_lemma,head_doc_pos,head_doc_to_compare_pos)
        #4. compare subjects in both sentences
        subjects_measure = comparesubjects(subjects_doc,subjects_doc_compare)
        # 5.compare objects in both sentences
        objects_measure = compareobjects(objects_doc, objects_doc_compare)
        #6. compare heads of all the words in the dependency tree except stopwords
        compare_head_words_measure = count_head/(count_tokens+count_tokens_compare)
        # 7. compare verbs
        compare_verbs_both_sentences = compare_verbs_docs(verbs_doc, verbs_doc_compare)
        # 8. compare nouns in both the sentences
        compare_nouns_both_sentences = compare_nouns_docs(nouns_doc, nouns_doc_compare)
        # 9. compute tf-idf
        tf_idf_measure = calculate_tfidf(sentence1, sentence2)


        # compose X
        x_i = np.hstack((np.asarray((tf_idf_measure,compare_nouns_both_sentences, unigram_overlap, heads_are_same_measure, subjects_measure,objects_measure,compare_head_words_measure,compare_verbs_both_sentences)), np.asarray(abs_diff)))
        # x_i = np.asarray((unigram_overlap, heads_are_same_measure, subjects_measure, compare_head_words_measure))
        csv_line = sample.get_id() + ","
        if y is not None:
            csv_line += str(y[idx]) + ","
        csv_line += ','.join(['%.5f' % num for num in x_i])
        csv_rows.append(csv_line)
        if X is None:
            X = x_i
        else:
            X = np.vstack((X, x_i))
        idx = idx + 1
    return X, y, csv_rows



def load_data():
    """
    Loads data as dictionary format: id = ((text, TextToCompare), label)
    By, text is converted to lower case
    :return:
    """
    cr = CorpusReader()
    train_samples = Dataset(cr.read_data(TRAIN_DATA_FILE))
    valid_samples = Dataset(cr.read_data(VALID_DATA_FILE))
    print("Data load completed")
    return {'train': train_samples, 'valid': valid_samples}


def main():
    data_partitions = load_data()
    # 1. Feature Engineering
    print("Computing Train data features")
    X_train, y_train, csv_rows = generate_features(data_partitions["train"])

    # save features and gold of train
    csv_fp = open(TRAIN_DATA_FEATURES_CSV, mode='w')
    csv_fp.write("id,GT," + ",".join(["f" + str(idx) for idx in range(np.shape(X_train)[1])]) + "\n")
    for line in csv_rows:
        csv_fp.write(line+"\n")
    csv_fp.close()

    np.save(TRAIN_DATA_X_FILE, X_train)
    np.save(TRAIN_DATA_y_FILE, y_train)
    print("Saved Train numpy arrays (X, y) to ..data/tmp/")

    print("Computing Dev data features")
    X_valid, y_valid, csv_rows_valid = generate_features(data_partitions["valid"])

    # save features and gold of train
    csv_fp = open(VALID_DATA_FEATURES_CSV, mode='w')
    csv_fp.write("id,GT," + ",".join(["f" + str(idx) for idx in range(np.shape(X_valid)[1])]) + "\n")
    for line in csv_rows_valid:
        csv_fp.write(line + "\n")
    csv_fp.close()

    np.save(VALID_DATA_X_FILE, X_valid)
    np.save(VALID_DATA_y_FILE, y_valid)
    print("Saved Dev numpy arrays (X, y) to ..data/tmp/")


if __name__ == '__main__':
    main()