import numpy as np

"""
A Sample in the data set
"""


class Sample:

    def __init__(self, id, tokenized_document, tokenized_document_to_compare, label):
        self.id = id

        self.document = str(" ".join(tokenized_document))
        self.tokenized_document = tokenized_document

        self.document_to_compare = " ".join(tokenized_document_to_compare)
        self.tokenized_document_to_compare = tokenized_document_to_compare

        self.label = label

    def get_document(self, in_lower_case=False):
        if in_lower_case:
            return self.document.lower()
        else:
            return self.document

    def get_tokenized_document(self, in_lower_case=False):
        if in_lower_case:
            return [token.lower() for token in self.tokenized_document]
        else:
            return self.tokenized_document

    def get_document_to_compare(self, in_lower_case=False):
        if in_lower_case:
            return self.document_to_compare.lower()
        else:
            return self.document_to_compare

    def get_tokenized_document_to_compare(self, in_lower_case=False):
        if in_lower_case:
            return [token.lower() for token in self.tokenized_document_to_compare]
        else:
            return self.tokenized_document_to_compare

    def get_label(self):
        return self.label

    def get_id(self):
        return self.id


"""
A data set
"""


class Dataset:

    def __init__(self, samples):
        # default auto initialization into respective variable for ease of use
        self.ids = list()
        self.samples = samples
        self.documents = list()
        self.label_by_id = dict()

        labels = None
        for sample in samples:
            self.ids.append(sample.get_id())

            self.documents.append(sample.get_document())
            self.documents.append(sample.get_document_to_compare())

            if sample.get_label() is not None:
                if labels is None:
                    labels = list()
                labels.append(sample.get_label())
                self.label_by_id[sample.get_id()] = sample.get_label()

        self.X = None
        if labels is not None and len(labels) > 0:
            self.y = np.asarray(labels)
            unique, counts = np.unique(self.y, return_counts=True)
            print("Samples By Label: ", dict(zip(unique, counts)))
        else:
            self.y = None

        # extra variables
        self.samples_with_tf_idf_values = None

    def get_ids(self):
        return self.ids

    def get_samples(self):
        return self.samples

    def get_Xy(self):
        return self.X, self.y

    def get_y(self):
        return self.y

    def get_X(self):
        return self.X

    def get_documents(self, in_lower_case=False, remove_stop_words=False):
        to_return = self.documents
        if in_lower_case:
            to_return = [str(doc).lower() for doc in self.documents]
        return to_return




