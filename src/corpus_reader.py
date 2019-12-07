from dataset import Sample
import nltk


class CorpusReader:

    def read_data(self, file):
        samples = list()
        with open(file,encoding="utf8") as data:
            idx = 0
            for row in data.readlines():
                row = row.split("\t")
                if idx != 0:
                    if len(row) == 4:
                        label = int(row[-1].replace("\n", ""))
                        if label == 0:
                            label = 1
                        samples.append(Sample(row[0],
                                              nltk.word_tokenize(str(row[1])),
                                              nltk.word_tokenize(str(row[2])), label))
                    else:
                        try:
                            samples.append(Sample(row[0],
                                                  nltk.word_tokenize(str(row[1])),
                                                  nltk.word_tokenize(str(row[2])), None))
                        except IndexError as e:
                            id = row[0]
                            if len(row) < 3:
                                p1, p2 = row[1].split("\t")
                                samples.append(Sample(id,
                                                      nltk.word_tokenize(str(p1).lower()),
                                                      nltk.word_tokenize(str(p2).lower()), None))
                idx = idx+1
        print(str(len(samples)) + " samples loaded and tokenized from " + file)
        return samples
