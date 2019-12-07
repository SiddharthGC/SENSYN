from model.ml_models import SVRModel, GBModel
from dataset import Dataset
from ioutil.corpus_reader import CorpusReader
from sts_features import generate_features
from sklearn.externals import joblib
import math

TEST_DATA_FILE = '../data/test-set.txt'
TEST_DATA_PRED_FILE = '../data/predictions/test-set-predicted-answers.txt'

SVR_MODEL_FILE = "../data/models/svr-model.sav"
GB_MODEL_FILE = "../data/models/gb-model.sav"


def load_data():
    """
    Loads data as dictionary format: id = ((text, TextToCompare), label)
    By, text is converted to lower case
    :return:
    """
    cr = CorpusReader()
    test_samples = Dataset(cr.read_data(TEST_DATA_FILE))
    print("Data load completed")
    return {'test': test_samples}


def main():
    data_partitions = load_data()
    # 1. Feature Engineering
    print("Computing Test data features")
    X, y, csv_features = generate_features(data_partitions["test"])

    # model = SVRModel(joblib.load(SVR_MODEL_FILE))
    model = GBModel(joblib.load(GB_MODEL_FILE))
    y_pred = model.predict(X)

    # write to answers file
    fp_out = open(TEST_DATA_PRED_FILE, mode='w')
    fp_out.write("id" + "\t" + "Predicted Tag" + "\n")
    for idx in range(0, len(data_partitions['test'].get_ids())):
        s_id = data_partitions['test'].get_ids()[idx]
        pred_label = round(y_pred[idx])
        fp_out.write(s_id + "\t" + str(int(pred_label)) + "\n")
    print("output written to " + TEST_DATA_PRED_FILE)


if __name__ == '__main__':
    main()