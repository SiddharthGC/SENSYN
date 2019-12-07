from dataset import Dataset
from ioutil.corpus_reader import CorpusReader
from model.ml_models import SVRModel, GBModel
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np
np.random.seed(234)
import math

TRAIN_DATA_FILE = '../data/train-set.txt'
VALID_DATA_FILE = '../data/dev-set.txt'
VALID_DATA_FEATURES_CSV = "../data/csv/dev-data.csv"
VALID_DATA_FEATURES_W_PREDS_CSV = "../data/csv/dev-data-w-preds.csv"
VALID_DATA_PRED_FILE = '../data/predictions/dev-set-predicted-answers.txt'
GB_MODEL_FILE = "../data/models/gb-model.sav"
SVR_MODEL_FILE = "../data/models/svr-model.sav"
TRAIN_DATA_X_FILE = '../data/tmp/train-set-X.npy'
TRAIN_DATA_y_FILE = '../data/tmp/train-set-y.npy'
VALID_DATA_X_FILE = '../data/tmp/dev-set-X.npy'
VALID_DATA_y_FILE = '../data/tmp/dev-set-y.npy'


def load_data():
    """
    Loads data as dictionary format: id = ((text, TextToCompare), label)
    By, text is converted to lower case
    :return:
    """
    cr = CorpusReader()
    valid_samples = Dataset(cr.read_data(VALID_DATA_FILE))
    print("Data load completed")
    return {'valid': valid_samples}


def main():
    data_partitions = load_data()
    X_train, y_train = np.load(TRAIN_DATA_X_FILE), np.load(TRAIN_DATA_y_FILE)
    X_valid, y_valid = np.load(VALID_DATA_X_FILE), np.load(VALID_DATA_y_FILE)

    model = GBModel(None)
    params = {'n_estimators': 200, 'max_depth': 4, 'learning_rate': 0.05, 'loss': 'huber',
              'min_samples_split': 6}
    model.train(X_train, y_train, params)
    joblib.dump(model.get_model(), GB_MODEL_FILE)
    # model = SVRModel()
    # model.train(X_train, y_train)
    # joblib.dump(model.get_model(), SVR_MODEL_FILE)
    # Accuracy on validation data
    model = GBModel(joblib.load(GB_MODEL_FILE))
    # model = GBModel(joblib.load(SVR_MODEL_FILE))
    y_valid_pred = model.predict(X_valid)

    # csv features w/ predicted labels
    fp_out = open(VALID_DATA_FEATURES_W_PREDS_CSV, mode='w')
    idx = 0
    rows = open(VALID_DATA_FEATURES_CSV).readlines()
    fp_out.write(rows[0].replace("\n", "") + "PT,PTR\n")
    for line in rows[1:]:
        pred_label = min(5, round(y_valid_pred[idx]))
        fp_out.write(line.replace("\n", "") + "," + str(y_valid_pred[idx]) + "," + str(int(pred_label)) + "\n")
        idx = idx + 1
    fp_out.close()

    # Validation Data Output File
    fp_out = open(VALID_DATA_PRED_FILE, mode='w')
    fp_out.write("id" + "\t" + "Predicted Tag" + "\n")
    for idx in range(0, len(data_partitions['valid'].get_ids())):
        s_id = data_partitions['valid'].get_ids()[idx]
        pred_label = min(5, round(y_valid_pred[idx]))
        fp_out.write(s_id + "\t" + str(int(pred_label)) + "\n")
    print("output written to " + VALID_DATA_PRED_FILE)
    unique, counts = np.unique(np.asarray([min(5, round(i)) for i in y_valid_pred]), return_counts=True)
    print("Predictions By Label: ", dict(zip(unique, counts)))
    print(accuracy_score(y_valid, np.asarray([min(5, round(i)) for i in y_valid_pred])))

if __name__ == '__main__':
    main()