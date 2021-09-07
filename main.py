import argparse
import os

import pandas as pd

from jobs import train, test
from preprocess import preprocess_inputs

parser = argparse.ArgumentParser()

parser.add_argument('-dp', '--dataset_path', default='./', type=str,
                    help='Path to the dataset')
parser.add_argument('-mp', '--model_path', default='./', type=str,
                    help='Path to save models')
parser.add_argument('-dump', '--dump', default=True, type=bool,
                    help='Flag to dump the model')

args = parser.parse_args()

if __name__ == '__main__':
    input_data = pd.read_csv(os.path.join(args.dataset_path, 'train.csv'))

    X_train, X_test, y_train, y_test = preprocess_inputs(input_data, args.dump,
                                                         args.model_path,
                                                         test=False)

    model = train(X_train, y_train)
    test(X_test, y_test, model, args.dump, args.model_path)
