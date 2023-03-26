import json 
import os 
import argparse
from pathlib import Path
from utils import RAW_DATA_PATH, PROCESSED_DATA_PATH


def load_data(data_path, train_size, val_size):

    # Read data from json file
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Split data into train, validation and test sets
    data = data['data']
    train = data[:int(train_size*len(data))]
    val = data[int(train_size*len(data)):int((train_size+val_size)*len(data))]
    test = data[int((train_size+val_size)*len(data)):]
    
    return train, val, test

def save_data(train, val, test, path):

    # Initialize paths
    train_path = os.path.join(path, 'train.json')
    val_path = os.path.join(path, 'val.json')
    test_path = os.path.join(path, 'test.json')

    # Save train, validation and test sets into json files
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
        print('Train set saved to {}'.format(train_path))

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val, f, indent=4, ensure_ascii=False)
        print('Validation set saved to {}'.format(val_path))

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
        print('Test set saved to {}'.format(test_path))




def main(input_file):
    # Your code goes here
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    # Load data from json file

    input_filepath = os.path.join(RAW_DATA_PATH, input_file)
    train, val, test = load_data(input_filepath, 0.2, 0.05)

    # Save data into json files
    save_data(train, val, test, PROCESSED_DATA_PATH)

if __name__ == '__main__':
    # Initialize argument parser

    parser = argparse.ArgumentParser(description='Description of your script')

    # Add input file argument
    parser.add_argument('-i', '--input_file', type=str, required=True, help='Input file inside data/raw directory')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call main function with input file argument


    main(args.input_file)

