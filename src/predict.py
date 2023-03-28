import argparse
from utils import make_prediction

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="The question to answer.")
    parser.add_argument("--context", type=str, required=True, help="The context to search for the answer in.")
    args = parser.parse_args()
    context = args.context
    question = args.question
    print("Answer:", make_prediction(context, question))
    

