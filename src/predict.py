
import argparse
import json
import torch
import simpletransformers as st
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from utils import format_squad
import argparse


def main(context, question):

    to_predict = [
    {
        "context": context,
        "qas": [
            {
                "question": question,
                "id": "0",
            }
        ],
    }
    ]

    predictions, raw_outputs = model.predict(to_predict)
    predictions = predictions[0]['answer']
    probability = raw_outputs[0]['probability']
    index = probability.index(max(probability))
    print(f"Answer: {predictions[index]}")



if __name__ == "__main__":
    model = QuestionAnsweringModel('camembert', 'models/camembert-base/best_model', args={'use_multiprocessing': False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True, help="The question to answer.")
    parser.add_argument("--context", type=str, required=True, help="The context to search for the answer in.")
    args = parser.parse_args()
    context = args.context
    question = args.question
    main(context, question)