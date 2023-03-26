import os 
from dotenv import find_dotenv, load_dotenv


project_path = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_path, '.env')
load_dotenv(dotenv_path)

RAW_DATA_PATH = project_path + os.getenv('RAW_DATA_PATH')
PROCESSED_DATA_PATH = project_path + os.getenv('PROCESSED_DATA_PATH')


def format_squad(input_data):
    data = []
    for article in input_data:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                id = qa["id"]
                for answer in qa["answers"]:
                    answer_text = answer["text"]
                    answer_start = answer["answer_start"]
                    data.append({"context": context, "qas": [{"question": question, "id":id, "answers": [{"text": answer_text, "answer_start": answer_start}]}]})
    return data
