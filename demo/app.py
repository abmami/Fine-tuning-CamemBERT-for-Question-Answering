from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import simpletransformers as st
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from fastapi.staticfiles import StaticFiles
app = FastAPI()
templates = Jinja2Templates(directory="demo/templates")

app.mount("/static", StaticFiles(directory="demo/static"), name="static")
model = QuestionAnsweringModel('camembert', 'models/camembert-base/best_model', args={'use_multiprocessing': False})

class QARequest(BaseModel):
    question: str
    context: str


def make_prediction(context, question):
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
    return predictions[index]

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(qa_request: QARequest):
    question = qa_request.question
    context = qa_request.context
    answer = make_prediction(context, question)
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, port=8000)
