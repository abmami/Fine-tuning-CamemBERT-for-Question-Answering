from flask import Flask, jsonify, render_template, request
from simpletransformers.question_answering import QuestionAnsweringModel
import os 
project_path = os.path.join(os.path.dirname(__file__), os.pardir)
app = Flask('demo', template_folder='')
# Set the root path explicitly
#app.root_path = project_path + '/demo'
# Load the Camembert model
model = QuestionAnsweringModel('camembert', 'models/camembert-base/best_model', args={'use_multiprocessing': False})

# Serve the index.html file on the main page
@app.route('/')
def index():
    return render_template('index.html')

# Define a predict endpoint for the API
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input question and context from the request
    data = request.get_json()
    question = data['question']
    context = data['context']
    
    # Use the fine-tuned model to predict the answer
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
    predictions = model.predict(to_predict)
    answer = predictions[0][0]['answer']
    
    # Return the predicted answer as a JSON response
    return jsonify({'answer': answer})

if __name__ == '__main__':
    print("Starting Flask server...")
    print("Loading the Camembert model...")
    print("App is running on http://localhost:5000")
    app.run(debug=True)
