Fine-tuning CamemBERT on a subset of FQuAD dataset for Question Answering
==============================

This repository contains code for fine-tuning CamemBERT, a French version of the BERT language model, on a portion of the FQuAD (French Question Answering Dataset) for Question Answering (QA) tasks.

### Dataset
The FQuAD dataset is a collection of questions and answers in French. It contains over 25,000 question-answer pairs and covers a wide range of topics, including history, science, and literature. 

The CamemBERT model was fine-tuned on this dataset to create a French question-answering system. Due to the high computation requirements of the model, a CUDA-enabled GPU was used to train and evaluate the model, and only a subset of the dataset containing ~4,500 question-answer pairs was used.

### Requirements 
- Python 3.10
- CUDA-enabled GPU
- CUDA 11.7 (Download from [here](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local))
- Torch 2.0.0 

You can install the required Python packages using pip after creating a virtual environment:
```bash
  pip install -r requirements.txt
```
*Note:* 
If there is matching distribution found for "torch==2.0.0+cu117", you can install using this command:*
```bash
  pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html 
```

### Prepare datasets
To prepare and split the data stored at "data/raw/", run the make_dataset.py script
```bash
  python src/make_dataset.py -i data.json
```

### Fine-tuning
To fine-tune CamemBERT on a portion of the FQuAD dataset, run the train.py script:

```bash
  python src/train.py
```

The train.py script loads the dataset, prepares it for training, and fine-tunes the CamemBERT model. The trained model is saved to the models/ directory.

### Evaluate

The evaluation of the model is saved in a bit messy way. Results are saved in results.json in the root level of the repo. Other evaluations during training can be found at "/models/". Will be fixed in the future.

### Inference
To use the trained model for inference on new question-answer pairs, run the predict.py script:

```bash
  python src/predict.py --question "A question" --context "A context"
```

The predict.py script loads the saved model from the models/ directory and uses it to answer the provided question in the given context.

Example: 
```bash
   python src/predict.py --question "Quand le Doric est-il livré ?" --context "Avant la Première Guerre mondiale, l'International Mercantile Marine Co. commande aux chantiers Harland & Wolff la construction de plusieurs navires destinés à ses compagnies. Les deux premiers, le Regina et le Pittsburgh ébauchés en 1913, sont achevés après la guerre et mis en service au début des années 1920. Le Doric est le troisième navire construit sur ce modèle et un quatrième, légèrement plus grand, le Laurentic, suivra en 1927. La quille du Doric est posée bien après la guerre, en 1921, et sa construction est rapide, puisqu'il est lancé dès le 8 août 1922 et livré le 29 mai 1923."
```
```bash
Output: Answer: ['', 'début des années 1920.']
```
Note: The answer to the question in the example is obviously wrong, but the model should give better results if trained for more epochs. In this example, the model was trained for 5 epochs only.

### Demo

The fine-tuned CamemBERT model was deployed using Flask. You can try the demo by running app.py
```bash
  python demo/app.py
```
Once the server is running, open a web browser and go to the URL http://localhost:5000.

You should see the home page of the Flask app. Enter a question and context in the input fields and click on the "Ask Me" button to see the predicted answer.

![alt text](demo/demo.png)


### References 

FQuAD: French Question Answering Dataset - 2020arXiv200206071

