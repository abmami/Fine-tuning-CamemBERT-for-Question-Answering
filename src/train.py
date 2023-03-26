import torch
import simpletransformers as st
import json 
from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from utils import format_squad, PROCESSED_DATA_PATH
import os

def load_datasets():
    # Load train, validation sets
    train_path = os.path.join(PROCESSED_DATA_PATH, 'train.json')
    val_path = os.path.join(PROCESSED_DATA_PATH, 'val.json')

    with open(train_path, encoding="utf-8") as f:
        train = json.load(f)

    with open(val_path, encoding="utf-8") as f:
        valid = json.load(f)

    return format_squad(train), format_squad(valid)



def main():
    # Load train, validation sets
    train, valid = load_datasets()

    # Initialize model
    model_type = "camembert"
    model_name = "camembert-base"

    train_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "use_cached_eval_features": True,
    "output_dir": f"models/{model_name}",
    "best_model_dir": f"models/{model_name}/best_model",
    "evaluate_during_training": True,
    "max_seq_length": 128,
    "num_train_epochs": 5,
    "evaluate_during_training_steps": 1000,
    "use_wandb": False,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "n_best_size":3,
    "train_batch_size": 16, # Reduced to 16 after getting CUDA out of memory error
    "eval_batch_size": 16, # Reduced to 16 after getting CUDA out of memory error
    }

    # Configure the model 
    model_args = QuestionAnsweringArgs()
    model_args.train_batch_size = 16
    model_args.evaluate_during_training = True
    model_args.n_best_size=3
    model_args.num_train_epochs=5

    # Create a QuestionAnsweringModel
    model = QuestionAnsweringModel(
    model_type,model_name, args=train_args, use_cuda=True)

    # Train the model
    model.train_model(train, eval_data=valid)

    # Evaluate the model
    result, texts = model.eval_model(valid)
    with open('results.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print('Results are saved to results.json')


if __name__ == "__main__":
    main()

    