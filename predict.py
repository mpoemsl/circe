""" Predicts lexical semantic change ranking for a dataset with the context-free or context-dependent model. """

from models.context_dependent import make_classification_dataset, finetune_bert, extract_representations, compare_context_dependent_representations 
from models.context_free import preprocess_texts, train_word2vec, align_embeddings, compare_context_free_representations

import argparse
import os


parser = argparse.ArgumentParser(description="Predicts lexical semantic change ranking for a dataset with the context-free or context-dependent model.")

# general arguments
parser.add_argument("model_name", type=str, help="Model to use for prediction: One of {context-free, context-dependent}")
parser.add_argument("dataset_dir", type=str, help="Path to folder where the dataset is stored (must contain c1.txt, c2.txt and targets.tsv)")
parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite experiment folder if it already exists")

# context-free model arguments
parser.add_argument("--unfiltered", dest="filtered", action="store_false", help="Do not employ word frequenciy filtering")
parser.add_argument("--dim", type=int, default=300, help="Word2vec vector dimension")
parser.add_argument("--n_window", type=int, default=10, help="Word2vec window size")
parser.add_argument("--n_negative", type=int, default=1, help="Word2vec negative samples")

# context-dependent model arguments
parser.add_argument("--unmasked", dest="masked", action="store_false", help="Do not mask BERT training data")
parser.add_argument("--device", type=str, default="cpu", help="Name of device to train BERT model on: Usually one of {cpu, cuda}")
parser.add_argument("--n_epochs", type=int, default=1, help="Number of epochs for BERT training")
parser.add_argument("--bert_name", type=str, default="bert-base-multilingual-cased", help="Name of the pretrained BERT model to use")
parser.add_argument("--batch_size", type=int, default=10, help="Batch size for BERT training")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Ratio of warmup steps for BERT training")
parser.add_argument("--learning_rate", type=float, default=4e-5, help="Learning rate for BERT training")


def predict(model_name="", dataset_dir="", overwrite=False, filtered=True, **params):
    """ Predicts lexical semantic change ranking for a dataset with the context-free or the context-dependent model. """

    # organisational data and directory checks

    assert os.path.exists(dataset_dir), "Folder {} does not exist!".format(dataset_dir)

    if not dataset_dir.endswith("/"):
        dataset_dir += "/"
    
    dataset_name = dataset_dir.split("/")[-2]
    experiment_name = "{}_{}".format(model_name, dataset_name)

    print("Predicting with {} model for dataset {} ...".format(model_name, dataset_name))

    if not os.path.exists("experiments"):
        os.mkdir("experiments")

    experiment_dir = "experiments/" + experiment_name + "/"
    
    if os.path.exists(experiment_dir):
        assert overwrite, "Experiment folder already exists and --overwrite flag not used, aborting experiment ..."
    else:
        os.mkdir(experiment_dir)

    print("Experiment data will be stored in {} ...".format(experiment_dir))

    # prediction experiment execution

    if model_name == "context-free":
        
        print("Preprocessing texts ...")
        preprocess_texts(dataset_dir, experiment_dir, filtered)

        print("Training Word2Vec ...")
        train_word2vec(experiment_dir, **params)

        print("Aligning embeddings ...")
        align_embeddings(experiment_dir)

        print("Comparing context-free representations ...")
        compare_context_free_representations(dataset_dir, experiment_dir)
        
    elif model_name == "context-dependent":

        print("Making classification dataset ...")
        make_classification_dataset(dataset_dir, experiment_dir)

        print("Finetuning BERT ...")
        finetune_bert(experiment_dir, **params)

        print("Extracting representations ...")
        extract_representations(dataset_dir, experiment_dir, **params)

        print("Comparing context-dependent representations ...")
        compare_context_dependent_representations(dataset_dir, experiment_dir)

    else:

        raise Exception("'{}' is not a valid model name.".format(model_name))

    print("Finished experiment. Prediction can be found in {}.".format(experiment_dir + "prediction.tsv"))


if __name__ == "__main__":

    args = parser.parse_args()
    params = vars(args)
    predict(**params)

