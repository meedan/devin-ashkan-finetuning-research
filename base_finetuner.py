import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
from datasets import Dataset, DatasetDict

from torch import Tensor

from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

from sentence_transformers import SentenceTransformer, models

MODEL_NAME = 'thenlper/gte-base'
# Can also use other popular models like 
# google-bert/bert-base-uncased and sentence-transformers/all-MiniLM-L6-v2

class Finetuner:
    @staticmethod
    def evaluate_model(transformer, file_path="encoded_texts.json"):
        """
        Evaluates the performance of a transformer model by predicting car prices using embeddings.

        Args:
            transformer (SentenceTransformer): The transformer model used for generating embeddings.
            file_path (str): Path to the encoded texts JSON file.

        Returns:
            dict: A dictionary containing R² score, median absolute error, and mean absolute error.
        """
        dataset = [json.loads(e) for e in open(file_path).read().split("\n")[:-1]]
        np.random.shuffle(dataset)
        dataset = dataset[:20000]
        embeddings = transformer.encode([e["text"] for e in dataset])
        encoded_texts = []
        for i, row in enumerate(dataset):
            encoded_texts.append({"id": row["id"]["$oid"], "price": row["price"], "embedding": embeddings[i]})
        np.random.shuffle(encoded_texts)
        model = xgb.XGBRegressor(
            n_estimators=500,
            device='cuda',
            n_jobs=500
        )
        embeddings = pd.DataFrame([e["embedding"] for e in encoded_texts])
        target = pd.DataFrame([e["price"] for e in encoded_texts])
        ids = pd.DataFrame([e["id"] for e in encoded_texts])
        y_preds = []
        actual_ys = []
        actual_ids = []
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(target):
            X_train, X_test = embeddings.iloc[train_index], embeddings.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            id_train, id_test = ids.iloc[train_index], ids.iloc[test_index]
            actual_ids.extend(id_test.values.flatten())
            actual_ys.extend(y_test.values.flatten())
            model.fit(X_train, y_train)
            y_preds.extend(model.predict(X_test))
        return {
            "r2_score": r2_score(actual_ys, y_preds),
            "median": np.median(np.abs(np.array(actual_ys) - y_preds)),
            "mean": np.mean(np.abs(np.array(actual_ys) - y_preds))
        }
    
    @staticmethod
    def finetune_model(output_model_name="finetuned_model", document_sample_count=1000, 
                       file_path="encoded_texts.json", model_name=MODEL_NAME, 
                       per_device_train_batch_size=64, save_steps=1000, use_fp16=True):
        """
        Finetunes a pre-trained language model on a custom dataset.

        Args:
            output_model_name (str): Name for the output model.
            document_sample_count (int): Number of documents to use for finetuning.
            file_path (str): Path to the encoded texts JSON file.
            model_name (str): Name of the pre-trained model to finetune.
            per_device_train_batch_size (int): Batch size per device during training.
            save_steps (int): Number of steps between saving checkpoints.
            use_fp16 (bool): Whether to use 16-bit floating-point precision.

        Returns:
            str: Path to the directory where the trained model is saved.
            str: Path to the directory where the finetuned SentenceTransformer model is saved.
        """
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        training_args = TrainingArguments(
            output_dir=f"output/{output_model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            overwrite_output_dir=True,
            num_train_epochs=10,
            learning_rate=2e-5,
            weight_decay=0.01,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            fp16=use_fp16,
            save_steps=save_steps,
            logging_steps=save_steps,
            evaluation_strategy="steps",
            eval_steps=save_steps,
            load_best_model_at_end=True,
        )
        if document_sample_count:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
            with open(file_path) as f:
                dataset = [json.loads(e) for e in f.read().split("\n")[:-1]]
            train_sentences = [e["text"] for e in dataset]
            np.random.shuffle(train_sentences)
            tokenized_inputs = tokenizer(train_sentences[:document_sample_count], padding=True, truncation=True, return_tensors="pt")
            tokenized_datasets = DatasetDict({
                "train": Dataset.from_dict({
                    "input_ids": tokenized_inputs["input_ids"],
                    "attention_mask": tokenized_inputs["attention_mask"]
                })
            })
            chunk_size = tokenizer.model_max_length
            def group_texts(examples):
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // chunk_size) * chunk_size
                result = {
                    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result
            lm_datasets = tokenized_datasets.map(group_texts, batched=True)
            train_size = int(0.9 * document_sample_count)
            test_size = int(0.1 * document_sample_count)
            split_dataset = lm_datasets["train"].train_test_split(
                train_size=train_size, test_size=test_size, seed=42
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=split_dataset["train"],
                eval_dataset=split_dataset["test"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            trainer.train()
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        word_embedding_model = models.Transformer(training_args.output_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        final_model_path = training_args.output_dir+"_finetuned"
        sentence_model.save(final_model_path)
        print(f"Training done and model saved to: {final_model_path}")
        return training_args.output_dir, final_model_path
    
    @staticmethod
    def manual_encode_from_trained_mlm(text, output_dir):
        """
        Manually encodes a given text using a trained Masked Language Model (MLM) and average pooling.

        Args:
            text (str): The text to encode.
            output_dir (str): Directory of the trained model.

        Returns:
            list: The embedding vector for the input text.
        """
        def average_pool(last_hidden_states: Tensor,
                         attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        tmp_tokenizer = AutoTokenizer.from_pretrained(output_dir)
        tmp_model = AutoModel.from_pretrained(output_dir)
        batch_dict = tmp_tokenizer(
            [text],
            max_length=tmp_tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        batch_dict.to("cuda")
        tmp_model.to("cuda")
        outputs = tmp_model(**batch_dict)
        return average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).tolist()[0]
    
    @staticmethod
    def cosine_similarity_numpy(a, b):
        """
        Calculates the cosine similarity between two vectors.

        Args:
            a (np.array): First vector.
            b (np.array): Second vector.

        Returns:
            np.array: Cosine similarity score.
        """
        a = np.array(a)
        b = np.array(b)
        dot_product = np.dot(a, b.T)
        a_norm = np.linalg.norm(a, axis=-1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=-1, keepdims=True)
        similarity = dot_product / (a_norm * b_norm.T)
        return similarity
    
    @staticmethod
    def consistency_check(raw_output_dir, sentence_transformer_output_dir, input_text):
        """
        Validates that the model assets perform consistently after being saved in different formats.

        Args:
            raw_output_dir (str): Directory of the raw trained model.
            sentence_transformer_output_dir (str): Directory of the finetuned SentenceTransformer model.
            input_text (str): Input text for consistency checking.
        """
        loaded_model = SentenceTransformer(sentence_transformer_output_dir)
        assert Finetuner.cosine_similarity_numpy(
            Finetuner.manual_encode_from_trained_mlm(input_text, raw_output_dir),
            loaded_model.encode(input_text).tolist()
        ) > 0.99
    
    @staticmethod
    def run(finetune_document_sample_count=1000, file_path="encoded_texts.json"):
        """
        Runs the finetuning process and evaluates the model.

        Args:
            finetune_document_sample_count (int): Number of documents for finetuning.
            file_path (str): Path to the encoded texts JSON file.

        Returns:
            dict: A dictionary containing the evaluation metrics (R², median, and mean absolute errors).
        """
        raw_output_dir, sentence_transformer_output_dir = Finetuner.finetune_model(
            document_sample_count=finetune_document_sample_count,
            file_path=file_path
        )
        Finetuner.consistency_check(raw_output_dir, sentence_transformer_output_dir, "hello world")
        return Finetuner.evaluate_model(SentenceTransformer(sentence_transformer_output_dir), file_path=file_path)
    
    @staticmethod
    def full_finetuning_test():
        """
        Conducts a full finetuning test over a range of document sample sizes and stores the results.

        Returns:
            dict: A dictionary mapping sample sizes to their respective evaluation results.
        """
        keyed_results = {}
        for i in np.arange(0, 120000, 10000):
            if not keyed_results.get(str(i)):
                test_result = Finetuner.run(i)
                keyed_results[str(i)] = test_result
            with open("stored_results.json", "w") as f:
                f.write(json.dumps(keyed_results))
        return keyed_results
