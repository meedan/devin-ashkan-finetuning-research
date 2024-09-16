import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import xgboost as xgb
from datasets import Dataset, DatasetDict

from torch import Tensor
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling,
    T5EncoderModel,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

from sentence_transformers import SentenceTransformer, models

MODEL_NAME = 'thenlper/gte-base'
# Can also use other popular models like 
# google-bert/bert-base-uncased and sentence-transformers/all-MiniLM-L6-v2

class EvaluationCallback(TrainerCallback):
    """
    Custom callback to evaluate the model at each checkpoint.
    """
    def __init__(self, finetuner_instance, model_name, output_dir, file_path, results_dict):
        self.finetuner_instance = finetuner_instance
        self.model_name = model_name
        self.output_dir = output_dir
        self.file_path = file_path
        self.results_dict = results_dict
        self.checkpoint_counter = 0

    def on_save(self, args, state, control, **kwargs):
        # Increment checkpoint counter
        self.checkpoint_counter += 1
        checkpoint_dir = os.path.join(self.output_dir, f"{self.model_name}_checkpoint_{self.checkpoint_counter}")

        # Save the current model and tokenizer
        kwargs['model'].save_pretrained(checkpoint_dir)
        kwargs['tokenizer'].save_pretrained(checkpoint_dir)

        # Convert to SentenceTransformer model
        word_embedding_model = models.Transformer(checkpoint_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        finetuned_model_path = checkpoint_dir + "_finetuned"
        sentence_model.save(finetuned_model_path)

        # Evaluate the model
        evaluation_result = self.finetuner_instance.evaluate_model(
            SentenceTransformer(finetuned_model_path), file_path=self.file_path
        )

        # Store the results
        self.results_dict[f"{self.model_name}_checkpoint_{self.checkpoint_counter}"] = evaluation_result

        # Save intermediate results
        with open(os.path.join(self.output_dir, "intermediate_results.json"), "w") as f:
            json.dump(self.results_dict, f, indent=4)

        print(f"Checkpoint {self.checkpoint_counter} evaluation completed.")

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
        dataset = [json.loads(e) for e in open(file_path).read().split("\n") if e]
        np.random.shuffle(dataset)
        dataset = dataset[:20000]
        embeddings = transformer.encode([e["text"] for e in dataset], show_progress_bar=False)
        encoded_texts = []
        for i, row in enumerate(dataset):
            encoded_texts.append({"id": row["id"]["$oid"], "price": row["price"], "embedding": embeddings[i]})
        np.random.shuffle(encoded_texts)
        model = xgb.XGBRegressor(
            n_estimators=500,
            device='cuda',
            n_jobs=4
        )
        embeddings_df = pd.DataFrame([e["embedding"] for e in encoded_texts])
        target = pd.DataFrame([e["price"] for e in encoded_texts])
        ids = pd.DataFrame([e["id"] for e in encoded_texts])
        y_preds = []
        actual_ys = []
        actual_ids = []
        kf = KFold(n_splits=5)
        for train_index, test_index in kf.split(target):
            X_train, X_test = embeddings_df.iloc[train_index], embeddings_df.iloc[test_index]
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
                       per_device_train_batch_size=8, use_fp16=False, finetuner_instance=None):
        """
        Finetunes a pre-trained language model on a custom dataset.
        """
        config = AutoConfig.from_pretrained(model_name)
        if config.model_type == 't5':
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            is_t5 = True
        else:
            model = AutoModelForMaskedLM.from_pretrained(model_name)
            is_t5 = False
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        output_dir = f"output/{output_model_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Load and preprocess the dataset
        with open(file_path) as f:
            dataset = [json.loads(e) for e in f.read().split("\n") if e]
        train_sentences = [e["text"] for e in dataset]
        np.random.shuffle(train_sentences)
        train_sentences = train_sentences[:document_sample_count]

        if is_t5:
            # Prepare dataset for sequence-to-sequence training
            def preprocess_function(examples):
                inputs = examples['text']
                model_inputs = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True, padding='max_length')
                # Use the same text as labels
                labels = tokenizer(inputs, max_length=tokenizer.model_max_length, truncation=True, padding='max_length')
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            tokenized_datasets = DatasetDict({
                "train": Dataset.from_dict({"text": train_sentences})
            })
            lm_datasets = tokenized_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=["text"],
            )
        else:
            # Original MLM tokenization and grouping
            tokenized_inputs = tokenizer(train_sentences, padding='max_length', truncation=True, max_length=tokenizer.model_max_length)
            tokenized_datasets = DatasetDict({
                "train": Dataset.from_dict(tokenized_inputs)
            })
            # Group texts into chunks
            def group_texts(examples):
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
                result = {
                    k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            lm_datasets = tokenized_datasets.map(group_texts, batched=True)

        # Split dataset
        train_size = int(0.9 * len(lm_datasets["train"]))
        test_size = len(lm_datasets["train"]) - train_size
        split_dataset = lm_datasets["train"].train_test_split(
            train_size=train_size, test_size=test_size, seed=42
        )

        # Calculate total steps and save_steps to have 10 checkpoints
        total_steps = (train_size * 10) // (per_device_train_batch_size)  # Assuming num_train_epochs=10
        save_steps = max(1, total_steps // 10)

        # Update training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            gradient_checkpointing=False,
            gradient_accumulation_steps=4,  # Adjust as needed
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
            save_total_limit=10,  # Limit to 10 checkpoints
        )

        # Prepare data collator
        if is_t5:
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

        # Prepare results dictionary
        results_dict = {}

        # Initialize the trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                EvaluationCallback(
                    finetuner_instance=finetuner_instance,
                    model_name=model_name,
                    output_dir=output_dir,
                    file_path=file_path,
                    results_dict=results_dict
                )
            ]
        )

        # Start training
        trainer.train()

        # Save final model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Convert to SentenceTransformer model
        if is_t5:
            word_embedding_model = models.Transformer(output_dir)
        else:
            word_embedding_model = models.Transformer(output_dir)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        final_model_path = output_dir + "_finetuned"
        sentence_model.save(final_model_path)

        # Save final evaluation result
        final_evaluation = finetuner_instance.evaluate_model(
            SentenceTransformer(final_model_path), file_path=file_path
        )
        results_dict[f"{model_name}_final"] = final_evaluation

        # Save all results
        with open(os.path.join(output_dir, "final_results.json"), "w") as f:
            json.dump(results_dict, f, indent=4)

        print(f"Training done and model saved to: {final_model_path}")
        return output_dir, final_model_path

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
        batch_dict = {k: v.to(tmp_model.device) for k, v in batch_dict.items()}
        tmp_model.to(tmp_model.device)
        with torch.no_grad():
            outputs = tmp_model(**batch_dict)
        return average_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().tolist()[0]

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
        similarity = Finetuner.cosine_similarity_numpy(
            Finetuner.manual_encode_from_trained_mlm(input_text, raw_output_dir),
            loaded_model.encode(input_text).tolist()
        )
        assert similarity > 0.99, f"Consistency check failed: similarity={similarity}"

    @staticmethod
    def run(finetune_document_sample_count=1000, file_path="encoded_texts.json", model_name=MODEL_NAME):
        """
        Runs the finetuning process and evaluates the model.

        Args:
            finetune_document_sample_count (int): Number of documents for finetuning.
            file_path (str): Path to the encoded texts JSON file.

        Returns:
            dict: A dictionary containing the evaluation metrics (R², median, and mean absolute errors).
        """
        finetuner_instance = Finetuner()
        raw_output_dir, sentence_transformer_output_dir = finetuner_instance.finetune_model(
            document_sample_count=finetune_document_sample_count,
            file_path=file_path,
            model_name=model_name,
            finetuner_instance=finetuner_instance,
            output_model_name=f"{model_name.replace('/', '_')}_{finetune_document_sample_count}"
        )
        finetuner_instance.consistency_check(raw_output_dir, sentence_transformer_output_dir, "hello world")
        return finetuner_instance.evaluate_model(SentenceTransformer(sentence_transformer_output_dir), file_path=file_path)

    @staticmethod
    def full_finetuning_test(model_name=MODEL_NAME):
        """
        Conducts a full finetuning test over a range of document sample sizes and stores the results.

        Returns:
            dict: A dictionary mapping sample sizes to their respective evaluation results.
        """
        finetuner_instance = Finetuner()
        keyed_results = {}
        for i in range(0, 120001, 10000):
            if not keyed_results.get(str(i)):
                test_result = finetuner_instance.run(i, model_name=model_name)
                keyed_results[str(i)] = test_result
                with open(f"stored_results_{model_name.replace('/', '_')}.json", "w") as f:
                    f.write(json.dumps(keyed_results, indent=4))
        return keyed_results

    @staticmethod
    def run_full_analysis(file_path="encoded_texts.json"):
        """
        Runs full analysis on multiple models and document sample sizes.

        Args:
            file_path (str): Path to the encoded texts JSON file.

        Returns:
            dict: A dictionary containing results for all models and sample sizes.
        """
        model_names = ['sentence-transformers/gtr-t5-base', 'sentence-transformers/gtr-t5-large', 'sentence-transformers/gtr-t5-xl']
        overall_results = {}

        for model_name in model_names:
            print(f"Starting analysis for model: {model_name}")
            finetuner_instance = Finetuner()
            model_results = {}
            for doc_count in range(10000, 120001, 10000):
                print(f"Training with {doc_count} documents for model {model_name}")
                raw_output_dir, sentence_transformer_output_dir = finetuner_instance.finetune_model(
                    document_sample_count=doc_count,
                    file_path=file_path,
                    model_name=model_name,
                    finetuner_instance=finetuner_instance,
                    output_model_name=f"{model_name.replace('/', '_')}_{doc_count}"
                )
                finetuner_instance.consistency_check(raw_output_dir, sentence_transformer_output_dir, "hello world")
                evaluation_result = finetuner_instance.evaluate_model(
                    SentenceTransformer(sentence_transformer_output_dir), file_path=file_path
                )
                model_results[str(doc_count)] = evaluation_result
                # Save intermediate results
                with open(f"analysis_results_{model_name.replace('/', '_')}.json", "w") as f:
                    json.dump(model_results, f, indent=4)
            overall_results[model_name] = model_results
            # Save results for each model
            with open(f"analysis_results_{model_name.replace('/', '_')}.json", "w") as f:
                json.dump(model_results, f, indent=4)
        # Save overall results
        with open("overall_analysis_results.json", "w") as f:
            json.dump(overall_results, f, indent=4)
        return overall_results
