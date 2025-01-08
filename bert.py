import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset, ClassLabel
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import os

def main(index):
    # Get XLA device
    device = xm.xla_device()
    print(f"Using device: {device}")

    # Load the dataset
    dataset = load_dataset("conll2012_ontonotesv5", "english_v4")

    NER_LABELS = [
        'O',           # 0
        'B-PERSON',    # 1
        'I-PERSON',    # 2
        'B-NORP',      # 3
        'I-NORP',      # 4
        'B-FAC',       # 5
        'I-FAC',       # 6
        'B-ORG',       # 7
        'I-ORG',       # 8
        'B-GPE',       # 9
        'I-GPE',       # 10
        'B-LOC',       # 11
        'I-LOC',       # 12
        'B-PRODUCT',   # 13
        'I-PRODUCT',   # 14
        'B-DATE',      # 15
        'I-DATE',      # 16
        'B-TIME',      # 17
        'I-TIME',      # 18
        'B-PERCENT',   # 19
        'I-PERCENT',   # 20
        'B-MONEY',     # 21
        'I-MONEY',     # 22
        'B-QUANTITY',  # 23
        'I-QUANTITY',  # 24
        'B-ORDINAL',   # 25
        'I-ORDINAL',   # 26
        'B-CARDINAL',  # 27
        'I-CARDINAL',  # 28
        'B-EVENT',     # 29
        'I-EVENT',     # 30
        'B-WORK_OF_ART', # 31
        'I-WORK_OF_ART', # 32
        'B-LAW',       # 33
        'I-LAW',       # 34
        'B-LANGUAGE',  # 35
        'I-LANGUAGE'   # 36
    ]

    id2label = {i: label for i, label in enumerate(NER_LABELS)}
    label2id = {label: i for i, label in enumerate(NER_LABELS)}
    max_length = 512

    # Load ModernBERT model and tokenizer
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if xm.is_master_ordinal():
        print("\nExample with meaningful labels:")
        print("Original: [0, 0, 0, 0, 0] -> ", [id2label[0], id2label[0], id2label[0], id2label[0], id2label[0]])
        print("Original: [7, 8] -> ", [id2label[7], id2label[8]])
        print("Original: [31, 32] -> ", [id2label[31], id2label[32]])

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], 
            truncation=True, 
            is_split_into_words=True,
            max_length=max_length,
            padding="max_length"
        )
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def preprocess_dataset(dataset):
        flattened_data = {
            "tokens": [],
            "ner_tags": []
        }
        for document in dataset:
            for sentence in document['sentences']:
                flattened_data["tokens"].append(sentence['words'])
                flattened_data["ner_tags"].append(sentence['named_entities'])
        
        flattened_dataset = Dataset.from_dict(flattened_data)
        
        if xm.is_master_ordinal():
            print("\nFirst few examples before tokenization:")
            print("----------------------------------------")
            for i in range(min(3, len(flattened_dataset))):
                print(f"\nExample {i+1}:")
                print(f"Tokens: {flattened_dataset[i]['tokens']}")
                print(f"NER tags: {flattened_dataset[i]['ner_tags']}")
                print(f"NER labels: {[id2label[tag] for tag in flattened_dataset[i]['ner_tags']]}")
        
        tokenized_dataset = flattened_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=flattened_dataset.column_names
        )
        
        if xm.is_master_ordinal():
            print("\nFirst example after tokenization:")
            print("----------------------------------")
            print(f"Input IDs: {tokenized_dataset[0]['input_ids']}")
            print(f"Labels: {tokenized_dataset[0]['labels']}")
        
        return tokenized_dataset

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
        return results

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=len(NER_LABELS),
        id2label=id2label,
        label2id=label2id,
        use_flash_attention_2=False
    )

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=False
    )

    # Move model to XLA device
    model = model.to(device)

    # Preprocess datasets
    train_dataset = preprocess_dataset(dataset["train"])
    eval_dataset = preprocess_dataset(dataset["validation"])
    test_dataset = preprocess_dataset(dataset["test"])

    training_args = TrainingArguments(
        output_dir="modernbert-base-conll2012_ontonotesv5-english_v4-ner",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=5e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_checkpointing=True,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        load_best_model_at_end=True,
        bf16=True,
        gradient_accumulation_steps=1,
        optim="adamw_hf",
        warmup_ratio=0.1,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        lr_scheduler_type="linear",
        metric_for_best_model="f1",
        no_cuda=True,
        use_xla=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    if xm.is_master_ordinal():
        print("Starting training...")
    
    trainer.train()

    if xm.is_master_ordinal():
        print("Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        print(f"Test results: {test_results}")
        
        print("Saving the model...")
        trainer.save_model("./final_model4ep")

        # Function to make predictions on new sentences
        def predict_ner(sentence):
            inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
            predictions = xm.mesh_reduce('predictions', predictions, lambda x: x[0])
            predicted_labels = [id2label[p.item()] for p in predictions[0] if p.item() != -100]
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            word_ids = inputs.word_ids()
            
            result = []
            for token, label, word_id in zip(tokens, predicted_labels, word_ids):
                if word_id is not None:
                    if token.startswith("##"):
                        result[-1][0] += token[2:]
                    else:
                        result.append([token, label])
            
            return result

        # Example usage
        sample_sentence = "Apple is looking at buying U.K. startup for \$1 billion"
        results = predict_ner(sample_sentence)
        print(f"NER results for '{sample_sentence}':")
        print(results)

if __name__ == "__main__":
    n_cores = int(os.environ.get('NEURON_RT_NUM_CORES', 32))
    xmp.spawn(main, args=(), nprocs=n_cores)
