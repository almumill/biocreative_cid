import argparse
import math
import numpy as np
import os
from tqdm.auto import tqdm
import pandas as pd
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import sys
import torch
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification

# Todos Sept. 7
# - Address comments left behind on each section of the code
# - Move evaluation stuff out of this script beyond validation loss calculations

def load_split_data(data_file):
  instances_df = pd.read_csv(data_file, sep="\t", header=None, dtype={"sentence": 'string', 'label': 'string'})

  instances_df.columns = ["sentence", "label"]

  label_id_map = dict()
  sorted_labels = sorted(list(instances_df.label.unique()))

  for idx, label in enumerate(sorted_labels):
    label_id_map[label] = idx

  instances_df.label = instances_df.label.apply(lambda x:label_id_map[x])

  return instances_df, label_id_map

# Args:
# model_name                              Name of model being trained
# --batch_size <batch size>               Batch size
# --num_epochs <num epochs>               Number of epochs to run through
# --min_epochs <num epochs>               Minimum number of epochs to run through if early stopping is enabled
# --lr <learning rate>                    Learning rate
# --early_stopping                        Perform early stopping if this flag is provided
# --early_stopping_threshold <thresh>     If the change in training loss is smaller than this threshold, stop early
# --train_file <path to file>             Training data file path
# --validation_file <path to file>        Validation data file path
# --training_only                         Only train the model if this flag is provided
# --validation_only                       Only evaluate a trained model on the validation split
# --path_to_model <path to file>          Load a fine-tuned model to evaluate on the validation split, must be provided when only evaluating
# --weighted_loss                         Use a weighted cross-entropy loss if this flag is provided
# --max_seq_len

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_name", help="File name of model being trained")
  parser.add_argument("model_output_dir", help="Directory to hold trained model and checkpoints if enabled")
  parser.add_argument("--save_checkpoints", help="Save checkpoints at each epoch", action="store_true", default=False)
  parser.add_argument("--batch_size", help="Batch size", type=int, default=32)
  parser.add_argument("--num_epochs", help="Number of epochs to run through",
                      type=int, default=10)
  parser.add_argument("--min_epochs", help="Minimum number of epochs if doing early stopping")
  parser.add_argument("--lr", help="Learning rate", type=float,
                      default=10 ** -5)
  parser.add_argument("--early_stopping", help="Do early stopping if this flag is provided", action="store_true")
  parser.add_argument("--early_stopping_threshold", help="Validation loss change threshold for early stopping", type=float, default=10 ** -3)
  parser.add_argument("train_file", help="Path to training data file")
  parser.add_argument("validation_file", help="Path to validation data file")
  parser.add_argument("--training_only", help="Only train model", action="store_true")
  parser.add_argument("--validation_only", help="Evaluate a particular saved model on the validation data", action="store_true")
  parser.add_argument("--path_to_trained_model", help="Path to saved model to evaluate on validation data (only used when --validation_only is provided")
  parser.add_argument("--weighted_loss", help="Provide this flag to use a weighted cross-entropy loss where weights are the inverse frequencies of each class.", action='store_true')
  parser.add_argument("--max_seq_len", help="Maximum sequence length for relation extraction instances", type=int, default=256)

  args = parser.parse_args()

  model_output_dir = args.model_output_dir
  if not os.path.exists(model_output_dir):
    os.makedirs(model_output_dir)

  model_name = args.model_name
  save_checkpoints = args.save_checkpoints
  batch_size = args.batch_size
  num_epochs = args.num_epochs
  min_epochs = args.min_epochs
  lr = args.lr

  early_stopping = True if args.early_stopping else False
  early_stopping_threshold = args.early_stopping_threshold

  train_file = args.train_file
  validation_file = args.validation_file

  training_only = True if args.training_only else False
  validation_only = True if not args.validation_only else False
  path_to_trained_model = args.path_to_trained_model if args.path_to_trained_model else ""

  weighted_loss = True if args.weighted_loss else False
  max_seq_len = args.max_seq_len

  instances_df, label_id_map = load_split_data(train_file)
  val_instances_df, label_id_map = load_split_data(validation_file)

  id_label_map = {v: k for k, v in label_id_map.items()}
  print(id_label_map)

  args_file = open(os.path.join(model_output_dir, "args.txt"), "w")
  args_file_out = ""

  for param, val in vars(args).items():
    args_file_out += param + ": " + str(val) + "\n"

  # Dump all parameters
  args_file.write(args_file_out)
  args_file.close()

  classes = val_instances_df["label"].unique()

  torch.manual_seed(0)
  np.random.seed(0)

  tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
  model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2", num_labels=2)

  if torch.cuda.is_available():
    model = model.cuda()
    dev = "cuda"
  else:
    dev = "cpu"

  device = torch.device(dev)
  model = model.to(device)
  optimizer = AdamW(model.parameters(), lr=lr)

  # shuffle instances
  instances_df = instances_df.sample(frac=1, random_state=0)

  # compute inverse class frequencies to use as weights
  weights = instances_df.groupby(["label"]).nunique().values.tolist()
  weights = [1/float(n[0]) for n in weights]

  weights /= np.max(weights)
  for idx, weight in enumerate(weights):
    print("Class " + str(idx) + " - " + str(weight))

  weights = torch.tensor(weights)
  val_instances_df = val_instances_df.sample(frac=1, random_state=0)

  instance_indices = np.arange(instances_df.shape[0])
  np.random.shuffle(instance_indices)
  batch_indices = [list(l) for l in np.array_split(instance_indices, math.floor(instances_df.shape[0]/batch_size))]

  batches = []

  for batch in batch_indices:
    sentences = list(instances_df.iloc[batch].sentence.values)
    labels = list(instances_df.iloc[batch].label.values)

    batch_in = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_seq_len)
    batch_in["labels"] = torch.tensor(labels)
    batches += [batch_in]


  progress_bar = tqdm(range(len(batches)))
  f1 = 0
  prev_f1 = 0

  stopped_early = False

  val_losses = []
  f1s = []
  prev_loss = 0
  loss_sum = 0

  confusion_matrices = dict()
  for label in labels:
    confusion_matrices[label] = [0, 0, 0, 0]

  for epoch in range(num_epochs):
    print("Epoch " + str(epoch))
    model = model.train()
    tqdm_obj = tqdm(range(len(batches)))
    prev_loss = loss_sum
    loss_sum = 0
    
    positive_predictions = 0
    accs = []
    if weighted_loss:
      loss = torch.nn.CrossEntropyLoss(weight=weights.float()).to(device)
    else:
      loss = torch.nn.CrossEntropyLoss().to(device)


    for batch_id in tqdm_obj:
      batch = batches[batch_id]
      b_tokens = batch["input_ids"].to(device)
      b_input_mask = batch["attention_mask"].to(device)
      b_labels = batch["labels"].to(device)

      out = model(b_tokens, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      logits = out.logits

      predictions = np.argmax(logits.detach().cpu().numpy(), axis=1)
      acc = 0

      b_loss = loss(out.logits, b_labels).to(device)
              
      for idx in range(len(list(b_labels))):
        if predictions[idx] != 1:
            positive_predictions += 1
        if b_labels[idx] == predictions[idx]:
          acc += 1.0

      accs += [float(float(acc) / len(b_labels))]

      loss_sum += b_loss.item()

      optimizer.zero_grad()
      b_loss.backward()
      optimizer.step()
      tqdm_obj.set_postfix(pospred=positive_predictions, avg_acc=np.mean(accs))

    torch.save(model.state_dict(), os.path.join(model_output_dir, model_name) + "_c" + str(epoch))
    progress_bar.close()
        
    val_batch_indices = np.arange(val_instances_df.shape[0])
    np.random.shuffle(val_batch_indices)
    val_batch_indices = [list(l) for l in np.array_split(val_batch_indices, math.floor(val_instances_df.shape[0]))]

    val_batches = []

    for batch in val_batch_indices:
      sentences = list(val_instances_df.iloc[batch].sentence.values)
      labels = list(val_instances_df.iloc[batch].label.values)

      batch_in = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=max_seq_len)
      batch_in["labels"] = torch.tensor(labels)

      val_batches += [batch_in]

    model.eval()

    predictions, labels = [], []

    tqdm_obj = tqdm(range(len(val_batches)))
    #tqdm_obj = tqdm(range(10))
    val_loss = 0


    for batch_id in tqdm_obj:
        loss = torch.nn.CrossEntropyLoss().to(device)
        batch = val_batches[batch_id]
        #print("here")
        # input_ids', 'token_type_ids', 'attention_mask', 'labels
        b_tokens = batch["input_ids"].to(device)
        b_input_mask = batch["attention_mask"].to(device)
        b_labels = batch["labels"].to(device)
        with torch.no_grad():
          out = model(b_tokens, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels).logits
          loss = loss(out, b_labels)
          val_loss += loss.item()


        #print(out)
        logits = out

        predictions.append(logits.detach().cpu().numpy())
        labels.append(b_labels.to('cpu').numpy())


  # print(predictions)
    prev_f1 = f1
    acc = 0
    all_predictions = np.concatenate(predictions, axis=0)
    all_predictions = np.argmax(all_predictions, axis=1).flatten()
    all_labels = np.concatenate(labels, axis=0)

    for label in classes:
      confusion_matrices[label] = [0, 0, 0, 0]

    for idx in range(len(all_labels)):
      prediction = all_predictions[idx]
      label = all_labels[idx]

      if label == label_id_map["No_relation"] and prediction == label_id_map["No_relation"]:
        continue

      # if label not in confusion_matrices.keys():
      #    confusion_matrics[label] = [0, 0, 0, 0]


      # Sept. 7 Comments for case logic, simplify where possible
      if prediction == label:
        confusion_matrices[label][0] += 1
      else:
        if prediction != label_id_map["No_relation"]:
          confusion_matrices[prediction][1] += 1

        if label != label_id_map["No_relation"]:
          confusion_matrices[label][2] += 1

        for class_id in classes:
          if class_id == label or class_id == prediction or class_id == label_id_map["No_relation"]:
            continue
          else:
            confusion_matrices[class_id][3] += 1

    print("Loss: " + str(loss_sum) + "\nPrev: " + str(prev_loss))
    print("Validation loss: " + str(val_loss))

    agg_confusion_matrix = np.sum([confusion_matrices[mat_id] for mat_id in confusion_matrices.keys()], axis=0)
    precision, recall, f1 = prec_rec_f1(agg_confusion_matrix)

    f1s += [f1]
    val_losses += [val_loss]
    print("F1: " + str(f1))
    print(f1s)

    open(os.path.join(model_output_dir, "train_metrics.txt"), "w").write("\n".join([str(i) + "\t" + str(val_losses[i]) + "\t" + str(f1s[i]) for i in range(len(f1s))]))
    #if f1 == max(f1s):
    #  print("Best so far")
    #  torch.save(model.state_dict(), model_path)

  #if not stopped_early:
  #open(os.path.join(MODEL_NAME, "train_metrics.txt"), "w").write("\n".join([str(i) + "\t" + str(val_losses[i]) + "\t" + str(f1s[i]) for i in range(len(f1s))]))
  #  torch.save(model.state_dict(), model_path)



  open(os.path.join(model_output_dir, "f1s.txt"), "w").write("\n".join([str(f1) for f1 in f1s]))

def prec_rec_f1(confusion_matrix):
  # confusion_matrices[label] = [TP, FP, FN, TN]
  TP = confusion_matrix[0]
  FP = confusion_matrix[1]
  FN = confusion_matrix[2]
  TN = confusion_matrix[3]

  if TP == 0:
    return 0, 0, 0

  precision = float(TP) / (TP + FP)
  recall = float(TP) / (TP + FN)
  f1 = float(2 * TP) / ((2*TP) + FP + FN)

  return precision, recall, f1

if __name__ == "__main__":
  main()
