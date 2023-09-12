import math
import numpy as np
import os
from tqdm.auto import tqdm
import pandas as pd
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
import torch
#from sklearn.metrics import f1
import sys
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification


# TODO@Alex: Save predictions to disk, save a tsv of results

BATCH_SIZE = 1

def load_split_data(data_file):
  instances_df = pd.read_csv(data_file, sep="\t", header=None)
  instances_df.columns = ["sentence", "label"]

  label_id_map = dict()
  sorted_labels = sorted(list(instances_df.label.unique()))

  for idx, label in enumerate(sorted_labels):
    label_id_map[label] = idx

  instances_df.label = instances_df.label.apply(lambda x:label_id_map[x])

  return instances_df, label_id_map

def prec_rec_f1(confusion_matrix):
  # confusion_matrices[label] = [TP, FP, FN, TN]
  TP = confusion_matrix[0]
  FP = confusion_matrix[1]
  FN = confusion_matrix[2]
  TN = confusion_matrix[3]

  if TP == 0:
    return 0, 0, 0


  acc = (TP + TN) / (TP + TN + FP + FN)
  precision = float(TP) / (TP + FP)
  recall = float(TP) / (TP + FN)
  f1 = float(2 * TP) / ((2*TP) + FP + FN)

  return precision, recall, f1, acc

def main():
  if sys.argv[3] == "test":
    instances_df, label_id_map = load_split_data("cdr_data/cdr_test_data.tsv")
  else:
    instances_df, label_id_map = load_split_data("cdr_data/cdr_dev_data.tsv")

  id_label_map = {v: k for k, v in label_id_map.items()}
  print(id_label_map.items())
  print(len(id_label_map.keys()))
  #print(id_label_map)
  #exit(1)
  # confusion_matrices[label] = [TP, FP, FN, TN]
  confusion_matrices = dict()

  classes = instances_df["label"].unique()

  for label in classes:
      confusion_matrices[label] = [0, 0, 0, 0]


  all_predictions = []
  all_labels = []
  pred_file_name = sys.argv[1]

  from_disk = False

  if os.path.exists(pred_file_name):
      pred_file_lines = open(pred_file_name, "r").readlines()
      for line in pred_file_lines:
          label, prediction = line.strip().split("\t")

          all_predictions += [int(prediction)]
          all_labels += [int(label)]
          from_disk = True
  else:
      #print(instances_df)
      tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
      model = AutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", num_labels=2)
      #model.load_state_dict(torch.load("biobert_finetuned_4.bin"))
      #model.load_state_dict(torch.load("trained_model_augmented.bin"))
      model.load_state_dict(torch.load(sys.argv[2]))
      #odel = torch.load("trained_model.bin", map_location='cuda:0')['model']
      model.cuda()
      if torch.cuda.is_available(): 
        dev = "cuda" 
      else: 
        dev = "cpu" 
      # torch.cuda.empty_cache()

      device = torch.device(dev) 
      #print(device)
      #exit(1)
      model = model.to(device)
      optimizer = AdamW(model.parameters())
      # optimizer = optimizer.to(device)

      instances_df = instances_df.sample(frac=1)

      np.random.seed(200)
      batch_indices = np.arange(instances_df.shape[0])
      np.random.shuffle(batch_indices)
      batch_indices = [list(l) for l in np.array_split(batch_indices, math.floor(instances_df.shape[0]/BATCH_SIZE))]

      batches = []

      for batch in batch_indices:
        sentences = list(instances_df.iloc[batch].sentence.values)
        labels = list(instances_df.iloc[batch].label.values)

        batch_in = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=256)
        batch_in["labels"] = torch.tensor(labels)

        #for key in batch_in.keys():
        #  batch_in[key] = batch_in[key].to(device)

        batches += [batch_in]

      # progress_bar = tqdm(range(len(batches)))
      #torch.save(model.state_dict(), "base_model.bin")
      model.eval()

      predictions, labels = [], []

      tqdm_obj = tqdm(range(len(batches)))
      #tqdm_obj = tqdm(range(10))

      for batch_id in tqdm_obj:
          batch = batches[batch_id]
          #print("here")
          # input_ids', 'token_type_ids', 'attention_mask', 'labels
          b_tokens = batch["input_ids"].to(device)
          b_input_mask = batch["attention_mask"].to(device)
          b_labels = batch["labels"].to(device)
          with torch.no_grad():
            out = model(b_tokens, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels).logits

          #print(out)
          logits = out

          predictions.append(logits.detach().cpu().numpy())
          labels.append(b_labels.to('cpu').numpy())


  # print(predictions)

      acc = 0
      all_predictions = np.concatenate(predictions, axis=0)
      all_predictions = np.argmax(all_predictions, axis=1).flatten()
      all_labels = np.concatenate(labels, axis=0)

  #print("Pred\tTrue")
  #for idx in range(len(all_labels)):
  #    if all_labels[idx] == label_id_map["No_relation"] and all_predictions[idx] == label_id_map["No_relation"]:
  #      continue
  #    print(str(all_predictions[idx]) + "\t" + str(all_labels[idx]))

  # confusion_matrices[label] = [TP, FP, FN, TN]
  confusion_matrices = dict()

  for label in classes:
      confusion_matrices[label] = [0, 0, 0, 0]

  pred_file_out = ""
  for idx in range(len(all_labels)):
      prediction = all_predictions[idx]
      label = all_labels[idx]

      pred_file_out += str(label) + "\t" + str(prediction) + "\n"

      if label == label_id_map["No_relation"] and prediction == label_id_map["No_relation"]:
          confusion_matrices[label][3] += 1
          continue

      #if label not in confusion_matrices.keys():
      #    confusion_matrics[label] = [0, 0, 0, 0]

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

  metrics_df = pd.DataFrame(columns=["Class_ID", "Precision", "Recall", "F1"])
  if not from_disk:
      open(pred_file_name, "w").write(pred_file_out)

  res_out = "class_id\tf1\tprecision\trecall\n"
  for class_id in sorted(classes):
      confusion_matrix = confusion_matrices[class_id]
      print("--")
      print(confusion_matrix)

      if class_id == label_id_map["No_relation"]:
         print(class_id)
         continue

      precision, recall, f1, acc = prec_rec_f1(confusion_matrix)
      metrics_df = pd.concat([metrics_df, pd.DataFrame({"Class_ID": class_id, "Precision": precision, "Recall": recall, "F!": f1}, index=[0])], ignore_index=True)
      print(str(class_id) + "\tF1: " + str(f1) + "\tPrecision: " + str(precision) + "\tRecall: " + str(recall))
      res_out += str(class_id) + "\tF1: " + str(f1) + "\tPrecision: " + str(precision) + "\tRecall: " + str(recall) + "\n"

  #print(metrics_df)  
  agg_confusion_matrix = np.sum([confusion_matrices[mat_id] for mat_id in confusion_matrices.keys()], axis=0)
  print(agg_confusion_matrix)


  precision, recall, f1, acc = prec_rec_f1(agg_confusion_matrix)
  print("Micro F1: " + str(f1))
  res_out += "Total\tF1: " + str(f1) + "\tPrecision: " + str(precision) + "\tRecall: " + str(recall) + "\tAccuracy: " + str(acc) + "\n"
  

  open(sys.argv[1] + "_res", "w").write(res_out)

if __name__ == "__main__":
  main()
