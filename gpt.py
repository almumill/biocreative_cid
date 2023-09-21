import argparse
import math
import numpy as np
import os
from tqdm.auto import tqdm
import pandas as pd
os.environ["HF_ENDPOINT"] = "https://huggingface.co"
import torch
from transformers import AutoTokenizer, BioGptForCausalLM, AdamW, set_seed

# Arg: batch size
BATCH_SIZE = 8

# Arg: num epcohs
NUM_EPOCHS = 5

# Arg: learning rate
LEARNING_RATE = 3 * (10 ** -5)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def load_split_data(data_file):
  instances_df = pd.read_csv(data_file, sep="\t", header=None)
  instances_df.columns = ["sentence", "label"]

  label_id_map = dict()
  sorted_labels = sorted(list(instances_df.label.unique()))

  for idx, label in enumerate(sorted_labels):
    label_id_map[label] = idx

  id_label_map = {v: k for (k, v) in label_id_map.items()}

  examples_df = pd.DataFrame(columns=["label", "example"])
  instances_df.label = instances_df.label.apply(lambda x:label_id_map[x])
  instances_df["example"] = instances_df.apply(lambda row:id_label_map[row["label"]].replace("-", " ").replace("_", ", ").capitalize() + " - " + row["sentence"], axis=1)


  return instances_df, label_id_map

def main():
# Arg: batch size
# Arg: num epcohs
# Arg: learning rate
  # Arg: path to train data
  # Arg: Class ID to finetune on
  # Arg: output model name

  parser = argparse.ArgumentParser()
  parser.add_argument("batch_size", type=int, help="Batch size")
  parser.add_argument("num_epochs", type=int,help="Number of epochs to finetune")
  parser.add_argument("learning_rate", type=float, help="Learning rate for finetuning")
  parser.add_argument("path_to_train_data", help="Path to train data")
  parser.add_argument("target_class", help="Name of class to finetune the model on")
  parser.add_argument("output_model_name", help="File name for output models")
  parser.add_argument("output_model_dir", help="Directory to store finetuned model (and optionally checkpoints.)")
  parser.add_argument("--output_checkpoints" help="If flag is provided, all model checkpoints are written to disk", default=False, action='store_true'))

  args = parser.parse_args()

  batch_size = args.batch_size
  num_epochs = args.num_epochs
  learning_rate = args.learning_rate
  path_to_train_data = args.path_to_train_data
  target_class = args.target_clas
  output_model_name = args.output_model_name
  output_model_dir = args.output_model_dir
  output_checkpoints = args.output_checkpoints

  # Arg: path to train data
  instances_df, label_id_map = load_split_data("path_to_train_data")
  id_label_map = {v: k for k, v in label_id_map.items()}
 
  tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
  model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

  model = model.cuda()

  if torch.cuda.is_available(): 
    dev = "cuda" 
  else: 
    dev = "cpu" 

  device = torch.device(dev) 
  model = model.to(device)
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

  # Arg: Class ID to finetune on
  target_class = 2
  target_class_name = id_label_map[target_class].replace("-", " ").replace("_", ", ").capitalize()

  num_examples = instances_df[instances_df["label"] == target_class].shape[0]

  example_list = [str(s) for s in instances_df[instances_df["label"] == target_class].sentence.values]

  np.random.seed(0)

  batch_indices = np.arange(len(example_list))
  np.random.shuffle(batch_indices)
  batch_indices = [list(l) for l in np.array_split(batch_indices, math.floor(len(example_list)/BATCH_SIZE))]

  batches = []

  for batch in batch_indices:
    # Note: what's going on here?
    sentences = [example_list[i].replace("$CHEMICAL$", "$ENTITY_A$").replace("$PROTEIN$", "$ENTITY_B$") for i in batch]
    batch_in = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", max_length=128)
    batches += [batch_in]
  progress_bar = tqdm(range(len(batches)))

  # Arg: output model name
  model_name = "gpt2_noprompts_" + str(NUM_EPOCHS) + "_LR_" + '{:e}'.format(LEARNING_RATE) + "_128.bin"

  if os.path.exists(model_name):
      model.load_state_dict(torch.load(model_name))
  else:
      print("Training model " + model_name)
      model = model.train()

      for epoch in range(NUM_EPOCHS):
        print("Epoch " + str(epoch))
        tqdm_obj = tqdm(range(len(batches)))
        loss_sum = 0

        accs = []
        positive_predictions = 0
        total_loss = 0

        for batch_id in tqdm_obj:
          batch = batches[batch_id].to(device)
          out = model(**batch, labels=batch['input_ids'])

          loss = out.loss

          total_loss += loss.item()

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        progress_bar.close()
        print("Loss: " + str(total_loss))
      torch.save(model.state_dict(), model_name)

if __name__ == "__main__":
  main()
