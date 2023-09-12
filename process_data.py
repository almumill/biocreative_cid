import argparse
from nltk import tokenize
import pandas as pd
import numpy as np
import os
import sys

CHEMICAL_TOKEN = "$CHEMICAL$"
DISEASE_TOKEN = "$DISEASE$"

TYPE_TOKEN_MAP = dict()
TYPE_TOKEN_MAP["Chemical"] = CHEMICAL_TOKEN
TYPE_TOKEN_MAP["Disease"] = DISEASE_TOKEN

"""
227508|t|Naloxone reverses the antihypertensive effect of clonidine.
227508|a|In unanesthetized, spontaneously hypertensive rats the decrease in blood pressure and heart rate produced by intravenous clonidine, 5 to 20 micrograms/kg, was inhibited or reversed by nalozone, 0.2 to 2 mg/kg. The hypotensive effect of 100 mg/kg alpha-methyldopa was also partially reversed by naloxone. Naloxone alone did not affect either blood pressure or heart rate. In brain membranes from spontaneously hypertensive rats clonidine, 10(-8) to 10(-5) M, did not influence stereoselective binding of [3H]-naloxone (8 nM), and naloxone, 10(-8) to 10(-4) M, did not influence clonidine-suppressible binding of [3H]-dihydroergocryptine (1 nM). These findings indicate that in spontaneously hypertensive rats the effects of central alpha-adrenoceptor stimulation involve activation of opiate receptors. As naloxone and clonidine do not appear to interact with the same receptor site, the observed functional antagonism suggests the release of an endogenous opiate by clonidine or alpha-methyldopa and the possible role of the opiate in the central control of sympathetic tone.
227508	0	8	Naloxone	Chemical	D009270
227508	49	58	clonidine	Chemical	D003000
227508	93	105	hypertensive	Disease	D006973
227508	181	190	clonidine	Chemical	D003000
227508	244	252	nalozone	Chemical	-1
227508	274	285	hypotensive	Disease	D007022
227508	306	322	alpha-methyldopa	Chemical	D008750
227508	354	362	naloxone	Chemical	D009270
227508	364	372	Naloxone	Chemical	D009270
227508	469	481	hypertensive	Disease	D006973
227508	487	496	clonidine	Chemical	D003000
227508	563	576	[3H]-naloxone	Chemical	-1
227508	589	597	naloxone	Chemical	D009270
227508	637	646	clonidine	Chemical	D003000
227508	671	695	[3H]-dihydroergocryptine	Chemical	-1
227508	750	762	hypertensive	Disease	D006973
227508	865	873	naloxone	Chemical	D009270
227508	878	887	clonidine	Chemical	D003000
227508	1026	1035	clonidine	Chemical	D003000
227508	1039	1055	alpha-methyldopa	Chemical	D008750
227508	CID	D008750	D007022
"""

# get abstracts is confusing, we're getting individual article info (?)
def get_documents(in_filename):
  file_contents = open(in_filename, "r").read()
  abstracts = file_contents.split("\n\n")[:-1]

  return abstracts
def process_abstract(abstract_text):
  # Entity line tab-delim-split format: ["Abs_ID", "Start_Pos", "End_Pos", "Entity", "Entity_Type"]
  lines = abstract_text.split("\n")
  title_line = lines[0]
  abs_text = lines[1]

  full_text = title_line.split("|")[2].strip() + " " + abs_text.split("|")[2].strip()
  entity_lines = lines[2:]

  chems = []
  diseases = []
  relations = dict()

  entity = []
  entity_1 = []
  entity_2 = []

  for line in entity_lines:
    tokens = [t.strip() for t in line.split("\t") if t != '']
    entity_count = 1

    if len(tokens) == 6:
      abs_id, start_pos, end_pos, name, type, id = tokens
      entity = [[int(start_pos), int(end_pos), name, type, id]]
    elif len(tokens) == 4:
      abs_id, rel_type, chem_id, disease_id = tokens
      relations[(chem_id, disease_id)] = 1
    else:
      abs_id, start_pos, end_pos, name, type, id = tokens[:6]
      ids = id.split("|")
      entity_count = 2
      entity_1 = [[int(start_pos), int(end_pos), name, type, ids[0]]]
      entity_2 = [[int(start_pos), int(end_pos), name, type, ids[1]]]

    if len(tokens) != 4:
      if entity_count == 1:
        if type == "Chemical":
          chems += [entity]
        else:
          diseases += [entity]
      else:
        if type == "Chemical":
          chems += [entity_1]
          chems += [entity_2]
        else:
          diseases += [entity_1]
          diseases += [entity_2]

  out_lines = ""

  seen_pairs = []

  for chem_mention in chems:
    for disease_mention in diseases:
      first_mentioned_entity, second_mentioned_entity = sorted([chem_mention[0], disease_mention[0]], key=lambda x: x[0])
      # Overlapping mention edge case:
      if first_mentioned_entity[1] > second_mentioned_entity[0]:
        continue

      # Create copy of abstract with token substitutions

      article_copy = full_text[:first_mentioned_entity[0]] + TYPE_TOKEN_MAP[first_mentioned_entity[3]] + \
                   full_text[first_mentioned_entity[1]:second_mentioned_entity[0]] + TYPE_TOKEN_MAP[
                     second_mentioned_entity[3]] + \
                   full_text[second_mentioned_entity[1]:]


      sentences = tokenize.sent_tokenize(article_copy)

      for sentence in sentences:
        if CHEMICAL_TOKEN in sentence and DISEASE_TOKEN in sentence:
          pair = (chem_mention[0][4], disease_mention[0][4])

          if pair in relations.keys() and pair not in seen_pairs:
            out_lines += sentence + "\tCID\n"
            seen_pairs += [pair]
          else:
            out_lines += sentence + "\tNo_relation\n"

  return out_lines

def create_split_data(in_filename, status_every_n):
  abstracts = get_documents(in_filename)

  out_lines = ""

  for idx, abstract in enumerate(abstracts):
    if idx % status_every_n == 0:
      print(str(idx) + " out of " + str(len(abstracts)))

    out_lines += process_abstract(abstract)

  return out_lines

def main():
  # data_dir = sys.argv[1]
  parser = argparse.ArgumentParser()
  parser.add_argument("raw_data_dir", help="Path to CDR raw data directory")
  parser.add_argument("output_data_dir_name", help="Output directory name for processed data (CID task)")
  parser.add_argument("--status_every_n", type=int, help="Print number of abstracts processed every n")


  # parser.add_argument("--help", "python3 process_data.py <raw data directory> <processed CID task data directory>")
  # parser.add_argument("--optionaltest")
  
  args = parser.parse_args()
  raw_data_dir = args.raw_data_dir
  output_data_dir_name = args.output_data_dir_name

  status_every_n = 50

  if args.status_every_n:
    status_every_n = args.status_every_n

  if not os.path.exists(raw_data_dir):
    print("Raw data path does not exist, see help with python3 process_data.py --help")

  if not os.path.exists(output_data_dir_name):
    os.makedirs(output_data_dir_name)

  train_file = os.path.join(raw_data_dir, "CDR_TrainingSet.PubTator.txt")
  dev_file = os.path.join(raw_data_dir, "CDR_DevelopmentSet.PubTator.txt")
  test_file = os.path.join(raw_data_dir, "CDR_TestSet.PubTator.txt")

  print("Processing training split")
  out_lines = create_split_data(train_file, status_every_n)
  open(os.path.join(output_data_dir_name, "cdr_train_data.tsv"), "w").write(out_lines)

  print("Processing dev split")
  out_lines = create_split_data(dev_file, status_every_n)
  open(os.path.join(output_data_dir_name, "cdr_dev_data.tsv"), "w").write(out_lines)

  print("Processing test split")
  out_lines = create_split_data(test_file, status_every_n)
  open(os.path.join(output_data_dir_name, "cdr_test_data.tsv"), "w").write(out_lines)

if __name__ == "__main__":
  main()
