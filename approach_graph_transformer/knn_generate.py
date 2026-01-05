import os
import re
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq, 
    DataCollatorWithPadding
)

from peft import LoraConfig, get_peft_model

from trl import (
    GRPOTrainer,
    GRPOConfig,
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)

import sacrebleu

import sys
# To ensure the custom package is found
path_to_repo = "/".join(os.path.abspath(__file__).split("/")[:-2])
if path_to_repo not in sys.path:
    sys.path.append(path_to_repo)

from baseline.data_utils import (
    load_id2emb,
    load_descriptions_from_graphs,
    PreprocessedGraphDataset,
    collate_fn,
    make_mol_repr,
    load_mol_cards_from_graphs,
    x_map,
    e_map
)
from baseline.train_gcn import MolGNN, load_molgnn_from_checkpoint
from model import GraphEncoder, load_graph_encoder_from_checkpoint
#from data_baseline.train_gcn_v3_gps import MolGNN_GPS, load_molgnn_gps_from_checkpoint

SUPPORTED_GNNS = {"MolGNN":load_molgnn_from_checkpoint, 
                "MolGNN_GPS":load_graph_encoder_from_checkpoint}




