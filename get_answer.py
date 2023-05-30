from src.data.universal_dataset import UniversalDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import argparse
from src.utils import get_optimizers, write_data
import torch
import torch.nn as nn
import numpy as np
import os
import random
from src.model.universal_model import UniversalModel, UniversalModel_Roberta
from collections import Counter
from src.eval.utils import compute_value_for_incremental_equations, compute
from typing import List, Tuple
import logging
from transformers import set_seed
from universal_main import parse_arguments, class_name_2_model
import csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)


model_path = "/home/smetase/test/math23k"


def get_batched_prediction_consider_multiple_m0(feature, all_logits: torch.FloatTensor, constant_num: int):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    device = feature.variable_indexs_start.device
    batched_prediction = [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        current_max_num_variable = max_num_variable + constant_num + k
        num_var_range = torch.arange(0, current_max_num_variable, device=feature.variable_indexs_start.device)
        combination = torch.combinations(num_var_range, r=2, with_replacement=True)  ##number_of_combinations x 2
        num_combinations, _ = combination.size()

        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)  ## batch_size, num_combinations/num_m0, num_labels
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)  ## batch_size, num_combinations
        best_m0_score, best_comb = best_temp_score.max(dim=-1)  ## batch_size
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  ## batch_size
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label] ## batch size

        # batch_size x 2
        best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                          best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(device)).squeeze(1)
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label) in enumerate(zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):  ## within each instances:
            left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction


def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, fp16:bool, constant_values: List, uni_labels:List,
             res_file: str= None, err_file:str = None) -> Tuple[float, float]:

    model.eval()
    predictions = []

    constant_num = len(constant_values) if constant_values else 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                module = model.module if hasattr(model, 'module') else model
                all_logits = module(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids=feature.token_type_ids.to(dev),
                             variable_indexs_start=feature.variable_indexs_start.to(dev),
                             variable_indexs_end=feature.variable_indexs_end.to(dev),
                             num_variables = feature.num_variables.to(dev),
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             return_dict=True, is_eval=True).all_logits
                batched_prediction = get_batched_prediction_consider_multiple_m0(feature=feature, all_logits=all_logits, constant_num=constant_num)
                for b, inst_predictions in enumerate(batched_prediction):
                    for p, prediction_step in enumerate(inst_predictions):
                        left, right, op_id, stop_id = prediction_step
                        if stop_id == 1:
                            batched_prediction[b] = batched_prediction[b][:(p+1)]
                            break

                predictions.extend(batched_prediction)
    
        total = len(predictions)
        insts = valid_dataloader.dataset.insts

        answers = []
        for inst_predictions, inst in zip(predictions, insts):
            num_list = inst["num_list"]
            id = inst["id"]
            pred_val, incremental_equations = compute_value_for_incremental_equations(inst_predictions, num_list, constant_num, uni_labels, constant_values)
            answers.append(pred_val)
        return answers

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)

    bert_model_name = conf.bert_model_name if conf.bert_folder == "" or conf.bert_folder=="none" else f"{conf.bert_folder}/{conf.bert_model_name}"
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)


    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    num_labels = 6
    conf.uni_labels = uni_labels
    constant2id = {"1": 0, "PI": 1}
    conf.uni_labels = conf.uni_labels + ['^', '^_rev']
    num_labels = len(conf.uni_labels)
    constant_values = [1.0, 3.14]
    constant_number = len(constant_values)

    logger.info("Initializing model.")
    model = UniversalModel.from_pretrained(model_path,
                                            num_labels=num_labels,
                                            height = conf.height,
                                            constant_num = constant_number,
                                        var_update_mode=conf.var_update_mode).to(conf.device)

    logger.info("Reading questions")
    questions = UniversalDataset(file=conf.test_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                    constant2id=constant2id, constant_values=constant_values, data_max_height=conf.height, pretrained_model_name=bert_model_name)
    questions_dataloader = DataLoader(questions, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                  collate_fn=questions.collate_function)

    answers = evaluate(questions_dataloader, model, conf.device, uni_labels=conf.uni_labels, fp16=bool(conf.fp16), constant_values=constant_values,
                    res_file=res_file, err_file=err_file)

if __name__ == "__main__":
    main()