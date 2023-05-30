from src.data.universal_dataset import UniversalDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModel
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)


model_path = "/home/smetase/test/math23k"


class_name_2_model = {
        "bert-base-cased": UniversalModel,
        "roberta-base": UniversalModel_Roberta,
        "roberta-large": UniversalModel_Roberta,
        "coref-roberta-base": UniversalModel_Roberta,
        "bert-base-multilingual-cased": UniversalModel,
        'bert-base-chinese': UniversalModel,
        "xlm-roberta-base": UniversalModel_Roberta,
        'hfl/chinese-bert-wwm-ext': UniversalModel,
        'hfl/chinese-roberta-wwm-ext': UniversalModel,
    }

def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:0", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'], help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--train_num', type=int, default=-1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default=-1, help="The number of development data, -1 means all data")
    parser.add_argument('--test_num', type=int, default=-1, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="/home/smetase/Projects/Deductive-MWP-R/data/task_dataset/math23k_trainset.json")
    parser.add_argument('--dev_file', type=str, default="/home/smetase/Projects/Deductive-MWP-R/data/task_dataset/math23k_valset.json")
    parser.add_argument('--test_file', type=str, default="/home/smetase/Projects/Deductive-MWP-R/data/task_dataset/math23k_test.json")
    # parser.add_argument('--train_file', type=str, default="data/mawps-single/mawps_train_nodup.json")
    # parser.add_argument('--dev_file', type=str, default="data/mawps-single/mawps_test_nodup.json")

    parser.add_argument('--train_filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--test_filtered_steps', default=None, nargs='+', help="some heights to filter")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math_solver", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="hfl/chinese-bert-wwm-ext",
                        help="The bert model name to used")
    # parser.add_argument('--bert_folder', type=str, default="", help="The folder name that contains the BERT model")
    # parser.add_argument('--bert_model_name', type=str, default="roberta-base",
    #                     help="The bert model name to used")
    parser.add_argument('--height', type=int, default=10, help="the model height")
    parser.add_argument('--train_max_height', type=int, default=100, help="the maximum height for training data")

    parser.add_argument('--var_update_mode', type=str, default="gru", help="variable update mode")

    # training
    parser.add_argument('--mode', type=str, default="test", choices=["train", "test"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=20, help="The number of epochs to run")
    parser.add_argument('--fp16', type=int, default=0, choices=[0,1], help="using fp16 to train the model")

    parser.add_argument('--parallel', type=int, default=0, choices=[0,1], help="parallelizing model")

    # testing a pretrained model
    parser.add_argument('--cut_off', type=float, default=-100, help="cut off probability that we don't want to answer")
    parser.add_argument('--print_error', type=int, default=0, choices=[0, 1], help="whether to print the errors")
    parser.add_argument('--error_file', type=str, default="results/error.json", help="The file to print the errors")
    parser.add_argument('--result_file', type=str, default="results/res.json",
                        help="The file to print the errors")

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(f"{k} = {args.__dict__[k]}")
    return args



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
    labels = []
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

                # for b, inst_labels in enumerate(batched_labels):
                #     for p, label_step in enumerate(inst_labels):
                #         left, right, op_id, stop_id = label_step
                #         if stop_id == 1:
                #             batched_labels[b] = batched_labels[b][:(p+1)]
                #             break

                predictions.extend(batched_prediction)

    
        total = 1200
        insts = valid_dataloader.dataset.insts
        ##value accuarcy
        val_corr = 0
        num_label_step_val_corr = Counter()
        ret = []
        corr = 0
        for inst_predictions, inst in zip(predictions, insts):
            num_list = inst["num_list"]
            pred_val, _ = compute_value_for_incremental_equations(inst_predictions, num_list, constant_num, uni_labels, constant_values)
            ret.append(pred_val)
        return ret

def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    os.makedirs("results", exist_ok=True)
    bert_model_name = conf.bert_model_name if conf.bert_folder == "" or conf.bert_folder=="none" else f"{conf.bert_folder}/{conf.bert_model_name}"

    tokenizer = AutoTokenizer.from_pretrained(bert_model_name, use_fast=True)


    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    num_labels = 6
    conf.uni_labels = uni_labels
    if "23k" in conf.train_file:
        constant2id = {"1": 0, "PI": 1}
        conf.uni_labels = conf.uni_labels + ['^', '^_rev']
        num_labels = len(conf.uni_labels)
        constant_values = [1.0, 3.14]
        constant_number = len(constant_values)


    # Read dataset
    if opt.mode == "train":
        logger.info("[Data Info] Reading training data")
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.train_num, filtered_steps=opt.train_filtered_steps,
                                   constant2id=constant2id, constant_values=constant_values,
                                   data_max_height=opt.train_max_height, pretrained_model_name=bert_model_name)
        logger.info("[Data Info] Reading validation data")
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values,
                                        data_max_height=conf.height, pretrained_model_name=bert_model_name)

        logger.info("[Data Info] Reading Testing data data")
        test_dataset = None
        if os.path.exists(conf.test_file):
            test_dataset = UniversalDataset(file=conf.test_file, tokenizer=tokenizer, uni_labels=conf.uni_labels,
                                            number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                            constant2id=constant2id, constant_values=constant_values,
                                            data_max_height=conf.height, pretrained_model_name=bert_model_name)
        logger.info(f"[Data Info] Training instances: {len(dataset)}, Validation instances: {len(eval_dataset)}")
        if test_dataset is not None:
            logger.info(f"[Data Info] Testing instances: {len(test_dataset)}")
        # Prepare data loader
        logger.info("[Data Info] Loading data")
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)
        test_loader = None
        if test_dataset is not None:
            logger.info("[Data Info] Loading Test data")
            test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)

        res_file = f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name = bert_model_name,
                      valid_dataloader = valid_dataloader, test_dataloader=test_loader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                      constant_values=constant_values, res_file=res_file, error_file=err_file)
        evaluate(valid_dataloader, model, conf.device, fp16=bool(conf.fp16), constant_values=constant_values, uni_labels=conf.uni_labels)
    else:
        logger.info(f"Testing the model now.")
        MODEL_CLASS = class_name_2_model[bert_model_name]
        model = MODEL_CLASS.from_pretrained(model_path,
                                               num_labels=num_labels,
                                               height = conf.height,
                                               constant_num = constant_number,
                                            var_update_mode=conf.var_update_mode).to(conf.device)
        logger.info("[Data Info] Reading test data")
        eval_dataset = UniversalDataset(file=conf.test_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, data_max_height=conf.height, pretrained_model_name=bert_model_name)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        os.makedirs("results", exist_ok=True)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        ret = evaluate(valid_dataloader, model, conf.device, uni_labels=conf.uni_labels, fp16=bool(conf.fp16), constant_values=constant_values,
                 res_file=res_file, err_file=err_file)

if __name__ == "__main__":
    # logger.addHandler(logging.StreamHandler())
    main()