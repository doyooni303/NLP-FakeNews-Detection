import time
import numpy as np
import pandas as pd
import wandb
import json
import logging
import os
import torch
import argparse
import yaml
import transformers
import pdb

from datasets import load_metric
from transformers import get_cosine_schedule_with_warmup
from models import create_model
from dataset import create_dataset, create_dataloader, create_tokenizer

from utils import convert_device

from log import setup_default_logging
from utils import torch_seed


_logger = logging.getLogger("pretrain")
transformers.logging.set_verbosity_error()


class RunningAverage:
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class Generation_Metrics(object):
    def __init__(self, metric_names: list) -> None:

        self.metric_names = metric_names
        self.metrics, self.results = {}, {}
        for name in self.metric_names:
            self.metrics[name] = load_metric(name)
            self.results[name] = RunningAverage()

    def compute(self, metric_name: str, predictions: list, references: list):
        if metric_name == "bleu":
            tokenized_predictions = [[prediction.split() for prediction in predictions]]
            tokenized_references = [[reference.split() for reference in references]]
            results = self.metrics[metric_name].compute(
                predictions=tokenized_predictions, references=tokenized_references
            )
            self.results[metric_name].update(results[metric_name])

        elif metric_name == "rouge":  # rougeL만 계산
            results = self.metrics[metric_name].compute(
                predictions=predictions, references=references
            )["rougeL"]
            self.results[metric_name].update(results.mid.fmeasure)

    def get_results(self, metric_name):
        return self.results[metric_name]()


def pretraining(
    model,
    tokenizer,
    num_training_steps: int,
    trainloader,
    validloader,
    optimizer,
    scheduler,
    log_interval: int,
    eval_interval: int,
    savedir: str,
    use_wandb: bool,
    accumulation_steps: int = 1,
    device: str = "cpu",
    metric_names=["bleu"],
):

    gen_metrics = Generation_Metrics(metric_names)
    data_time_m = RunningAverage()
    losses_m = RunningAverage()
    batch_time_m = RunningAverage()

    end = time.time()

    model.train()
    optimizer.zero_grad()

    step = 0
    best_score = 0
    train_mode = True
    while train_mode:
        for inputs, targets in trainloader:
            # batch
            inputs, targets = convert_device(inputs, device), targets.to(device)
            
            ## tensor size 변경:{batch_size, 1, seq_length}->{batch_size,seq_length}
            inputs = {key:value.squeeze(dim=1) for key, value in inputs.items()}
            targets = targets.squeeze(dim=1)
            print(targets)
            data_time_m.update(time.time() - end)

            # optimizer condition
            opt_cond = (step + 1) % accumulation_steps == 0

            # predict
            references = tokenizer.batch_decode(targets, skip_special_tokens=True)
            
            outputs = model.generate(inputs)
            loss = model(inputs,labels=targets).loss
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # generation metric score
            for metric_name in metric_names:
                gen_metrics.compute(
                    metric_name=metric_name,
                    predictions=predictions,
                    references=references,
                )

            # loss for accumulation steps
            loss /= accumulation_steps
            loss.backward()

            if opt_cond:
                # loss update
                optimizer.step()
                optimizer.zero_grad()

                if scheduler:
                    scheduler.step()

                losses_m.update(loss.item() * accumulation_steps)

                batch_time_m.update(time.time() - end)
                
                train_log = dict([(f"train_{k}", v()) for k, v in gen_metrics.results.items()])
                train_log["train_loss"] = losses_m()
                # wandb
                if use_wandb:
                    wandb.log(
                        train_log,
                        step=step,
                    )

                if ((step + 1) // accumulation_steps) % log_interval == 0 or step == 0:

                    _logger.info(
                        "TRAIN [{:>4d}/{}] Loss: {loss:>6.4f} "
                        "BLEU: {bleu:.2%} "
                        #"RougeL: {rouge:.2%} "
                        "LR: {lr:.3e} "
                        "Time: {batch_time:.3f}s "  # {rate:>7.2f}/s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s) '
                        "Data: {data_time:.3f} ".format(
                            (step + 1) // accumulation_steps,
                            num_training_steps,
                            num_training_steps,
                            loss=losses_m(),
                            bleu=100 * gen_metrics.get_results("bleu"),
                            #rouge=100 * gen_metrics.get_results("rouge"),
                            lr=optimizer.param_groups[0]["lr"],
                            batch_time=batch_time_m(),
                            data_time=data_time_m(),
                        )
                    )

                if (
                    ((step + 1) // accumulation_steps) % eval_interval == 0
                    and step != 0
                ) or step + 1 == num_training_steps:
                    eval_metrics = evaluate(
                        model,
                        tokenizer,
                        validloader,
                        log_interval,
                        device,
                    )
                    model.train()
                    eval_log = dict([(f"eval_{k}", v) for k, v in eval_metrics.items()])
                    # wandb
                    if use_wandb:

                        wandb.log(eval_log, step=step)

                    # checkpoint
                    if best_score < np.mean(np.array(eval_metrics.values())):
                        # save best score
                        state = {"best_step": step}
                        state.update(eval_log)
                        json.dump(
                            state,
                            open(os.path.join(savedir, "best_score.json"), "w"),
                            indent=4,
                        )

                        # save best model
                        torch.save(
                            model.state_dict(), os.path.join(savedir, f"best_model.pt")
                        )

                        best_score = np.mean(np.array(eval_metrics.values()))
                        best_metrics = eval_metrics

            end = time.time()

            step += 1

            if (step // accumulation_steps) >= num_training_steps:
                train_mode = False
                break

    # save best model
    torch.save(model.state_dict(), os.path.join(savedir, f"latest_model.pt"))

    _logger.info(
        "Best Metrics: {0} (step {1})".format(best_metrics, state["best_step"])
    )


def evaluate(
    model,
    tokenizer,
    dataloader,
    log_interval: int,
    device: str = "cpu",
    sample_check: bool = False,
    metric_names=["bleu"],
):
    gen_metrics = Generation_Metrics(metric_names)
    model.eval()
    with torch.no_grad():
        for idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = convert_device(inputs, device), targets.to(device)
            
            ## tensor size 변경:{batch_size, 1, seq_length}->{batch_size,seq_length}
            inputs = {key:value.squeeze(dim=1) for key, value in inputs.items()}
            targets = targets.squeeze(dim=1)
            references = tokenizer.batch_decode(targets, skip_special_tokens=True)

            # predict
            outputs = model.generate(inputs)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # generation metric score
            for metric_name in metric_names:
                gen_metrics.compute(
                    metric_name=metric_name,
                    predictions=predictions,
                    references=references,
                )

            if idx % log_interval == 0 and idx != 0:
                _logger.info(
                    "TEST [%d/%d]: BLEU: %.2f " #| Rouge: %.2f"
                    % (
                        idx + 1,
                        len(dataloader),
                        100 * gen_metrics.get_results("bleu"),
                        #100 * gen_metrics.get_results("rouge"),
                    )
                )

    _logger.info(
        "TEST: BLEU: %.2f " #| RougeL: %.2f"
        % (
            100 * gen_metrics.get_results("bleu"),
            # 100 * gen_metrics.get_results("rouge"),
        )
    )

    scores = {key: 100 * value() for key, value in gen_metrics.results.itmes()}
    if sample_check:
        results = {
            "targets": references,
            "preds": predictions,
        }
        return scores, results
    else:
        return scores


def run(cfg):

    # setting seed and device
    setup_default_logging()
    torch_seed(cfg["SEED"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _logger.info("Device: {}".format(device))

    # savedir
    savedir = os.path.join(cfg["RESULT"]["savedir"], cfg["EXP_NAME"])
    os.makedirs(savedir, exist_ok=True)

    # tokenizer
    tokenizer, word_embed = create_tokenizer(
        name=cfg["TOKENIZER"]["name"],
        vocab_path=cfg["TOKENIZER"].get("vocab_path", None),
        max_vocab_size=cfg["TOKENIZER"].get("max_vocab_size", None),
    )

    # Build Model
    model = create_model(
        modelname=cfg["MODEL"]["modelname"],
        hparams=cfg["MODEL"]["PARAMETERS"],
        word_embed=word_embed,
        tokenizer=tokenizer,
        freeze_word_embed=cfg["MODEL"].get("freeze_word_embed", False),
        use_pretrained_word_embed=cfg["MODEL"].get("use_pretrained_word_embed", False),
        checkpoint_path=cfg["MODEL"]["CHECKPOINT"]["checkpoint_path"],
    )
    model.to(device)

    _logger.info(
        "# of trainable params: {}".format(
            np.sum([p.numel() if p.requires_grad else 0 for p in model.parameters()])
        )
    )

    if cfg["MODE"]["do_train"]:
        # wandb
        if cfg["TRAIN"]["use_wandb"]:
            if "category_select" in cfg["EXP_NAME"]:
                wandb.init(
                    name=cfg["DATASET"]["name"]
                    if cfg["DATASET"]["wandb_name"] is None
                    else cfg["DATASET"]["wandb_name"],  # ModelName과 동일
                    group=cfg["DATASET"]["method"],
                    entity="fakenews-detection",
                    project="Fake-News-Detection-Task1-Direct",
                    config=cfg,
                )

            else:
                wandb.init(
                    name=cfg["EXP_NAME"],
                    entity="fakenews-detection",
                    project="Fake-News-Detection-Task1",
                    config=cfg,
                )

        # Build datasets
        trainset = create_dataset(
            name=cfg["DATASET"]["name"],
            data_path=cfg["DATASET"]["data_path"],
            direct_path=cfg["DATASET"].get("direct_path", None),
            split="train",
            tokenizer=tokenizer,
            saved_data_path=cfg["DATASET"]["saved_data_path"],
            **cfg["DATASET"]["PARAMETERS"],
        )

        validset = create_dataset(
            name=cfg["DATASET"]["name"],
            data_path=cfg["DATASET"]["data_path"],
            direct_path=cfg["DATASET"].get("direct_path", None),
            split="validation",
            tokenizer=tokenizer,
            saved_data_path=cfg["DATASET"]["saved_data_path"],
            **cfg["DATASET"]["PARAMETERS"],
        )

        trainloader = create_dataloader(
            dataset=trainset,
            batch_size=cfg["TRAIN"]["batch_size"],
            num_workers=cfg["TRAIN"]["num_workers"],
            shuffle=True,
        )
        validloader = create_dataloader(
            dataset=validset,
            batch_size=cfg["TRAIN"]["batch_size"],
            num_workers=cfg["TRAIN"]["num_workers"],
            shuffle=False,
        )

        # Set training
        optimizer = torch.optim.AdamW(
            params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg["OPTIMIZER"]["lr"],
            weight_decay=cfg["OPTIMIZER"]["weight_decay"],
        )

        # scheduler
        if cfg["SCHEDULER"]["use_scheduler"]:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(
                    cfg["TRAIN"]["num_training_steps"]
                    * cfg["SCHEDULER"]["warmup_ratio"]
                ),
                num_training_steps=cfg["TRAIN"]["num_training_steps"],
            )
        else:
            scheduler = None

        # Fitting model
        pretraining(
            model=model,
            tokenizer=tokenizer,
            num_training_steps=cfg["TRAIN"]["num_training_steps"],
            trainloader=trainloader,
            validloader=validloader,
            optimizer=optimizer,
            scheduler=scheduler,
            log_interval=cfg["LOG"]["log_interval"],
            eval_interval=cfg["LOG"]["eval_interval"],
            savedir=savedir,
            accumulation_steps=cfg["TRAIN"]["accumulation_steps"],
            device=device,
            use_wandb=cfg["TRAIN"]["use_wandb"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection - Task1")
    parser.add_argument("--yaml_config", type=str, default=None, help="exp config file")
    parser.add_argument("--method", type=str, default=None, help="Type of dataset")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of experience")
    parser.add_argument("--data_name", type=str, default=None, help="Name of dataset")
    args = parser.parse_args()

    # config
    cfg = yaml.load(open(args.yaml_config, "r"), Loader=yaml.FullLoader)
    cfg["DATASET"]["method"] = args.method
    if args.exp_name is not None:
        cfg["EXP_NAME"] = args.exp_name
    cfg["DATASET"]["wandb_name"] = args.data_name
    run(cfg)