import json
import logging
import os
import shutil
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from ogb.nodeproppred.evaluate import Evaluator as OGBEvaluator
from tgb.nodeproppred.evaluate import Evaluator
from tqdm import tqdm

import wandb
from evaluate_models_utils import (
    convert_ogb_predictions_from_one_hot,
    evaluate_model_node_classification,
)
from models.CompleteModel import TemporalNodeClassifier, create_dynamic_model
from utils.DataLoader import (
    get_idx_data_loader,
    get_node_classification_ogb,
    get_node_classification_tgb_data,
)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args
from utils.utils import (
    convert_to_gpu,
    create_optimizer,
    get_neighbor_sampler,
    get_parameter_sizes,
    set_random_seed,
)


def compute_train_metric(
    train_predicts_per_timeslot_dict, train_labels_per_timeslot_dict
):
    # compute the train metric for each timeslot
    for time_slot in tqdm(train_predicts_per_timeslot_dict):
        time_slot_predictions = np.stack(
            train_predicts_per_timeslot_dict[time_slot], axis=0
        )
        time_slot_labels = np.stack(train_labels_per_timeslot_dict[time_slot], axis=0)
        if args.dataset_name == "ogbn-arxiv":
            time_slot_labels, time_slot_predictions = (
                convert_ogb_predictions_from_one_hot(
                    time_slot_labels, time_slot_predictions
                )
            )

        # compute metric
        input_dict = {
            "y_true": time_slot_labels,
            "y_pred": time_slot_predictions,
            "eval_metric": [eval_metric_name],
        }
        eval_metric_result = evaluator.eval(input_dict)[eval_metric_name]
        train_metrics.append({eval_metric_name: eval_metric_result})
        wandb.log({f"train/{eval_metric_name}": eval_metric_result})
    return train_metrics


def log_metrics(metrics, stage: str):
    for metric_name in metrics[0].keys():
        mean_metric = np.mean([val_metric[metric_name] for val_metric in metrics])
        logger.info(f"{stage} {metric_name}, {mean_metric:.4f}")
        wandb.log({f"{stage}/{metric_name}": mean_metric})


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # get arguments
    args = get_node_classification_args(is_evaluation=False)

    wandb.login()
    wandb.init(project=args.dataset_name, config=args, name=args.model_name)

    # get data for training, validation and testing
    if args.dataset_name == "ogbn-arxiv":
        (
            node_raw_features,
            edge_raw_features,
            full_data,
            train_data,
            val_data,
            test_data,
            eval_metric_name,
            num_classes,
        ) = get_node_classification_ogb(
            dataset_name=args.dataset_name, use_messages=args.use_messages
        )
    else:
        (
            node_raw_features,
            edge_raw_features,
            full_data,
            train_data,
            val_data,
            test_data,
            eval_metric_name,
            num_classes,
        ) = get_node_classification_tgb_data(dataset_name=args.dataset_name)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
            exist_ok=True,
        )
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f"configuration is {args}")

        dynamic_backbone = create_dynamic_model(
            args=args,
            train_data=train_data,
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            train_neighbor_sampler=train_neighbor_sampler,
        )

        model = TemporalNodeClassifier(dynamic_backbone, args, num_classes=num_classes)
        model = torch.compile(model)

        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
            f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
        )

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if args.device == "cuda":
            model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        loss_func = nn.CrossEntropyLoss()
        if args.dataset_name == "ogbn-arxiv":
            evaluator = OGBEvaluator(name=args.dataset_name)
        else:
            evaluator = Evaluator(name=args.dataset_name)

        for epoch in range(args.num_epochs):
            model.train()
            if args.model_name in [
                "DyRep",
                "TGAT",
                "TGN",
                "CAWN",
                "TCL",
                "GraphMixer",
                "DyGFormer",
            ]:
                # training, only use training graph
                model.dynamic_backbone.set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # reinitialize memory of memory-based models at the start of each epoch
                model.dynamic_backbone.memory_bank.__init_memory_bank__()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            # store the results for each timeslot, and finally compute the metric for each timeslot
            # dictionary of list, key is the timeslot, value is a list, where each element is a prediction, np.ndarray with shape (num_classes, )
            train_predicts_per_timeslot_dict = defaultdict(list)
            # dictionary of list, key is the timeslot, value is a list, where each element is a label, np.ndarray with shape (num_classes, )
            train_labels_per_timeslot_dict = defaultdict(list)
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                    batch_labels,
                    batch_interact_types,
                    batch_node_label_times,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                    train_data.labels[train_data_indices],
                    train_data.interact_types[train_data_indices],
                    train_data.node_label_times[train_data_indices],
                )

                # split the batch data based on interaction types
                train_idx = torch.tensor(np.where(batch_interact_types == "train")[0])
                # val_idx = torch.tensor(np.where(batch_interact_types == 'validate')[0])
                # test_idx = torch.tensor(np.where(batch_interact_types == 'test')[0])
                # just_update_idx = torch.tensor(np.where(batch_interact_types == 'just_update')[0])
                # assert len(val_idx) == len(test_idx) == 0 and len(train_idx) + len(just_update_idx) == len(batch_interact_types), "The data are mixed!"

                # for memory-based models, we should use all the interactions to update memories (including 'train' and 'just_update'),
                # while other memory-free methods only need to compute on 'train'
                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # get temporal embedding of source and destination nodes, note that the memories are updated during the forward process
                    # two Tensors, with shape (batch_size, output_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = (
                        model.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                            src_node_ids=batch_src_node_ids,
                            dst_node_ids=batch_dst_node_ids,
                            node_interact_times=batch_node_interact_times,
                            edge_ids=batch_edge_ids,
                            edges_are_positive=True,
                            num_neighbors=args.num_neighbors,
                        )
                    )
                else:
                    if len(train_idx) > 0:
                        if args.model_name in ["TGAT", "CAWN", "TCL"]:
                            # get temporal embedding of source and destination nodes
                            # two Tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = (
                                model.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                                    src_node_ids=batch_src_node_ids,
                                    dst_node_ids=batch_dst_node_ids,
                                    node_interact_times=batch_node_interact_times,
                                    num_neighbors=args.num_neighbors,
                                )
                            )
                        elif args.model_name in ["GraphMixer"]:
                            # get temporal embedding of source and destination nodes
                            # two Tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = (
                                model.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                                    src_node_ids=batch_src_node_ids,
                                    dst_node_ids=batch_dst_node_ids,
                                    node_interact_times=batch_node_interact_times,
                                    num_neighbors=args.num_neighbors,
                                    time_gap=args.time_gap,
                                )
                            )
                        elif args.model_name in ["DyGFormer"]:
                            # get temporal embedding of source and destination nodes
                            # two Tensors, with shape (batch_size, output_dim)
                            batch_src_node_embeddings, batch_dst_node_embeddings = (
                                model.dynamic_backbone.compute_src_dst_node_temporal_embeddings(
                                    src_node_ids=batch_src_node_ids,
                                    dst_node_ids=batch_dst_node_ids,
                                    node_interact_times=batch_node_interact_times,
                                )
                            )
                        else:
                            raise ValueError(
                                f"Wrong value for model_name {args.model_name}!"
                            )
                    else:
                        batch_src_node_embeddings = None

                if len(train_idx) > 0:
                    # get predicted probabilities, shape (batch_size, num_classes)
                    predicts = model.node_classifier(
                        x=batch_src_node_embeddings
                    ).squeeze(dim=-1)
                    labels = torch.from_numpy(batch_labels).float().to(predicts.device)

                    loss = loss_func(
                        input=predicts[train_idx], target=labels[train_idx]
                    )
                    wandb.log({"train/loss": loss.item()})

                    train_losses.append(loss.item())
                    # append the predictions and labels to train_predicts_per_timeslot_dict and train_labels_per_timeslot_dict
                    for idx in train_idx:
                        train_predicts_per_timeslot_dict[
                            batch_node_label_times[idx]
                        ].append(predicts[idx].softmax(dim=0).cpu().detach().numpy())
                        train_labels_per_timeslot_dict[
                            batch_node_label_times[idx]
                        ].append(labels[idx].cpu().detach().numpy())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_idx_data_loader_tqdm.set_description(
                        f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item():.4f}"
                    )

                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model.dynamic_backbone.memory_bank.detach_memory_bank()

            wandb.log(
                {"train/loss_epoch": sum(train_losses) / len(train_idx_data_loader)}
            )

            train_metrics = compute_train_metric(
                train_predicts_per_timeslot_dict, train_labels_per_timeslot_dict
            )

            val_losses, val_metrics = evaluate_model_node_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                eval_stage="val",
                eval_metric_name=eval_metric_name,
                evaluator=evaluator,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

            log_metrics(val_metrics, "val")
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")

            if args.model_name in ["JODIE", "DyRep", "TGN"]:
                # backup memory bank after validating so it can be used for testing nodes (since test edges are strictly later in time than validation edges)
                val_backup_memory_bank = (
                    model.dynamic_backbone.memory_bank.backup_memory_bank()
                )

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}'
            )

            log_metrics(train_metrics, "train")

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_node_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_data=test_data,
                    eval_stage="test",
                    eval_metric_name=eval_metric_name,
                    evaluator=evaluator,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

                if args.model_name in ["JODIE", "DyRep", "TGN"]:
                    # reload validation memory bank for saving models
                    # note that since model treats memory as parameters, we need to reload the memory to val_backup_memory_bank for saving models
                    model.dynamic_backbone.memory_bank.reload_memory_bank(
                        val_backup_memory_bank
                    )

                logger.info(f"test loss: {np.mean(test_losses):.4f}")

                log_metrics(test_metrics, "test")

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (
                        metric_name,
                        np.mean(
                            [val_metric[metric_name] for val_metric in val_metrics]
                        ),
                        True,
                    )
                )
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f"get final performance on dataset {args.dataset_name}...")

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            val_losses, val_metrics = evaluate_model_node_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_data=val_data,
                eval_stage="val",
                eval_metric_name=eval_metric_name,
                evaluator=evaluator,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

        test_losses, test_metrics = evaluate_model_node_classification(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_data=test_data,
            eval_stage="test",
            eval_metric_name=eval_metric_name,
            evaluator=evaluator,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                average_val_metric = np.mean(
                    [val_metric[metric_name] for val_metric in val_metrics]
                )
                logger.info(f"validate {metric_name}, {average_val_metric:.4f}")
                val_metric_dict[metric_name] = average_val_metric

        logger.info(f"test loss: {np.mean(test_losses):.4f}")
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean(
                [test_metric[metric_name] for test_metric in test_metrics]
            )
            logger.info(f"test {metric_name}, {average_test_metric:.4f}")
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        if args.model_name not in ["JODIE", "DyRep", "TGN"]:
            result_json = {
                "validate metrics": {
                    metric_name: f"{val_metric_dict[metric_name]:.4f}"
                    for metric_name in val_metric_dict
                },
                "test metrics": {
                    metric_name: f"{test_metric_dict[metric_name]:.4f}"
                    for metric_name in test_metric_dict
                },
            }
        else:
            result_json = {
                "test metrics": {
                    metric_name: f"{test_metric_dict[metric_name]:.4f}"
                    for metric_name in test_metric_dict
                }
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json"
        )

        with open(save_result_path, "w") as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {args.num_runs} runs:")

    if args.model_name not in ["JODIE", "DyRep", "TGN"]:
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f"validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}"
            )
            logger.info(
                f"average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} "
                f"± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}"
            )

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f"test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}"
        )
        logger.info(
            f"average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} "
            f"± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}"
        )
        wandb.log(
            {
                f"test/{metric_name}": np.mean(
                    [
                        test_metric_single_run[metric_name]
                        for test_metric_single_run in test_metric_all_runs
                    ]
                )
            }
        )

    sys.exit()
