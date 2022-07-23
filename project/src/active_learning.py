# import argparse
# import logging
# import os
# import sys
# import time
# from collections import defaultdict

# import numpy as np
# import torch
# from project.src.preprocessing.readPoolData import read_pool_data
# from project.src.preprocessing.check_data import check_and_get_data
# from project.src.preprocessing.prepare_data_for_cal import prepare_data_for_cal
# from project.src.preprocessing.prepare_data_for_dal import prepare_data_for_dal
# from project.src.preprocessing.split_data import split_data
# from project.src.train.generate_cartography import (generate_cartography, generate_cartography_after_intervals,
#                                                     generate_cartography_by_idx, transform_correctness_to_bins)
# from project.src.train.initialize_train_pool_test import initialize_train_pool_test
# from project.src.train.train_cal_estimator import CALEstimator
# from project.src.train.train_dal_estimator import DALEstimator
# from project.src.train.train_estimator import MLPEstimator
# from project.src.utils.apply_acquisition_function import apply_acquisition_function
# from project.src.utils.get_vocab_and_labels import get_vector_matrix, get_vocab_and_label
# from project.src.utils.get_weight_distribution import get_distribution_weights
# from project.src.utils.remove_pool_add_train import add_and_remove_instances
# from project.src.utils.save_cartography import save_cartography
# from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BertConfig
# import torch.nn as nn

# logger = logging.getLogger(__name__)


# class Model(nn.Module):
#     def __init__(self, num_labels, config):
#         super().__init__()
#         self.num_labels = num_labels
#         config.num_labels = num_labels
#         self.config = config
#         self.bert = BertModel.from_pretrained("bert-base-uncased", config = config)
#         for param in self.bert.parameters():
#             param.requires_grad = False
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.Linear1 = nn.Linear(self.config.hidden_size, 256)
#         self.Linear2 = nn.Linear(256, 64)
#         self.Linear3 = nn.Linear(64,16)
#         self.same_linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
#         self.batchnorm1 = nn.BatchNorm1d(256)
#         self.batchnorm2 = nn.BatchNorm1d(64)
#         self.batchnorm3 = nn.BatchNorm1d(16)
#         self.relu = nn.ReLU()
#         self.classifier = nn.Linear(16, config.num_labels)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         #print("return_dict", return_dict)
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         #print(outputs)
#         pooled_output = outputs[1]
        
#         temp_outputs = self.Linear1(pooled_output)
#         temp_outputs = self.batchnorm1
#         (temp_outputs)
#         temp_outputs = self.relu(temp_outputs)
        
#         temp_outputs = self.Linear2(temp_outputs)
#         temp_outputs = self.batchnorm2(temp_outputs)
#         temp_outputs = self.relu(temp_outputs)
#         temp_outputs = self.dropout(temp_outputs)
        
#         temp_outputs = self.Linear3(temp_outputs)
#         temp_outputs = self.batchnorm3(temp_outputs)
#         temp_outputs = self.relu(temp_outputs)
        
#         logits = self.classifier(temp_outputs)
#         return self.log_softmax(logits)
#     def forward_discriminative(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
#             config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
#             `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         #print("return_dict", return_dict)
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         #print(outputs)
#         pooled_output = outputs[1]
        
# #         temp_outputs = self.Linear1(pooled_output)
# #         temp_outputs = self.batchnorm1
# #         (temp_outputs)
#         temp_outputs = self.relu(temp_outputs)
        
#         temp_outputs = self.same_linear(temp_outputs)
# #         temp_outputs = self.batchnorm2(temp_outputs)
# #         temp_outputs = self.relu(temp_outputs)
#         temp_outputs = self.dropout(temp_outputs)
        
# #         temp_outputs = self.Linear3(temp_outputs)
# #         temp_outputs = self.batchnorm3(temp_outputs)
# #         temp_outputs = self.relu(temp_outputs)
        
# #         logits = self.classifier(temp_outputs)
#         return temp_outputs


# def getModelAndTokenizer():
#     tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#     config  = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
#     num_labels = 2
#     config.num_labels = num_labels
#     model = Model(num_labels, config)
#     return (tokenizer, model)


# def reset_estimator(estimator) -> None:
#     estimator.weight_reset()
#     estimator.probabilities.clear()
#     estimator.correctness.clear()
#     estimator.gold_labels.clear()


# def start_active_learning(args: argparse.Namespace) -> tuple:
#     logger.info("{:30} {:25} {:30}".format("-" * 25, "Initializing Data", "-" * 25))

#     train, test = check_and_get_data(args)
#     pool_path = os.getenv("TREC_POOL")
#     #train = split_data(train, args.initial_size)
#     pool = read_pool_data(pool_path)
#     word_to_idx, label_to_idx, vocab_size, num_labels = get_vocab_and_label(train, test, pool)
#     embedding_matrix = get_vector_matrix(args, word_to_idx)
#     emb_dim = embedding_matrix.shape[1]

#     logging.info(f"Vocabulary size: {vocab_size}, "
#                  f"number of labels: {num_labels}")
#     if args.pretrained:
#         logging.info(f"pretrained embedding size: {embedding_matrix.shape}")
# #     tokenizer, model =  getModelAndTokenizer()
#     #X_train, y_train, X_pool, X_test, y_test = initialize_train_pool_test(args, train, pool, test, tokenizer)
#     X_train, y_train, X_pool, X_test, y_test = initialize_train_pool_test(args, train, pool, test,word_to_idx, label_to_idx)
#     print(len(X_train))
#     print(len(X_pool))
#     print(len(X_test))
#     #print(X_train[0])
#     #estimator = MLPEstimator(args, 2, model, tokenizer)
#     estimator = MLPEstimator(args, vocab_size, emb_dim, num_labels, embedding_matrix)
#     cartography = {"interval": [], "correctness": [], "variability": [], "confidence": []}

#     # train model on initial set / create cartography
#     if args.cartography:
#         estimator.train(X_train, y_train)
#         logger.info("{:30} {:25} {:30}".format("-" * 25, "Generating Cartography", "-" * 25))
#         if args.plot:
#             cartography = generate_cartography(cartography, estimator.probabilities, estimator.correctness)
#             generate_cartography_after_intervals(args, cartography)
#         else:
#             cartography = generate_cartography_by_idx(cartography, estimator.probabilities, estimator.correctness)
#             save_cartography(args, cartography)
#         sys.exit(-1)
#     else:
#         logger.info("{:30} {:25} {:30}".format("-" * 25, "Training Model", "-" * 25))
#         X_train_rep = estimator.train(X_train, y_train)
#         initial_accuracy = estimator.evaluate(X_test, y_test)
#         logger.info(f"Initial accuracy of estimator: {initial_accuracy}")
#         logger.info("{:30} {:25} {:30}".format("-" * 25, "Starting Iterations", "-" * 25))

#     active_learning_accuracy_history = [initial_accuracy]
#     selected_top_k, confidence_stats, variability_stats, correctness_stats = [], [], [], []

#     for i in range(int(os.getenv("ITERATIONS"))):
#         logging.info(f"Active learning iteration: {i + 1}, train size: {len(X_train)}, pool size: {len(X_pool)}")

#         if args.acquisition == "discriminative" or args.acquisition == "cartography":
#             # prepare representations and data for DAL
#             if i != 0:
#                 X_train_rep = estimator.train(X_train, y_train)
#             X_pool_rep = estimator.predict(X_pool)

#             if args.acquisition == "discriminative":
#                 X_train_dal, y_train_dal = prepare_data_for_dal(X_train_rep, X_pool_rep)
#                 class_weights = get_distribution_weights(y_train_dal)
#                 dal_estimator = DALEstimator(args, len(X_train_rep), X_train_rep[0].size, len(np.unique(y_train_dal)),
#                                              class_weights)
#                 top_k_indices = dal_estimator.train(X_train_dal, y_train_dal)
#                 dal_estimator.weight_reset()

#             elif args.acquisition == "cartography":
#                 X_train_cal, y_train_cal, X_pool_cal, y_pool_cal = prepare_data_for_cal(X_train_rep, X_pool_rep,
#                                                                                         estimator.correctness)
#                 class_weights = get_distribution_weights(y_train_cal)
#                 cal_estimator = CALEstimator(args, len(X_train_cal), X_train_rep[0].size, len(np.unique(y_train_cal)),
#                                              class_weights)
#                 cal_estimator.train(X_train_cal, y_train_cal)
#                 top_k_indices = cal_estimator.predict(X_pool_cal)
#                 cal_estimator.weight_reset()

#         else:
#             # apply model to the pool to retrieve top-k instances
#             probas = estimator.predict(X_pool)
#             top_k_indices = apply_acquisition_function(args, probas)

#         # add top-k instances from pool to train and remove from pool
#         #X_train, y_train, X_pool= add_and_remove_instances(X_train, y_train, X_pool, top_k_indices)

#         # retrain model, save accuracy, weight reset for next iter
#         #estimator.train(X_train, y_train)
# #         accuracy = estimator.evaluate(X_test, y_test)
# #         active_learning_accuracy_history.append(accuracy)

#         if args.analysis:
#             selected_top_k.append(top_k_indices)

#             if i != 0:
#                 confidences = {idx: sum(proba) / len(proba) for idx, proba in list(estimator.probabilities.items())}
#                 variability = {idx: np.std(proba) for idx, proba in list(estimator.probabilities.items())}
#                 correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in
#                                list(estimator.correctness.items())}
#                 confidence_stats.append(
#                     np.mean(list(confidences.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
#                 variability_stats.append(
#                     np.mean(list(variability.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
#                 correctness_stats.append(
#                     np.mean(list(correctness.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))

#         reset_estimator(estimator)
#         #logger.info(f"Accuracy history: {active_learning_accuracy_history}")

#     return active_learning_accuracy_history, selected_top_k, confidence_stats, variability_stats, correctness_stats

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from project.src.preprocessing.readPoolData import read_pool_data
from project.src.preprocessing.check_data import check_and_get_data
from project.src.preprocessing.prepare_data_for_cal import prepare_data_for_cal
from project.src.preprocessing.prepare_data_for_dal import prepare_data_for_dal
from project.src.preprocessing.split_data import split_data
from project.src.train.generate_cartography import (generate_cartography, generate_cartography_after_intervals,
                                                    generate_cartography_by_idx, transform_correctness_to_bins)
from project.src.train.initialize_train_pool_test import initialize_train_pool_test, initialize_train_pool_test2
from project.src.train.train_bert_estimator import BertEstimator
from project.src.train.train_cal_estimator import CALEstimator
from project.src.train.train_dal_estimator import DALEstimator
from project.src.train.train_cal_bert_estimator import BERTCALEstimator
from project.src.train.train_estimator import MLPEstimator
from project.src.utils.apply_acquisition_function import apply_acquisition_function
from project.src.utils.get_vocab_and_labels import get_vector_matrix, get_vocab_and_label
from project.src.utils.get_weight_distribution import get_distribution_weights
from project.src.utils.remove_pool_add_train import add_and_remove_instances
from project.src.utils.save_cartography import save_cartography
from transformers import BertForSequenceClassification, BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from project.src.estimators.bert_estimator import Model, getModelAndTokenizer

logger = logging.getLogger(__name__)


def reset_estimator(estimator) -> None:
    #estimator.weight_reset()
    estimator.probabilities.clear()
    estimator.correctness.clear()
    estimator.gold_labels.clear()


def start_active_learning(args: argparse.Namespace) -> tuple:
    if args.model =="bert": 
        logger.info("{:30} {:25} {:30}".format("-" * 25, "Initializing Data", "-" * 25))

        train, test = check_and_get_data(args)
        pool_path = os.getenv("TREC_POOL")
        #train = split_data(train, args.initial_size)
        pool = read_pool_data(pool_path)
        # word_to_idx, label_to_idx, vocab_size, num_labels = get_vocab_and_label(train, test, pool)
        # embedding_matrix = get_vector_matrix(args, word_to_idx)
        # emb_dim = embedding_matrix.shape[1]

        # logging.info(f"Vocabulary size: {vocab_size}, "
        #              f"number of labels: {num_labels}")
        # if args.pretrained:
        #     logging.info(f"pretrained embedding size: {embedding_matrix.shape}")
        tokenizer, model =  getModelAndTokenizer()
        X_train, y_train, X_pool, X_test, y_test = initialize_train_pool_test(args, train, pool, test, tokenizer)

        #estimator = MLPEstimator(args)
        estimator = BertEstimator(args, model=model)
        cartography = {"interval": [], "correctness": [], "variability": [], "confidence": []}

        # train model on initial set / create cartography
        if args.cartography:
            estimator.train(X_train, y_train)
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Generating Cartography", "-" * 25))
            if args.plot:
                cartography = generate_cartography(cartography, estimator.probabilities, estimator.correctness)
                generate_cartography_after_intervals(args, cartography)
            else:
                cartography = generate_cartography_by_idx(cartography, estimator.probabilities, estimator.correctness)
                save_cartography(args, cartography)
            sys.exit(-1)
        else:
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Training Model", "-" * 25))
            X_train_rep = estimator.train(X_train, y_train)
            initial_accuracy = estimator.evaluate(X_test, y_test)
            #initial_accuracy = 0
            logger.info(f"Initial accuracy of estimator: {initial_accuracy}")
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Starting Iterations", "-" * 25))

        active_learning_accuracy_history = [initial_accuracy]
        selected_top_k, confidence_stats, variability_stats, correctness_stats = [], [], [], []

        for i in range(int(os.getenv("ITERATIONS"))):
            logging.info(f"Active learning iteration: {i + 1}, train size: {len(X_train)}, pool size: {len(X_pool)}")

            if args.acquisition == "discriminative" or args.acquisition == "cartography":
                # prepare representations and data for DAL
                if i != 0:
                    X_train_rep = estimator.train(X_train, y_train)
                X_pool_rep = estimator.predict(X_pool)
                estimator.predict_class(X_pool,pool_path)
                if args.acquisition == "discriminative":
                    X_train_dal, y_train_dal = prepare_data_for_dal(X_train_rep, X_pool_rep)
                    class_weights = get_distribution_weights(y_train_dal)
                    dal_estimator = DALEstimator(args, len(X_train_rep), X_train_rep[0].size, len(np.unique(y_train_dal)),
                                                 class_weights)
                    top_k_indices = dal_estimator.train(X_train_dal, y_train_dal)
                    dal_estimator.weight_reset()

                elif args.acquisition == "cartography":
                    X_train_cal, y_train_cal, X_pool_cal, y_pool_cal = prepare_data_for_cal(X_train_rep, X_pool_rep,
                                                                                            estimator.correctness)
                    class_weights = get_distribution_weights(y_train_cal)
                    cal_estimator = CALEstimator(args, len(X_train_cal), X_train_rep[0].size, len(np.unique(y_train_cal)),
                                                class_weights)
#                     cal_estimator = BERTCALEstimator(args, model=model)
                    cal_estimator.train(X_train_cal, y_train_cal)
                    top_k_indices = cal_estimator.predict(X_pool_cal)
                    cal_estimator.weight_reset()

            else:
                # apply model to the pool to retrieve top-k instances
                probas = estimator.predict(X_pool)
                top_k_indices = apply_acquisition_function(args, probas)

            # add top-k instances from pool to train and remove from pool
            #X_train, y_train, X_pool= add_and_remove_instances(X_train, y_train, X_pool, top_k_indices)

            # retrain model, save accuracy, weight reset for next iter
            #estimator.train(X_train, y_train)
#             accuracy = estimator.evaluate(X_test, y_test)
#             active_learning_accuracy_history.append(accuracy)
            print(top_k_indices)
    #         for j in range(len(top_k_run)):
    #             idx_list = top_k_run[j]
    #             selected_top_k_dict[j].append([idx for idx in idx_list])

    #         with open(f"{os.getenv('INDICES_PATH')}/top_k_{args.task}_{args.acquisition}_{i}.json", "w") as f:
    #             json.dump(selected_top_k_dict, f)
            if args.analysis:
                selected_top_k.append(top_k_indices)

                if i != 0:
                    confidences = {idx: sum(proba) / len(proba) for idx, proba in list(estimator.probabilities.items())}
                    variability = {idx: np.std(proba) for idx, proba in list(estimator.probabilities.items())}
                    correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in
                                   list(estimator.correctness.items())}
                    confidence_stats.append(
                        np.mean(list(confidences.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                    variability_stats.append(
                        np.mean(list(variability.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                    correctness_stats.append(
                        np.mean(list(correctness.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))

            reset_estimator(estimator)
            logger.info(f"Accuracy history: {active_learning_accuracy_history}")
        logger.info("Returning from active learning! good job")
        ##return active_learning_accuracy_history, selected_top_k, confidence_stats, variability_stats, correctness_stats
        return selected_top_k
    else:
        logger.info("{:30} {:25} {:30}".format("-" * 25, "Initializing Data", "-" * 25))

        train, test = check_and_get_data(args)
        pool_path = os.getenv("TREC_POOL")
        #train = split_data(train, args.initial_size)
        pool = read_pool_data(pool_path)
        word_to_idx, label_to_idx, vocab_size, num_labels = get_vocab_and_label(train, test, pool)
        embedding_matrix = get_vector_matrix(args, word_to_idx)
        emb_dim = embedding_matrix.shape[1]

        logging.info(f"Vocabulary size: {vocab_size}, "
                     f"number of labels: {num_labels}")
        if args.pretrained:
            logging.info(f"pretrained embedding size: {embedding_matrix.shape}")
        #tokenizer, model =  getModelAndTokenizer()
        X_train, y_train, X_pool, X_test, y_test = initialize_train_pool_test2(args, train, pool, test, word_to_idx, label_to_idx)

        estimator = MLPEstimator(args, vocab_size, emb_dim, num_labels, embedding_matrix)
        #estimator = BertEstimator(args, model=model)
        cartography = {"interval": [], "correctness": [], "variability": [], "confidence": []}

        # train model on initial set / create cartography
        if args.cartography:
            estimator.train(X_train, y_train)
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Generating Cartography", "-" * 25))
            if args.plot:
                cartography = generate_cartography(cartography, estimator.probabilities, estimator.correctness)
                generate_cartography_after_intervals(args, cartography)
            else:
                cartography = generate_cartography_by_idx(cartography, estimator.probabilities, estimator.correctness)
                save_cartography(args, cartography)
            sys.exit(-1)
        else:
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Training Model", "-" * 25))
            X_train_rep = estimator.train(X_train, y_train)
            initial_accuracy = estimator.evaluate(X_test, y_test)
            #initial_accuracy = 0
            logger.info(f"Initial accuracy of estimator: {initial_accuracy}")
            logger.info("{:30} {:25} {:30}".format("-" * 25, "Starting Iterations", "-" * 25))

        active_learning_accuracy_history = [initial_accuracy]
        selected_top_k, confidence_stats, variability_stats, correctness_stats = [], [], [], []

        for i in range(int(os.getenv("ITERATIONS"))):
            logging.info(f"Active learning iteration: {i + 1}, train size: {len(X_train)}, pool size: {len(X_pool)}")

            if args.acquisition == "discriminative" or args.acquisition == "cartography":
                # prepare representations and data for DAL
                if i != 0:
                    X_train_rep = estimator.train(X_train, y_train)
                X_pool_rep = estimator.predict(X_pool)
                if args.acquisition == "discriminative":
                    X_train_dal, y_train_dal = prepare_data_for_dal(X_train_rep, X_pool_rep)
                    class_weights = get_distribution_weights(y_train_dal)
                    dal_estimator = DALEstimator(args, len(X_train_rep), X_train_rep[0].size, len(np.unique(y_train_dal)),
                                                 class_weights)
                    top_k_indices = dal_estimator.train(X_train_dal, y_train_dal)
                    dal_estimator.weight_reset()

                elif args.acquisition == "cartography":
                    X_train_cal, y_train_cal, X_pool_cal, y_pool_cal = prepare_data_for_cal(X_train_rep, X_pool_rep,
                                                                                            estimator.correctness)
                    class_weights = get_distribution_weights(y_train_cal)
                    cal_estimator = CALEstimator(args, len(X_train_cal), X_train_rep[0].size, len(np.unique(y_train_cal)),
                                                 class_weights)
                    #cal_estimator = BERTCALEstimator(args, model=model)
                    cal_estimator.train(X_train_cal, y_train_cal)
                    top_k_indices = cal_estimator.predict(X_pool_cal)
                    cal_estimator.weight_reset()

            else:
                # apply model to the pool to retrieve top-k instances
                probas = estimator.predict(X_pool)
                top_k_indices = apply_acquisition_function(args, probas)

            # add top-k instances from pool to train and remove from pool
            #X_train, y_train, X_pool= add_and_remove_instances(X_train, y_train, X_pool, top_k_indices)

            # retrain model, save accuracy, weight reset for next iter
            #estimator.train(X_train, y_train)
            accuracy = estimator.evaluate(X_test, y_test)
            active_learning_accuracy_history.append(accuracy)
            print(top_k_indices)
    #         for j in range(len(top_k_run)):
    #             idx_list = top_k_run[j]
    #             selected_top_k_dict[j].append([idx for idx in idx_list])

    #         with open(f"{os.getenv('INDICES_PATH')}/top_k_{args.task}_{args.acquisition}_{i}.json", "w") as f:
    #             json.dump(selected_top_k_dict, f)
            if args.analysis:
                selected_top_k.append(top_k_indices)

                if i != 0:
                    confidences = {idx: sum(proba) / len(proba) for idx, proba in list(estimator.probabilities.items())}
                    variability = {idx: np.std(proba) for idx, proba in list(estimator.probabilities.items())}
                    correctness = {idx: transform_correctness_to_bins(correct) for idx, correct in
                                   list(estimator.correctness.items())}
                    confidence_stats.append(
                        np.mean(list(confidences.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                    variability_stats.append(
                        np.mean(list(variability.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))
                    correctness_stats.append(
                        np.mean(list(correctness.values())[-int(os.getenv("ACTIVE_LEARNING_BATCHES")):]))

            reset_estimator(estimator)
            logger.info(f"Accuracy history: {active_learning_accuracy_history}")
        logger.info("Returning from active learning! good job")
        ##return active_learning_accuracy_history, selected_top_k, confidence_stats, variability_stats, correctness_stats
        return selected_top_k