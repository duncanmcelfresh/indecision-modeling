# helper functions for coin flip model fitting
import logging
import os
import time

from ax import Models, SimpleExperiment

from preference_classes import Query, Item
import pandas as pd
import numpy as np


def read_data_to_user_query_lists(filename, normalize_features=True):
    """read clean-data-all-e.csv and generate a list of answered queries for each user"""

    df = pd.read_csv(filename)

    # drop participants with 0 coin flips or 39 or more coin-flips
    flip_uids = (
        df[df["decision"] == 2].groupby(by="user_id")["decision"].count().reset_index()
    )
    keep_uids = list(flip_uids[flip_uids["decision"] <= 39]["user_id"])
    df = df.loc[df["user_id"].isin(keep_uids)]

    # keep only participants with seq_id = "self"
    df = df.loc[df["seq_id"] == "self"]

    # get each participants
    user_ids = df["user_id"].unique()

    user_query_dict = {}

    feat_1_max = 1.0
    feat_1_min = 0.0
    feat_2_max = 1.0
    feat_2_min = 0.0
    feat_3_max = 1.0
    feat_3_min = 0.0

    if normalize_features:
        feat_1_max = 70.0
        feat_1_min = 25.0
        feat_2_max = 5.0
        feat_2_min = 1.0
        feat_3_max = 2.0
        feat_3_min = 0.0

    # iterate through each user
    item_id = 0
    for uid in user_ids:
        df_user = df[df["user_id"] == uid]

        if len(df_user) < 30:
            print(
                "ignoring user %s: only %d observed responses (there must be at least 30)"
                % (uid, len(df_user))
            )
            continue

        # create answered queries from every row (every answered query)
        # in the table:
        #   decision = 0 indicates left > right,
        #   decision = 1 indicates left < right,
        #   decision = 2 indicates coin flip
        # we translate this to our convention (1 : A > B, -1 : A < B, 0 : A ~= B)
        answered_queries = []
        for i_row, row in df_user.iterrows():
            item_A = Item(
                [
                    (row["left_1"] - feat_1_min) / feat_1_max,
                    (row["left_2"] - feat_2_min) / feat_2_max,
                    (row["left_3"] - feat_3_min) / feat_3_max,
                ],
                item_id,
            )
            item_B = Item(
                [
                    (row["right_1"] - feat_1_min) / feat_1_max,
                    (row["right_2"] - feat_2_min) / feat_2_max,
                    (row["right_3"] - feat_3_min) / feat_3_max,
                ],
                item_id + 1,
            )

            item_id += 2

            if row["decision"] == 0:
                response = 1
            elif row["decision"] == 1:
                response = -1
            elif row["decision"] == 2:
                response = 0
            else:
                raise Warning("Invalid decision")

            answered_queries.append(Query(item_A, item_B, response=response))

        user_query_dict[uid] = answered_queries

    return user_query_dict


def read_queries(
    filename,
    num_training_users,
    seed,
    normalize_features=True,
    test_question_frac=0.2,
    test_queries_by_user=False,
):
    """
    read clean-data-all-e.csv and generate a list of training and testing queries

    num_training_users is the exact number of users in the training set (if -1, all users are training users)

    return values are: train_queries_by_user, test_queries, test_user_queries

    train_queries_by_user : list of lists of queries for each user
    test_queries : test queries for all training users
    test_user_queries : test queries for all test users

    """

    rs = np.random.RandomState(seed)
    df = pd.read_csv(filename)

    # drop individuals with 0 coin flips or 40 or more coin-flips
    flip_uids = (
        df[df["decision"] == 2].groupby(by="user_id")["decision"].count().reset_index()
    )
    keep_uids = list(flip_uids[~flip_uids["decision"].isin([0, 40])]["user_id"])
    df = df.loc[df["user_id"].isin(keep_uids)]

    # keep only participants with seq_id = "self"
    df = df.loc[df["seq_id"] == "self"]

    feat_1_max = 1.0
    feat_1_min = 0.0
    feat_2_max = 1.0
    feat_2_min = 0.0
    feat_3_max = 1.0
    feat_3_min = 0.0

    if normalize_features:
        feat_1_max = 70.0
        feat_1_min = 25.0
        feat_2_max = 5.0
        feat_2_min = 1.0
        feat_3_max = 2.0
        feat_3_min = 0.0

    # select the training users
    if num_training_users == -1:
        training_user_ids = df["user_id"].unique()
    else:
        assert num_training_users < len(df["user_id"].unique())
        training_user_ids = rs.choice(
            list(df["user_id"].unique()), num_training_users, replace=False
        )

    # split training user responses into train/test set, and include all non-training users in the test set

    # get the number of questions per user
    num_questions = df["decision_rank"].max()

    num_test = int(np.round(test_question_frac * num_questions))
    test_question_inds = rs.choice(num_questions, num_test, replace=False)

    if test_queries_by_user:
        test_queries_by_user = {uid: [] for uid in training_user_ids}
    else:
        test_queries = []
    train_queries = {uid: [] for uid in training_user_ids}
    test_user_queries = []
    item_id = 0
    for i_row, row in df.iterrows():
        item_A = Item(
            [
                (row["left_1"] - feat_1_min) / feat_1_max,
                (row["left_2"] - feat_2_min) / feat_2_max,
                (row["left_3"] - feat_3_min) / feat_3_max,
            ],
            item_id,
        )
        item_B = Item(
            [
                (row["right_1"] - feat_1_min) / feat_1_max,
                (row["right_2"] - feat_2_min) / feat_2_max,
                (row["right_3"] - feat_3_min) / feat_3_max,
            ],
            item_id + 1,
        )

        item_id += 2

        if row["decision"] == 0:
            response = 1
        elif row["decision"] == 1:
            response = -1
        elif row["decision"] == 2:
            response = 0
        else:
            raise Warning("Invalid decision")

        if row["user_id"] not in training_user_ids:
            # this is a test user
            test_user_queries.append(Query(item_A, item_B, response=response))
        else:
            # this is a training user
            if row["decision_rank"] in test_question_inds:
                # if this is a test question
                if test_queries_by_user:
                    test_queries_by_user[row["user_id"]].append(
                        Query(item_A, item_B, response=response)
                    )
                else:
                    test_queries.append(Query(item_A, item_B, response=response))
            else:
                # this is a train question
                train_queries[row["user_id"]].append(
                    Query(item_A, item_B, response=response)
                )

    if test_queries_by_user:
        return train_queries, test_queries_by_user, test_user_queries
    else:
        return train_queries, test_queries, test_user_queries


def read_queries_strict(
    filename,
    num_training_users,
    seed,
    normalize_features=True,
    test_question_frac=0.2,
    test_queries_by_user=False,
):
    """
    equivalent to read_queries but only for strict queries (no coinflip responses
    """

    rs = np.random.RandomState(seed)
    df = pd.read_csv(filename)

    feat_1_max = 1.0
    feat_1_min = 0.0
    feat_2_max = 1.0
    feat_2_min = 0.0
    feat_3_max = 1.0
    feat_3_min = 0.0

    if normalize_features:
        feat_1_max = 70.0
        feat_1_min = 25.0
        feat_2_max = 5.0
        feat_2_min = 1.0
        feat_3_max = 2.0
        feat_3_min = 0.0

    # select the training users
    if num_training_users == -1:
        training_user_ids = df["user_id"].unique()
    else:
        assert num_training_users < len(df["user_id"].unique())
        training_user_ids = rs.choice(
            list(df["user_id"].unique()), num_training_users, replace=False
        )

    # split training user responses into train/test set, and include all non-training users in the test set

    # get the number of questions per user
    num_questions = df["decision_rank"].max()

    num_test = int(np.round(test_question_frac * num_questions))
    test_question_inds = rs.choice(num_questions, num_test, replace=False)

    if test_queries_by_user:
        test_queries_by_user = {uid: [] for uid in training_user_ids}
    else:
        test_queries = []
    train_queries = {uid: [] for uid in training_user_ids}
    test_user_queries = []
    item_id = 0
    for i_row, row in df.iterrows():
        item_A = Item(
            [
                (row["left_1"] - feat_1_min) / feat_1_max,
                (row["left_2"] - feat_2_min) / feat_2_max,
                (row["left_3"] - feat_3_min) / feat_3_max,
            ],
            item_id,
        )
        item_B = Item(
            [
                (row["right_1"] - feat_1_min) / feat_1_max,
                (row["right_2"] - feat_2_min) / feat_2_max,
                (row["right_3"] - feat_3_min) / feat_3_max,
            ],
            item_id + 1,
        )

        item_id += 2

        if row["decision"] == 0:
            response = 1
        elif row["decision"] == 1:
            response = -1
        elif row["decision"] == 2:
            raise Exception("coin flip not expected in strict data")
        else:
            raise Exception("Invalid decision")

        if row["user_id"] not in training_user_ids:
            # this is a test user
            test_user_queries.append(Query(item_A, item_B, response=response))
        else:
            # this is a training user
            if row["decision_rank"] in test_question_inds:
                # if this is a test question
                if test_queries_by_user:
                    test_queries_by_user[row["user_id"]].append(
                        Query(item_A, item_B, response=response)
                    )
                else:
                    test_queries.append(Query(item_A, item_B, response=response))
            else:
                # this is a train question
                train_queries[row["user_id"]].append(
                    Query(item_A, item_B, response=response)
                )

    if test_queries_by_user:
        return train_queries, test_queries_by_user, test_user_queries
    else:
        return train_queries, test_queries, test_user_queries


def generate_random_point_nsphere(n, rs=None):
    # generate a random point on the n-dimensional sphere
    if rs is None:
        rs = np.random.RandomState(0)
    x = rs.normal(size=n)
    return x / np.linalg.norm(x, ord=2)


LOG_FORMAT = "[%(asctime)-15s] [%(filename)s:%(funcName)s] : %(message)s"


def get_logger(logfile=None):
    logger = logging.getLogger("experiment_logs")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FORMAT)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    else:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    return logger


logger = get_logger()


def generate_filepath(output_dir, name, extension):
    # generate filepath, of the format <name>_YYYYMMDD_HHMMDD<extension>
    timestr = time.strftime("%Y%m%d_%H%M%S")
    output_string = (name + "_%s." + extension) % timestr
    return os.path.join(output_dir, output_string)


class MyNamespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def custom_optimize(
    obj_func,
    search_space,
    minimize=False,
    time_limit=600,
    num_gpei_points=5,
    num_sobol_points=30,
    max_num_loops=100,
):
    """run an optimization loop using AX, with a time limit"""

    exp = SimpleExperiment(
        name="custom_optimize",
        search_space=search_space,
        evaluation_function=obj_func,
        objective_name="objective",
        minimize=minimize,
    )

    if minimize:
        raise NotImplemented()

    # start a training loop, each consisting of num_sobol_points sobol samples, and num_gpei_points gpei samples.
    # run this loop until the time limit is reached, or we have complete max_num_loops
    best_arm = None
    best_obj = -1e10
    done = False
    t0 = time.time()
    for i_loop in range(max_num_loops):
        sobol = Models.SOBOL(exp.search_space)
        for i in range(num_sobol_points):
            exp.new_trial(generator_run=sobol.gen(1))

        data = exp.eval()
        best_trial_name = data.df[data.df["mean"] == data.df["mean"].max()][
            "arm_name"
        ].values[0]
        tmp_obj = data.df["mean"].max()
        if tmp_obj > best_obj:
            best_arm = exp.arms_by_name[best_trial_name]
            best_obj = tmp_obj

        logger.info(
            f"finished sobol batch : best objective = {best_obj} (loop {i_loop})"
        )
        if time.time() - t0 > time_limit:
            break

        for i in range(num_gpei_points):

            gpei = Models.GPEI(experiment=exp, data=data)
            generator_run = gpei.gen(1)
            new_arm, obj = generator_run.best_arm_predictions
            new_obj = obj[0]["objective"]
            exp.new_trial(generator_run=generator_run)

            if new_obj > best_obj:
                best_arm = new_arm
                best_obj = new_obj

            logger.info(
                f"finished GPEI iteration {i} : best objective = {best_obj} (loop {i_loop})"
            )
            if time.time() - t0 > time_limit:
                done = True
                break

        logger.info(f"finished loop iteration {i_loop} : best objective = {best_obj}")

        if done:
            break

    return best_arm.parameters, best_obj
