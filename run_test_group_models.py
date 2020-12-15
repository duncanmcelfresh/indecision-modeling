# run group model-fitting experiments

from fit_preference_models import optimize_parameters, calculate_ll, OTHER_MODEL_NAMES
from utils import (
    get_logger,
    generate_filepath,
    read_queries,
)
from fit_preference_models import SCORE_MODEL_NAMES
import argparse
import os

import numpy as np

logger = get_logger()

DELIMITER = ";"


def experiment(args):

    # gather different models to run
    fit_model_list = []
    for name in args.model_name:
        if not name in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES:
            raise Exception(
                f"model name {name} not recognized. model names must be in {SCORE_MODEL_NAMES}."
            )
        fit_model_list.append(name)

    if fit_model_list == []:
        logger.info("no model names provided. running random baselines only")

    train_queries_by_user, test_queries, test_user_queries = read_queries(
        args.data_filename,
        args.num_training_users,
        args.seed,
        normalize_features=True,
        test_question_frac=args.test_question_frac,
    )

    # get a list of all training queries
    train_queries = []
    user_query_indices = []
    query_index = 0
    for user_queries in train_queries_by_user.values():
        user_query_indices.append(
            list(range(query_index, query_index + len(user_queries)))
        )
        train_queries.extend(user_queries)
        query_index = query_index + len(user_queries)

    # generate output file
    output_file = generate_filepath(
        args.output_dir, f"coin_flip_group_model_expt", "csv"
    )
    logger.info("generating output file: {}".format(output_file))

    # list of output column names
    col_list = [
        "model_name",
        "params",
        "train_ll",
        "test_ll",
        "test_user_ll",
        "overall_test_ll",
        "avg_train_ll",
        "avg_test_ll",
        "avg_test_user_ll",
        "avg_overall_test_ll",
    ]

    num_flips_train = len([q for q in train_queries if q.response == 0])
    num_flips_test = len([q for q in test_queries if q.response == 0])

    # # add some stuff to args (for bookkeeping)
    # info = {
    #     "num_train": len(train_queries),
    #     "num_flips_train": num_flips_train,
    #     "num_test": len(test_queries),
    #     "num_flips_test": num_flips_test,
    # }

    # write the file header: write experiment parameters to the first line of the output file
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((DELIMITER.join(col_list) + "\n"))

    logger.info(
        f"number training queries: {len(train_queries)} (# flip = {num_flips_train})"
    )
    logger.info(f"number test queries: {len(test_queries)} (# flip = {num_flips_test})")
    logger.info(f"number (entire population) test queries: {len(test_user_queries)}")

    # train each model
    for model_name in fit_model_list:

        logger.info(f"model: {model_name}")

        if model_name in ["voter_mixture", "uniform_voter_mixture"]:
            user_indices = user_query_indices
        else:
            user_indices = None

        model_ll, model_params = optimize_parameters(
            train_queries, model_name, args, user_query_indices=user_indices
        )

        # re-evaluate model parameters, just to be sure...
        train_ll = calculate_ll(train_queries, model_name, model_params)

        # calculate test ll on expert queries
        test_ll = calculate_ll(test_queries, model_name, model_params)

        # calculate test user ll
        test_user_ll = calculate_ll(test_user_queries, model_name, model_params)

        # calculate overall test ll
        overall_test_ll = calculate_ll(
            test_user_queries + test_queries, model_name, model_params
        )

        if model_name == "ann_classifier":
            model_params = str(model_params).replace("\n", " ")

        result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
            model_name,
            model_params,
            train_ll,
            test_ll,
            test_user_ll,
            overall_test_ll,
            train_ll / float(len(train_queries)),
            test_ll / float(len(test_queries)),
            test_user_ll / float(len(test_user_queries)),
            overall_test_ll / float(len(test_user_queries) + len(test_queries)),
        )

        with open(output_file, "a") as f:
            f.write(result_str)

    # use the uniform-random baseline
    # re-evaluate model parameters, just to be sure...
    model_name = "uniform_rand"
    fixed_probs = {1: 1.0 / 3.0, -1: 1.0 / 3.0, 0: 1.0 / 3.0}
    # train_ll = calculate_ll(train_queries, "fixed_rand", None, fixed_probs=fixed_probs)
    #
    # # calculate test ll
    # test_ll = calculate_ll(
    #     test_queries, "fixed_rand", None, fixed_probs=fixed_probs
    # )

    # re-evaluate model parameters, just to be sure...
    train_ll = calculate_ll(train_queries, "fixed_rand", None, fixed_probs=fixed_probs)

    # calculate test ll on expert queries
    test_ll = calculate_ll(test_queries, "fixed_rand", None, fixed_probs=fixed_probs)

    # calculate test user ll
    test_user_ll = calculate_ll(
        test_user_queries, "fixed_rand", None, fixed_probs=fixed_probs
    )

    # calculate overall test ll
    overall_test_ll = calculate_ll(
        test_user_queries + test_queries, "fixed_rand", None, fixed_probs=fixed_probs
    )

    result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
        model_name,
        fixed_probs,
        train_ll,
        test_ll,
        test_user_ll,
        overall_test_ll,
        train_ll / float(len(train_queries)),
        test_ll / float(len(test_queries)),
        test_user_ll / float(len(test_user_queries)),
        overall_test_ll / float(len(test_user_queries) + len(test_queries)),
    )

    with open(output_file, "a") as f:
        f.write(result_str)

    # use the naive random baseline
    # find the proportion of training responses for each
    n_r0 = len(list(filter(lambda q: q.response == 0, train_queries)))
    n_strict = len(train_queries) - n_r0

    # now naive random...
    model_name = "naive_rand"
    fixed_probs = {
        1: n_strict / (2.0 * len(train_queries)),
        -1: n_strict / (2.0 * len(train_queries)),
        0: n_r0 / len(train_queries),
    }

    # re-evaluate model parameters, just to be sure...
    train_ll = calculate_ll(train_queries, "fixed_rand", None, fixed_probs=fixed_probs)

    # calculate test ll on expert queries
    test_ll = calculate_ll(test_queries, "fixed_rand", None, fixed_probs=fixed_probs)

    # calculate test user ll
    test_user_ll = calculate_ll(
        test_user_queries, "fixed_rand", None, fixed_probs=fixed_probs
    )

    # calculate overall test ll
    overall_test_ll = calculate_ll(
        test_user_queries + test_queries, "fixed_rand", None, fixed_probs=fixed_probs
    )

    result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
        model_name,
        fixed_probs,
        train_ll,
        test_ll,
        test_user_ll,
        overall_test_ll,
        train_ll / float(len(train_queries)),
        test_ll / float(len(test_queries)),
        test_user_ll / float(len(test_user_queries)),
        overall_test_ll / float(len(test_user_queries) + len(test_queries)),
    )
    with open(output_file, "a") as f:
        f.write(result_str)


def get_parser():
    # parse some args
    parser = argparse.ArgumentParser()

    # experiment params
    parser.add_argument("--data-filename", type=str, help="data file", default="")
    parser.add_argument(
        "--output-dir", type=str, help="directory for writing output", default=""
    )

    # different models to fit
    parser.add_argument(
        "-m",
        "--model-name",
        type=str,
        action="append",
        help="list of model names to fit, subset of: {'max_diff', 'min_diff', 'max_like', 'min_like', 'dom', 'simple_logit', 'model_mixture'}",
        default=[],
    )
    parser.add_argument(
        "--k", type=int, help="number of models to include in the k-mixture", default=2,
    )

    parser.add_argument(
        "--num-gpei-points",
        type=int,
        help="number of GPEI samples during each optimization loop",
        default=5,
    )
    parser.add_argument(
        "--num-sobol-points",
        type=int,
        help="number of (random) Sobol samples during each optimization loop",
        default=30,
    )
    parser.add_argument(
        "--max-opt-loops",
        type=int,
        help="maximum number of training loops",
        default=10,
    )
    parser.add_argument(
        "--opt-time-limit",
        type=float,
        help="time limit for each optimization run, in seconds",
        default=60,
    )

    parser.add_argument(
        "--num-submodel-ax-trials",
        type=int,
        help="number of ax trials for each submodel (only for aggregation methods",
        default=35,
    )

    parser.add_argument(
        "--num-training-users",
        type=int,
        help="number of users to include in the training set. the rest go to testing",
        default=10,
    )
    parser.add_argument(
        "--test-question-frac",
        type=float,
        help="number of questions to use for testing, from each training user",
        default=0.2,
    )

    parser.add_argument(
        "--seed", type=int, help="random seed for train/test split", default=0,
    )
    parser.add_argument(
        "--DEBUG",
        action="store_true",
        help="if set, use a fixed arg string for debugging. otherwise, parse args.",
        default=False,
    )

    return parser


if __name__ == "__main__":

    parser = get_parser()

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:

        arg_str = "--data-filename " + os.path.join(
            os.getcwd(), "data", "clean-data-all-e.csv"
        )
        arg_str += " --output-dir " + os.path.join(os.getcwd(), "output")
        arg_str += " --num-training-users 20"
        arg_str += " --test-question-frac 0.5"

        # arg_str += " --max-opt-loops 1"
        # arg_str += " --num-sobol-points 10"
        # arg_str += " --num-gpei-points 1"
        arg_str += " --opt-time-limit 3600"
        # arg_str += " --k 2"
        # arg_str += " --seed 2"

        # arg_str += " -m k_mixture"
        # arg_str += " -m k_min_diff"
        arg_str += " -m ann_classifier"
        # arg_str += " -m simple_logit"
        # arg_str += " -m all"
        # arg_str += " -m uniform_voter_mixture"
        # arg_str += " -m max_diff"

        args = parser.parse_args(arg_str.split())

    experiment(args)
