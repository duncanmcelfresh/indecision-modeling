# run individual model-fitting experiments

from fit_preference_models import optimize_parameters, calculate_ll
from utils import read_data_to_user_query_lists, get_logger, generate_filepath
from fit_preference_models import SCORE_MODEL_NAMES
import argparse
import os

import numpy as np

logger = get_logger()

DELIMITER = ";"


def experiment(args):
    """
    required args:
    - data_filename
    - output_dir
    - num_lam_points
    - num_u_points
    - num_test_samples
    - num_cv_folds
    - seed
    """

    rs = np.random.RandomState(args.seed)

    # gather different models to run
    fit_model_list = []
    for name in args.model_name:
        if not name in SCORE_MODEL_NAMES + ["simple_logit"]:
            raise Exception(
                f"model name {name} not recognized. model names must be in {SCORE_MODEL_NAMES}."
            )
        fit_model_list.append(name)

    if fit_model_list == []:
        logger.info("no model names provided. running random baselines only")

    user_query_dict = read_data_to_user_query_lists(
        args.data_filename, normalize_features=True
    )
    user_ids = list(user_query_dict.keys())

    # generate output file
    output_file = generate_filepath(args.output_dir, f"coin_flip_model_expt", "csv")
    logger.info("generating output file: {}".format(output_file))

    # list of output column names
    col_list = [
        "uid",
        "num_queries",
        "num_flips",
        "num_response_left",
        "num_response_right",
        "test_inds",
        "model_name",
        "train_ll",
        "u_vec",
        "lambda",
        "params",
        "test_ll",
    ]

    # write the file header: write experiment parameters to the first line of the output file
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((DELIMITER.join(col_list) + "\n"))

    for uid in user_ids:

        queries = user_query_dict[uid]
        num_flips = len([q for q in queries if q.response == 0])
        num_response_left = len([q for q in queries if q.response == 1])
        num_response_right = len([q for q in queries if q.response == -1])

        logger.info(f"user: {uid}")
        logger.info(f"number of answered queries: {len(queries)}")
        logger.info(
            f"number of coin flips: {len([q for q in queries if q.response == 0])}"
        )

        # randomly select test samples for each participant
        if args.num_test_samples > 0:
            test_inds = rs.choice(
                range(len(queries)), args.num_test_samples, replace=False
            )
            test_queries = [q for i, q in enumerate(queries) if i in test_inds]
            train_queries = [q for i, q in enumerate(queries) if i not in test_inds]
        else:
            test_inds = []
            test_queries = []
            train_queries = queries

        for model_name in fit_model_list:
            model_ll, model_params = optimize_parameters(
                train_queries, model_name, args
            )
            # re-evaluate model parameters, just to be sure...
            train_ll = calculate_ll(train_queries, model_name, model_params)

            # calculate test ll
            test_ll = 0
            if args.num_test_samples > 0:
                test_ll = calculate_ll(test_queries, model_name, model_params)

            if model_name == "simple_logit":
                lam = ""
                params = {"p_flip": model_params[-1]}
            else:
                lam = model_params[-1]
                params = ""

            result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
                uid,
                len(queries),
                num_flips,
                num_response_left,
                num_response_right,
                test_inds,
                model_name,
                train_ll,
                model_params[:-1],
                lam,
                params,
                test_ll,
            )

            with open(output_file, "a") as f:
                f.write(result_str)

        # use the uniform-random baseline
        # re-evaluate model parameters, just to be sure...
        model_name = "uniform_rand"
        fixed_probs = {1: 1.0 / 3.0, -1: 1.0 / 3.0, 0: 1.0 / 3.0}
        rand_train_ll = calculate_ll(
            train_queries, "fixed_rand", None, fixed_probs=fixed_probs
        )

        # calculate test ll
        test_ll = 0
        if args.num_test_samples > 0:
            rand_test_ll = calculate_ll(
                test_queries, "fixed_rand", None, fixed_probs=fixed_probs
            )

        result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
            uid,
            len(queries),
            num_flips,
            num_response_left,
            num_response_right,
            test_inds,
            model_name,
            rand_train_ll,
            "",
            "",
            fixed_probs,
            rand_test_ll,
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
        rand_train_ll = calculate_ll(
            train_queries, "fixed_rand", None, fixed_probs=fixed_probs
        )

        # calculate test ll
        test_ll = 0
        if args.num_test_samples > 0:
            rand_test_ll = calculate_ll(
                test_queries, "fixed_rand", None, fixed_probs=fixed_probs
            )

        result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
            uid,
            len(queries),
            num_flips,
            num_response_left,
            num_response_right,
            test_inds,
            model_name,
            rand_train_ll,
            "",
            "",
            fixed_probs,
            rand_test_ll,
        )
        with open(output_file, "a") as f:
            f.write(result_str)


if __name__ == "__main__":

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
        help="list of model names to fit, subset of: {'max_diff', 'min_diff', 'max_like', 'min_like', 'dom'}",
        default=[],
    )

    subparsers = parser.add_subparsers(help="optimization method", dest="optimizer")

    # use gridsearch to find model parameters
    gridsearch_parser = subparsers.add_parser("gridsearch")
    gridsearch_parser.add_argument(
        "--num-lam-points", type=int, help="num lambda points for each model", default=3
    )

    gridsearch_parser.add_argument(
        "--num-u-points",
        type=int,
        help="num lambda points for utility vec, each feature",
        default=3,
    )

    # use ax to find model parameters
    ax_parser = subparsers.add_parser("ax")
    ax_parser.add_argument(
        "--num-ax-trials", type=int, help="number of ax trials", default=50,
    )

    parser.add_argument(
        "--num-test-samples",
        type=int,
        help="number of answered queries to hold out for testing, for each individual",
        default=8,
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

    args = parser.parse_args()

    if args.DEBUG:
        # fixed set of parameters, for debugging:

        arg_str = "--data-filename " + os.path.join(
            os.getcwd(), "data", "clean-data-all-e.csv"
        )
        arg_str += " --output-dir " + os.path.join(os.getcwd(), "output")
        # arg_str += " -m simple_logit"
        arg_str += " ax"
        # arg_str += " gridsearch"
        # arg_str += " --num-lam-points 10"
        # arg_str += " --num-u-points 10"

        args = parser.parse_args(arg_str.split())

    experiment(args)
    #
    # data_filename = os.path.join(os.getcwd(), "data", "clean-data-all-e.csv")
    #
    # user_query_dict = read_data_to_user_query_lists(data_filename, normalize_features=True)
    #
    # user_ids = list(user_query_dict.keys())
    #
    # num_flips = {
    #     uid: len([q for q in user_query_dict[uid] if q.response == 0])
    #     for uid in user_ids
    # }
    #
    # final_user_ids = [u for u in user_ids if (num_flips[u] > 0) and (num_flips[u] < 40)]
    #
    # uid = final_user_ids[5]
    # queries = user_query_dict[uid]
    #
    # print(f"user: {uid}")
    # print(f"number of coin flips: {len([q for q in queries if q.response == 0])}")
    #
    # optimize_parameters(queries, "min_diff", num_u_points=10, num_lam_points=10)
    # optimize_parameters(queries, "max_diff", num_u_points=10, num_lam_points=10)
