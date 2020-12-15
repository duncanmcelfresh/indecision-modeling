# run group model-fitting experiments, using only strict responses

from fit_preference_models import optimize_parameters_strict, calculate_ll_strict
from utils import (
    get_logger,
    generate_filepath,
    read_queries_strict,
)
from fit_preference_models import SCORE_MODEL_NAMES
import argparse
import os

logger = get_logger()

DELIMITER = ";"


def experiment(args):

    # gather different models to run

    if "all" in args.model_name:
        fit_model_list = SCORE_MODEL_NAMES + ["simple_logit", "ann_classifier"]
    else:
        fit_model_list = []
        for name in args.model_name:
            if not name in SCORE_MODEL_NAMES + ["simple_logit", "ann_classifier"]:
                raise Exception("model name {name} not recognized")
            fit_model_list.append(name)

    if fit_model_list == []:
        logger.info("no model names provided. running random baselines only")

    train_queries_by_user, test_queries_by_user, _ = read_queries_strict(
        args.data_filename,
        args.num_training_users,  # get all users
        args.seed,
        normalize_features=True,
        test_question_frac=args.test_question_frac,
        test_queries_by_user=True,  # split test queries by user
    )

    assert train_queries_by_user.keys() == test_queries_by_user.keys()

    # generate output file
    output_file = generate_filepath(args.output_dir, f"strict_individual_expt", "csv")
    logger.info("generating output file: {}".format(output_file))

    # list of output column names
    col_list = [
        "user_id",
        "model_name",
        "params",
        "train_ll",
        "test_ll",
        "avg_train_ll",
        "avg_test_ll",
    ]

    # write the file header: write experiment parameters to the first line of the output file
    with open(output_file, "w") as f:
        f.write(str(args) + "\n")
        f.write((DELIMITER.join(col_list) + "\n"))

    for i, user_id in enumerate(test_queries_by_user.keys()):

        if i < args.start_user:
            continue

        logger.info(f"starting user: {user_id} ({i} of {len(test_queries_by_user)})")

        train_queries = train_queries_by_user[user_id]
        test_queries = test_queries_by_user[user_id]

        # train each model
        for model_name in fit_model_list:

            model_ll, model_params = optimize_parameters_strict(
                train_queries, model_name, args
            )

            # re-evaluate model parameters, just to be sure...
            train_ll = calculate_ll_strict(train_queries, model_name, model_params)

            # calculate test ll on expert queries
            test_ll = calculate_ll_strict(test_queries, model_name, model_params)

            if model_name == "ann_classifier":
                model_params = str(model_params).replace("\n", " ")

            result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
                user_id,
                model_name,
                model_params,
                train_ll,
                test_ll,
                train_ll / float(len(train_queries)),
                test_ll / float(len(test_queries)),
            )

            with open(output_file, "a") as f:
                f.write(result_str)

        # use the uniform-random baseline
        model_name = "uniform_rand"
        fixed_probs = {1: 0.5, -1: 0.5}

        # re-evaluate model parameters, just to be sure...
        train_ll = calculate_ll_strict(
            train_queries, "fixed_rand", None, fixed_probs=fixed_probs
        )

        # calculate test ll on expert queries
        test_ll = calculate_ll_strict(
            test_queries, "fixed_rand", None, fixed_probs=fixed_probs
        )

        result_str = (DELIMITER.join(len(col_list) * ["{}"]) + "\n").format(
            user_id,
            model_name,
            fixed_probs,
            train_ll,
            test_ll,
            train_ll / float(len(train_queries)),
            test_ll / float(len(test_queries)),
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
        "--start-user", type=int, help="user to start at", default=0,
    )

    parser.add_argument(
        "--num-training-users",
        type=int,
        help="number of users to include in the training set. the rest go to testing",
        default=3,
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

        arg_str = "--data-filename " + os.path.join(os.getcwd(), "data", "nocf.csv")
        arg_str += " --output-dir " + os.path.join(os.getcwd(), "output")
        arg_str += " --num-training-users -1"
        arg_str += " --test-question-frac 0.5"

        arg_str += " --max-opt-loops 1"
        arg_str += " --start-user 100"
        arg_str += " --num-sobol-points 100"
        arg_str += " --num-gpei-points 0"
        arg_str += " --opt-time-limit 60"
        arg_str += " --seed 0"

        arg_str += " -m all"

        args = parser.parse_args(arg_str.split())

    experiment(args)
