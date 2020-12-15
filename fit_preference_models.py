# code for fitting coin-flip preference models
import itertools

import numpy as np
from ax import ParameterType, RangeParameter, SearchSpace, ChoiceParameter

from scipy.special import softmax
from sklearn.neural_network import MLPClassifier

from ax.service.managed_loop import optimize

from utils import get_logger, MyNamespace, custom_optimize

MIN_M = 0.0
MAX_M = 10.0

logger = get_logger()
SCORE_MODEL_NAMES = [
    "max_diff",
    "min_diff",
    "max_like",
    "min_like",
    "dom",
]

OTHER_MODEL_NAMES = [
    "simple_logit",
    # "model_mixture",
    "voter_mixture",
    "uniform_voter_mixture",
    "ann_classifier",
    "k_mixture",
    "k_min_diff",
]

BASELINE_MODEL_NAMES = [
    "fixed_rand",
]


EPS = 1e-10


def optimize_parameters(
    answered_queries, model_name, args, eps=0.2, user_query_indices=None,
):
    """
    search the parameter space of a specified model type, and return the model with max likelihood

    args:
    - answered_queries : list(Query). each query should have a response associated with it
        (see preference_classes.Query)
    - model_name : model type to fit
    - args: must have 'optimizer' subparser from run_test.py
    - optional: uid_question_indices : a list of lists - each sub-list contains all indices in answered_queries for each
        user

    valid model_name values includes anything in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES

    returns:
    - best_model: dict of form {"param_1_name": param_1_value, ..., "ll": log_likelihood}
    """

    if model_name not in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES:
        raise Exception("model not recognized")

    if any([q.response is None for q in answered_queries]):
        raise Exception("not all queries are answered")

    num_features = len(answered_queries[0].item_A.features)

    max_u = [1.0 for _ in range(num_features)]
    min_u = [-1.0 for _ in range(num_features)]

    # find max possible difference between item utility
    # iterate over all extreme points of feasible region for u to find max | u^T z | for all  queries z
    max_abs_diff = 0
    for q in answered_queries:
        z = q.z
        for u in itertools.product([0, 1], repeat=num_features):
            abs_diff = abs(np.dot(z, u))
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff

    # print(f"maximum |u^T z| for all queries is {max_abs_diff}")

    # find max possible item utility
    # iterate over all extreme points of feasible region for u to find max | u^T z | for all  queries z
    max_util = 0
    for q in answered_queries:
        for item in [q.item_A, q.item_B]:
            for u in itertools.product([0, 1], repeat=num_features):
                tmp_util = np.dot(item.features, u)
                if tmp_util > max_util:
                    max_util = tmp_util

    # print(f"maximum (u^T x) for all items is {max_util}")

    if model_name in SCORE_MODEL_NAMES:
        s_func, f_func, min_lam, max_lam = get_score_functions(
            model_name, max_util, max_abs_diff, eps, num_features
        )

    # partition queries by their response
    queries_r1 = list(filter(lambda q: q.response == 1, answered_queries))
    queries_r0 = list(filter(lambda q: q.response == 0, answered_queries))
    queries_rm1 = list(filter(lambda q: q.response == -1, answered_queries))

    if model_name == "simple_logit":

        # fit the simple logit model using ax
        def calc_ll(params):
            u_vec = [params["u1"], params["u2"], params["u3"]]
            p_flip = params["p_flip"]
            log_1_minus_pflip = np.log(1.0 - p_flip)
            log_pflip = np.log(p_flip)

            # report the total (not average) ll over all samples.

            ll_list = (
                [
                    log_1_minus_pflip - np.log(1 + np.exp(-np.dot(q.z, u_vec)))
                    for q in queries_r1
                ]
                + [
                    log_1_minus_pflip - np.log(1 + np.exp(np.dot(q.z, u_vec)))
                    for q in queries_rm1
                ]
                + [log_pflip] * len(queries_r0)
            )

            return {"objective": (np.sum(ll_list), 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[i],
                upper=max_u[i],
            )
            for i in range(3)
        ]

        p_flip = RangeParameter(
            name="p_flip",
            parameter_type=ParameterType.FLOAT,
            lower=EPS,
            upper=(1.0 - EPS),
        )

        search_space = SearchSpace(parameters=u_vec_params + [p_flip])

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        return best_ll, best_parameters

    elif model_name == "k_mixture":

        num_models = int(args.k)
        assert num_models > 0

        # now learn a mixture of these voter models, using regularized "importance" parameters
        # define a list of s_func and f_func handles
        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, eps, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # learn a mixture of models with a common u-vector,
        def calc_ll(params):

            # u vectors
            u_vec_list = [
                [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
                for k in range(num_models)
            ]

            # model thresholds
            lam_list = [params[f"lam_{k}"] for k in range(num_models)]

            # s/f functions
            s_func_list = [
                s_func_map[params[f"model_name_{k}"]] for k in range(num_models)
            ]
            f_func_list = [
                f_func_map[params[f"model_name_{k}"]] for k in range(num_models)
            ]

            # model importance
            m_list = [params[f"m_{k}"] for k in range(num_models)]
            m_list_exp = [np.exp(m) for m in m_list]

            # report the sum ll over all samples.

            ll_sum = (
                sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_rm1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(
                                f_func_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                )
                            )
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r0
                )
            ) - len(answered_queries) * np.log(sum(m_list_exp))

            return {"objective": (ll_sum, 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u_{k}_{i}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[0],
                upper=max_u[0],
            )
            for k, i in itertools.product(range(num_models), [1, 2, 3])
        ]

        lam_params = [
            RangeParameter(
                name=f"lam_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=(max(max_util, max_abs_diff) + 0.1),
            )
            for k in range(num_models)
        ]

        model_name_params = [
            ChoiceParameter(
                name=f"model_name_{k}",
                parameter_type=ParameterType.STRING,
                values=SCORE_MODEL_NAMES,
            )
            for k in range(num_models)
        ]

        importance_params = [
            RangeParameter(
                name=f"m_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=MIN_M,
                upper=MAX_M,
            )
            for k in range(num_models)
        ]

        search_space = SearchSpace(
            parameters=u_vec_params + lam_params + model_name_params + importance_params
        )

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        best_params = best_parameters
        best_params["k"] = num_models

        return best_ll, best_params

    elif model_name == "k_min_diff":

        num_models = int(args.k)
        assert num_models > 0

        # learn a mixture of min_diff models
        s_func, f_func, _, _ = get_score_functions("min_diff", 0, 0, eps, num_features)

        # learn a mixture of models with a common u-vector,
        def calc_ll(params):

            # u vectors
            u_vec_list = [
                [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
                for k in range(num_models)
            ]

            # model thresholds
            lam_list = [params[f"lam_{k}"] for k in range(num_models)]

            # model importance
            m_list = [params[f"m_{k}"] for k in range(num_models)]
            m_list_exp = [np.exp(m) for m in m_list]

            # report the sum ll over all samples.

            ll_sum = (
                sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_rm1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * np.exp(
                                f_func(q.item_A, q.item_B, u_vec_list[k], lam_list[k])
                            )
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r0
                )
            ) - len(answered_queries) * np.log(sum(m_list_exp))

            return {"objective": (ll_sum, 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u_{k}_{i}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[0],
                upper=max_u[0],
            )
            for k, i in itertools.product(range(num_models), [1, 2, 3])
        ]

        lam_params = [
            RangeParameter(
                name=f"lam_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=(max(max_util, max_abs_diff) + 0.1),
            )
            for k in range(num_models)
        ]

        importance_params = [
            RangeParameter(
                name=f"m_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=MIN_M,
                upper=MAX_M,
            )
            for k in range(num_models)
        ]

        search_space = SearchSpace(
            parameters=u_vec_params + lam_params + importance_params
        )

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        best_params = best_parameters
        best_params["k"] = num_models

        return best_ll, best_params

    elif model_name == "voter_mixture":
        # learn a individual model for each voter in the dataset (no train/test split),
        # by training a score-based model for each and selecting that with the best train ll

        raise NotImplemented("needs to be updated, to use custom_optimize()")
        #
        # logger.info(f"learning {len(user_query_indices)} voter models...")
        # # learn a model for each user
        #
        # tmp_args = MyNamespace(num_ax_trials=args.num_submodel_ax_trials,)
        #
        # user_models = []
        # for user_inds in user_query_indices:
        #     user_queries = [q for i, q in enumerate(answered_queries) if i in user_inds]
        #
        #     # learn one of each model class
        #     best_model = ""
        #     best_parameters = []
        #     best_ll = -1e10
        #     for m_name in SCORE_MODEL_NAMES:
        #         model_ll, model_params = optimize_parameters(
        #             user_queries, m_name, tmp_args
        #         )
        #         # re-evaluate model parameters, just to be sure...
        #         new_ll = calculate_ll(user_queries, m_name, model_params)
        #         if new_ll > best_ll:
        #             best_model = m_name
        #             best_parameters = model_params
        #             best_ll = new_ll
        #
        #     user_models.append(
        #         {"model_name": best_model, "parameters": best_parameters}
        #     )
        #
        # # get a list of all params
        # model_name_list = [user["model_name"] for user in user_models]
        # u_vec_list = [user["parameters"][:-1] for user in user_models]
        # lam_list = [user["parameters"][-1] for user in user_models]
        #
        # # now learn a mixture of these voter models, using regularized "importance" parameters
        # # define a list of s_func and f_func handles
        # s_func_map = {}
        # f_func_map = {}
        # for m_name in SCORE_MODEL_NAMES:
        #     s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, eps, num_features)
        #     s_func_map[m_name] = s_func
        #     f_func_map[m_name] = f_func
        #
        # # get a list of all function handles
        # s_func_user_list = [s_func_map[m_name] for m_name in model_name_list]
        # f_func_user_list = [f_func_map[m_name] for m_name in model_name_list]
        #
        # # magnitude of l2 regularization
        # alpha = 0.1
        #
        # # learn a mixture of (fixed) agent models
        # def calc_ll(params):
        #     # agent importance
        #     m_list = [params[f"m_{i}"] for i in range(len(user_models))]
        #     m_softmax = softmax(m_list)
        #
        #     # report the sum ll over all samples.
        #
        #     ll_list = (
        #         [
        #             np.log(
        #                 sum(
        #                     m_softmax[i]
        #                     * np.exp(
        #                         s_func_user_list[i](q.item_A, q.item_B, u_vec_list[i])
        #                     )
        #                     / (
        #                         np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_B, q.item_A, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             f_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i], lam_list[i],
        #                             )
        #                         )
        #                     )
        #                     for i in range(len(user_models))
        #                 )
        #             )
        #             for q in queries_r1
        #         ]
        #         + [
        #             np.log(
        #                 sum(
        #                     m_softmax[i]
        #                     * np.exp(
        #                         s_func_user_list[i](q.item_B, q.item_A, u_vec_list[i])
        #                     )
        #                     / (
        #                         np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_B, q.item_A, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             f_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i], lam_list[i],
        #                             )
        #                         )
        #                     )
        #                     for i in range(len(user_models))
        #                 )
        #             )
        #             for q in queries_rm1
        #         ]
        #         + [
        #             np.log(
        #                 sum(
        #                     m_softmax[i]
        #                     * np.exp(
        #                         f_func_user_list[i](
        #                             q.item_A, q.item_B, u_vec_list[i], lam_list[i]
        #                         )
        #                     )
        #                     / (
        #                         np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             s_func_user_list[i](
        #                                 q.item_B, q.item_A, u_vec_list[i]
        #                             )
        #                         )
        #                         + np.exp(
        #                             f_func_user_list[i](
        #                                 q.item_A, q.item_B, u_vec_list[i], lam_list[i],
        #                             )
        #                         )
        #                     )
        #                     for i in range(len(user_models))
        #                 )
        #             )
        #             for q in queries_r0
        #         ]
        #     )
        #
        #     return {"ll": (np.sum(ll_list) - alpha * np.linalg.norm(m_list), 0,)}
        #
        # parameter_dict_list = [
        #     {"name": f"m_{i}", "type": "range", "bounds": [MIN_M, MAX_M]}
        #     for i in range(len(user_models))
        # ]
        #
        # best_parameters, values, experiment, model = optimize(
        #     parameters=parameter_dict_list,
        #     objective_name="ll",
        #     total_trials=args.num_ax_trials,
        #     evaluation_function=calc_ll,
        #     minimize=False,
        # )
        #
        # best_ll = values[0]["ll"]
        # best_params = {
        #     "m_list": [best_parameters[f"m_{i}"] for i in range(len(user_models))],
        #     "voter_model_names": model_name_list,
        #     "u_vec_list": u_vec_list,
        #     "lam_list": lam_list,
        # }
        # return best_ll, best_params

    elif model_name == "uniform_voter_mixture":
        # learn a individual model for each voter in the dataset (no train/test split),
        # by training a score-based model for each and selecting that with the best train ll

        logger.info(f"learning {len(user_query_indices)} voter models...")
        # learn a model for each user

        model_name_list = []
        u_vec_list = []
        lam_list = []

        for user_inds in user_query_indices:
            user_queries = [q for i, q in enumerate(answered_queries) if i in user_inds]

            # learn one of each model class
            best_model = ""
            best_parameters = []
            best_ll = -1e10
            for m_name in SCORE_MODEL_NAMES:
                model_ll, model_params = optimize_parameters(user_queries, m_name, args)
                # re-evaluate model parameters, just to be sure...
                new_ll = calculate_ll(user_queries, m_name, model_params)
                if new_ll > best_ll:
                    best_model = m_name
                    best_parameters = model_params
                    best_ll = new_ll

            model_name_list.append(best_model)
            u_vec_list.append(
                [best_parameters["u1"], best_parameters["u2"], best_parameters["u3"]]
            )
            lam_list.append(best_parameters["lam"])

        params = {
            "voter_model_names": model_name_list,
            "u_vec_list": u_vec_list,
            "lam_list": lam_list,
        }

        ll = calculate_ll(answered_queries, model_name, params)

        return ll, params

    elif model_name == "ann_classifier":

        # fit an ann classifier
        clf = MLPClassifier(
            solver="lbfgs",
            alpha=0.1,
            hidden_layer_sizes=(32, 16),
            random_state=1,
            max_iter=5000,
        )

        # get training data
        X = [q.z for q in answered_queries]
        y = [str(q.response) for q in answered_queries]

        clf.fit(X, y)

        # calculate log-likelihood using our functions...
        best_ll = get_ann_log_likelihood(clf, answered_queries)

        return best_ll, clf

    else:
        # score/threshold-based rule
        def calc_ll(params):
            u_vec = [params["u1"], params["u2"], params["u3"]]
            lam = params["lam"]

            # report the sum ll over all samples.

            ll_list = (
                [
                    s_func(q.item_A, q.item_B, u_vec)
                    - np.log(
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    for q in queries_r1
                ]
                + [
                    s_func(q.item_B, q.item_A, u_vec)
                    - np.log(
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    for q in queries_rm1
                ]
                + [
                    f_func(q.item_A, q.item_B, u_vec, lam)
                    - np.log(
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    for q in queries_r0
                ]
            )

            return {
                "objective": (
                    np.sum(ll_list),
                    0,
                )  # np.std(ll_list) / np.sqrt(total_num_queries))
            }

        u_params = [
            RangeParameter(
                name=f"u{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[i],
                upper=max_u[i],
            )
            for i in range(3)
        ]

        lam_param = RangeParameter(
            name=f"lam",
            parameter_type=ParameterType.FLOAT,
            lower=min_lam,
            upper=max_lam,
        )

        search_space = SearchSpace(parameters=u_params + [lam_param])

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        return best_ll, best_parameters


def optimize_parameters_strict(
    answered_queries, model_name, args, eps=0.2, user_query_indices=None,
):
    """
    identical to optimize_parameters, but for strict (no coinflip) responses only
    """

    if model_name not in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES:
        raise Exception("model not recognized")

    if any([q.response is None for q in answered_queries]):
        raise Exception("not all queries are answered")

    num_features = len(answered_queries[0].item_A.features)

    max_u = [1.0 for _ in range(num_features)]
    min_u = [-1.0 for _ in range(num_features)]

    # find max possible difference between item utility
    # iterate over all extreme points of feasible region for u to find max | u^T z | for all  queries z
    max_abs_diff = 0
    for q in answered_queries:
        z = q.z
        for u in itertools.product([0, 1], repeat=num_features):
            abs_diff = abs(np.dot(z, u))
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff

    # print(f"maximum |u^T z| for all queries is {max_abs_diff}")

    # find max possible item utility
    # iterate over all extreme points of feasible region for u to find max | u^T z | for all  queries z
    max_util = 0
    for q in answered_queries:
        for item in [q.item_A, q.item_B]:
            for u in itertools.product([0, 1], repeat=num_features):
                tmp_util = np.dot(item.features, u)
                if tmp_util > max_util:
                    max_util = tmp_util

    # print(f"maximum (u^T x) for all items is {max_util}")

    if model_name in SCORE_MODEL_NAMES:
        s_func, f_func, min_lam, max_lam = get_score_functions(
            model_name, max_util, max_abs_diff, eps, num_features
        )

    # partition queries by their response (strict only)
    queries_r1 = list(filter(lambda q: q.response == 1, answered_queries))
    queries_rm1 = list(filter(lambda q: q.response == -1, answered_queries))
    if len(queries_r1) + len(queries_rm1) != len(answered_queries):
        raise Exception("only strict responses can be used")

    if model_name == "simple_logit":

        # fit the simple logit model using ax
        def calc_ll(params):
            u_vec = [params["u1"], params["u2"], params["u3"]]

            # report the total (not average) ll over all samples.
            ll = -sum(
                np.log(1 + np.exp(-np.dot(q.z, u_vec))) for q in queries_r1
            ) - sum(np.log(1 + np.exp(np.dot(q.z, u_vec))) for q in queries_rm1)

            return {"objective": (ll, 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[i],
                upper=max_u[i],
            )
            for i in range(3)
        ]

        search_space = SearchSpace(parameters=u_vec_params)

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        return best_ll, best_parameters

    elif model_name == "k_mixture":

        num_models = int(args.k)
        assert num_models > 0

        # now learn a mixture of these voter models, using regularized "importance" parameters
        # define a list of s_func and f_func handles
        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, eps, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # learn a mixture of models with a common u-vector,
        def calc_ll(params):

            # u vectors
            u_vec_list = [
                [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
                for k in range(num_models)
            ]

            # model thresholds
            lam_list = [params[f"lam_{k}"] for k in range(num_models)]

            # s/f functions
            s_func_list = [
                s_func_map[params[f"model_name_{k}"]] for k in range(num_models)
            ]
            f_func_list = [
                f_func_map[params[f"model_name_{k}"]] for k in range(num_models)
            ]

            q_param_list = [params[f"q_{k}"] for k in range(num_models)]

            # model importance
            m_list = [params[f"m_{k}"] for k in range(num_models)]
            m_list_exp = [np.exp(m) for m in m_list]

            # report the sum ll over all samples.

            ll_sum = (
                sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * (
                                q_param_list[k]
                                * (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_A, q.item_B, u_vec_list[k]
                                        )
                                    )
                                    + 0.5
                                    * np.exp(
                                        f_func_list[k](
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                / (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_A, q.item_B, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        s_func_list[k](
                                            q.item_B, q.item_A, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        f_func_list[k](
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                + (1.0 - q_param_list[k])
                                * np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                / (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_A, q.item_B, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        s_func_list[k](
                                            q.item_B, q.item_A, u_vec_list[k]
                                        )
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * (
                                q_param_list[k]
                                * (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_B, q.item_A, u_vec_list[k]
                                        )
                                    )
                                    + 0.5
                                    * np.exp(
                                        f_func_list[k](
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                / (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_A, q.item_B, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        s_func_list[k](
                                            q.item_B, q.item_A, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        f_func_list[k](
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                + (1.0 - q_param_list[k])
                                * np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                / (
                                    np.exp(
                                        s_func_list[k](
                                            q.item_A, q.item_B, u_vec_list[k]
                                        )
                                    )
                                    + np.exp(
                                        s_func_list[k](
                                            q.item_B, q.item_A, u_vec_list[k]
                                        )
                                    )
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_rm1
                )
                - len(answered_queries) * np.log(sum(m_list_exp))
            )

            return {"objective": (ll_sum, 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u_{k}_{i}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[0],
                upper=max_u[0],
            )
            for k, i in itertools.product(range(num_models), [1, 2, 3])
        ]

        lam_params = [
            RangeParameter(
                name=f"lam_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=(max(max_util, max_abs_diff) + 0.1),
            )
            for k in range(num_models)
        ]

        model_name_params = [
            ChoiceParameter(
                name=f"model_name_{k}",
                parameter_type=ParameterType.STRING,
                values=SCORE_MODEL_NAMES,
            )
            for k in range(num_models)
        ]

        importance_params = [
            RangeParameter(
                name=f"m_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=MIN_M,
                upper=MAX_M,
            )
            for k in range(num_models)
        ]

        q_params = [
            RangeParameter(
                name=f"q_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=EPS,
                upper=(1.0 - EPS),
            )
            for k in range(num_models)
        ]

        search_space = SearchSpace(
            parameters=u_vec_params + lam_params + model_name_params + importance_params + q_params
        )

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        best_params = best_parameters
        best_params["k"] = num_models

        return best_ll, best_params

    elif model_name == "k_min_diff":

        num_models = int(args.k)
        assert num_models > 0

        # learn a mixture of min_diff models
        s_func, f_func, _, _ = get_score_functions("min_diff", 0, 0, eps, num_features)

        # learn a mixture of models with a common u-vector,
        def calc_ll(params):

            # u vectors
            u_vec_list = [
                [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
                for k in range(num_models)
            ]

            # model thresholds
            lam_list = [params[f"lam_{k}"] for k in range(num_models)]

            q_param_list = [params[f"q_{k}"] for k in range(num_models)]

            # model importance
            m_list = [params[f"m_{k}"] for k in range(num_models)]
            m_list_exp = [np.exp(m) for m in m_list]

            # report the sum ll over all samples.

            ll_sum = (
                sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * (
                                q_param_list[k]
                                * (
                                    np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                    + 0.5
                                    * np.exp(
                                        f_func(
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                / (
                                    np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                    + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                    + np.exp(
                                        f_func(
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                + (1.0 - q_param_list[k])
                                * np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                / (
                                    np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                    + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_r1
                )
                + sum(
                    np.log(
                        sum(
                            m_list_exp[k]
                            * (
                                q_param_list[k]
                                * (
                                    np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                    + 0.5
                                    * np.exp(
                                        f_func(
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                / (
                                    np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                    + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                    + np.exp(
                                        f_func(
                                            q.item_A,
                                            q.item_B,
                                            u_vec_list[k],
                                            lam_list[k],
                                        )
                                    )
                                )
                                + (1.0 - q_param_list[k])
                                * np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                / (
                                    np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                    + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                )
                            )
                            for k in range(num_models)
                        )
                    )
                    for q in queries_rm1
                )
                - len(answered_queries) * np.log(sum(m_list_exp))
            )

            return {"objective": (ll_sum, 0,)}

        u_vec_params = [
            RangeParameter(
                name=f"u_{k}_{i}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[0],
                upper=max_u[0],
            )
            for k, i in itertools.product(range(num_models), [1, 2, 3])
        ]

        lam_params = [
            RangeParameter(
                name=f"lam_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=0.0,
                upper=(max(max_util, max_abs_diff) + 0.1),
            )
            for k in range(num_models)
        ]

        q_params = [
            RangeParameter(
                name=f"q_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=EPS,
                upper=(1.0 - EPS),
            )
            for k in range(num_models)
        ]

        importance_params = [
            RangeParameter(
                name=f"m_{k}",
                parameter_type=ParameterType.FLOAT,
                lower=MIN_M,
                upper=MAX_M,
            )
            for k in range(num_models)
        ]

        search_space = SearchSpace(
            parameters=u_vec_params + lam_params + importance_params + q_params
        )

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        best_params = best_parameters
        best_params["k"] = num_models

        return best_ll, best_params

    elif model_name == "voter_mixture":
        # learn a individual model for each voter in the dataset (no train/test split),
        # by training a score-based model for each and selecting that with the best train ll

        raise NotImplemented()

    elif model_name == "uniform_voter_mixture":
        # learn a individual model for each voter in the dataset (no train/test split),
        # by training a score-based model for each and selecting that with the best train ll

        logger.info(f"learning {len(user_query_indices)} voter models...")
        # learn a model for each user

        model_name_list = []
        u_vec_list = []
        lam_list = []
        q_param_list = []

        for user_inds in user_query_indices:
            user_queries = [q for i, q in enumerate(answered_queries) if i in user_inds]

            # learn one of each model class
            best_model = ""
            best_parameters = []
            best_ll = -1e10
            for m_name in SCORE_MODEL_NAMES:
                model_ll, model_params = optimize_parameters_strict(
                    user_queries, m_name, args
                )
                # re-evaluate model parameters, just to be sure...
                new_ll = calculate_ll_strict(user_queries, m_name, model_params)
                if new_ll > best_ll:
                    best_model = m_name
                    best_parameters = model_params
                    best_ll = new_ll

            model_name_list.append(best_model)
            u_vec_list.append(
                [best_parameters["u1"], best_parameters["u2"], best_parameters["u3"]]
            )
            lam_list.append(best_parameters["lam"])
            q_param_list.append(best_parameters["q"])

        params = {
            "voter_model_names": model_name_list,
            "u_vec_list": u_vec_list,
            "lam_list": lam_list,
            "q_param_list": q_param_list,
        }

        ll = calculate_ll_strict(answered_queries, model_name, params)

        return ll, params

    elif model_name == "ann_classifier":

        # fit an ann classifier
        clf = MLPClassifier(
            solver="lbfgs",
            alpha=0.1,
            hidden_layer_sizes=(32, 16),
            random_state=1,
            max_iter=5000,
        )

        # get training data
        X = [q.z for q in answered_queries]
        y = [str(q.response) for q in answered_queries]

        clf.fit(X, y)

        # calculate log-likelihood using our functions...
        best_ll = get_ann_log_likelihood_strict(clf, answered_queries)

        return best_ll, clf

    else:
        # score/threshold-based rule
        def calc_ll(params):
            u_vec = [params["u1"], params["u2"], params["u3"]]
            lam = params["lam"]
            q_param = params["q"]

            # report the sum ll over all samples.

            ll = sum(
                np.log(
                    q_param
                    * (
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + 0.5 * np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    / (
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    + (1.0 - q_param)
                    * np.exp(s_func(q.item_A, q.item_B, u_vec))
                    / (
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                    )
                )
                for q in queries_r1
            ) + sum(
                np.log(
                    q_param
                    * (
                        np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + 0.5 * np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    / (
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                        + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                    )
                    + (1.0 - q_param)
                    * np.exp(s_func(q.item_B, q.item_A, u_vec))
                    / (
                        np.exp(s_func(q.item_A, q.item_B, u_vec))
                        + np.exp(s_func(q.item_B, q.item_A, u_vec))
                    )
                )
                for q in queries_rm1
            )

            return {"objective": (ll, 0,)}

        u_params = [
            RangeParameter(
                name=f"u{i+1}",
                parameter_type=ParameterType.FLOAT,
                lower=min_u[i],
                upper=max_u[i],
            )
            for i in range(3)
        ]

        lam_param = RangeParameter(
            name=f"lam",
            parameter_type=ParameterType.FLOAT,
            lower=min_lam,
            upper=max_lam,
        )

        q_param = RangeParameter(
            name=f"q", parameter_type=ParameterType.FLOAT, lower=EPS, upper=(1.0 - EPS),
        )

        search_space = SearchSpace(parameters=u_params + [lam_param, q_param])

        best_parameters, best_ll = custom_optimize(
            calc_ll,
            search_space,
            minimize=False,
            time_limit=args.opt_time_limit,
            num_gpei_points=args.num_gpei_points,
            num_sobol_points=args.num_sobol_points,
            max_num_loops=args.max_opt_loops,
        )

        return best_ll, best_parameters


# ----------------------------------------------------------------------------------------------------------------------
#  helper functions
# ----------------------------------------------------------------------------------------------------------------------


def get_score_functions(model_name, max_util, max_abs_diff, eps, num_features):
    """
    s_func and f_func are function handles which take Item objects, a u vector (list), and lam (float), and
    return a score (float)
    """

    if model_name not in SCORE_MODEL_NAMES:
        raise Exception("invalid model_name")

    if model_name == "min_diff":

        # define a range of lambda values
        min_lam = 0.0
        max_lam = max_abs_diff + eps

        def s_func(item_A, item_B, u):
            return np.dot(u, item_A.features - item_B.features)

        def f_func(item_A, item_B, u, lam):
            return lam

    if model_name == "max_diff":

        # define a range of lambda values
        min_lam = 0.0
        max_lam = max_abs_diff + eps

        def s_func(item_A, item_B, u):
            return np.dot(u, item_A.features - item_B.features)

        def f_func(item_A, item_B, u, lam):
            return (
                2 * abs(np.dot(u, item_A.features) - np.dot(u, item_B.features)) - lam
            )

    if model_name == "min_like":

        # define a range of lambda values
        min_lam = - max_util - eps
        max_lam = max_util + eps

        def s_func(item_A, item_B, u):
            return np.dot(u, item_A.features)

        def f_func(item_A, item_B, u, lam):
            return lam

    if model_name == "max_like":

        # define a range of lambda values
        min_lam = - max_util - eps
        max_lam = max_util + eps

        def s_func(item_A, item_B, u):
            return np.dot(u, item_A.features)

        def f_func(item_A, item_B, u, lam):
            return (
                2 * min([np.dot(u, item_A.features), np.dot(u, item_B.features)]) - lam
            )

    if model_name == "dom":

        # define a range of lambda values
        min_lam = - max_abs_diff - eps
        max_lam = max_abs_diff + eps

        def s_func(item_A, item_B, u):
            return min(
                u[i] * (item_A.features[i] - item_B.features[i])
                for i in range(num_features)
            )

        def f_func(item_A, item_B, u, lam):
            return lam

    return s_func, f_func, min_lam, max_lam


def calculate_ll(queries, model_name, params, fixed_probs=None):

    num_features = len(queries[0].item_A.features)

    if model_name not in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES + BASELINE_MODEL_NAMES:
        raise Exception("model not recognized")

    if model_name == "fixed_rand":
        # evaluate a randomized baseline, with fixed probabilities for each response type
        # in this case, fixed_probs must be a list s.t. fixed_probs[i] is the prob of response i
        assert fixed_probs is not None
        assert len(fixed_probs) == 3
        if not np.isclose(fixed_probs[0] + fixed_probs[-1] + fixed_probs[1], 1.0):
            raise Exception("fixed_probs must sum to 1")

        # number of each responses
        num_r1 = len(list(filter(lambda q: q.response == 1, queries)))
        num_rm1 = len(list(filter(lambda q: q.response == -1, queries)))
        num_r0 = len(list(filter(lambda q: q.response == 0, queries)))
        assert num_r1 + num_rm1 + num_r0 == len(queries)

        return (
            num_r1 * np.log(fixed_probs[1])
            + num_rm1 * np.log(fixed_probs[-1])
            + num_r0 * np.log(fixed_probs[0])
        )

    elif model_name == "simple_logit":
        # params are u [vec] + [p_flip]
        u_vec = [params["u1"], params["u2"], params["u3"]]
        p_flip = params["p_flip"]

        # report the total (not average) ll over all samples.

        return (
            sum(
                np.log(1.0 - p_flip) - np.log(1 + np.exp(-np.dot(q.z, u_vec)))
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(1.0 - p_flip) - np.log(1 + np.exp(np.dot(q.z, u_vec)))
                for q in queries
                if q.response == -1
            )
            + np.log(p_flip) * len([q for q in queries if q.response == 0])
        )

    elif model_name == "k_mixture":

        num_models = int(params["k"])
        assert num_models > 0

        # now learn a mixture of these voter models, using regularized "importance" parameters
        # define a list of s_func and f_func handles
        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, 0, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # u vectors
        u_vec_list = [
            [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
            for k in range(num_models)
        ]

        # model thresholds
        lam_list = [params[f"lam_{k}"] for k in range(num_models)]

        # s/f functions
        s_func_list = [s_func_map[params[f"model_name_{k}"]] for k in range(num_models)]
        f_func_list = [f_func_map[params[f"model_name_{k}"]] for k in range(num_models)]

        # model importance
        m_list = [params[f"m_{k}"] for k in range(num_models)]
        m_list_exp = [np.exp(m) for m in m_list]

        # report the sum ll over all samples.

        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                        / (
                            np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                        / (
                            np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == -1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(
                            f_func_list[k](
                                q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                            )
                        )
                        / (
                            np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 0
            )
        ) - len(queries) * np.log(sum(m_list_exp))

    elif model_name == "k_min_diff":

        # mixture of k min-diff models
        num_models = params["k"]

        s_func, f_func, _, _ = get_score_functions("min_diff", 0, 0, 0, num_features)

        # u vectors
        u_vec_list = [
            [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
            for k in range(num_models)
        ]

        # model thresholds
        lam_list = [params[f"lam_{k}"] for k in range(num_models)]

        # model importance
        m_list = [params[f"m_{k}"] for k in range(num_models)]
        m_list_exp = [np.exp(m) for m in m_list]

        # report the sum ll over all samples.
        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                        / (
                            np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func(q.item_A, q.item_B, u_vec_list[k], lam_list[k],)
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                        / (
                            np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func(q.item_A, q.item_B, u_vec_list[k], lam_list[k],)
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == -1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(f_func(q.item_A, q.item_B, u_vec_list[k], lam_list[k]))
                        / (
                            np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                            + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            + np.exp(
                                f_func(q.item_A, q.item_B, u_vec_list[k], lam_list[k],)
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 0
            )
        ) - len(queries) * np.log(sum(m_list_exp))

    elif model_name == "voter_mixture":

        # define a list of s_func and f_func handles
        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, 0, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        m_softmax = softmax(params["m_list"])
        model_name_list = params["voter_model_names"]
        u_vec_list = params["u_vec_list"]
        lam_list = params["lam_list"]
        num_users = len(m_softmax)

        # get a list of all function handles
        s_func_user_list = [s_func_map[m_name] for m_name in model_name_list]
        f_func_user_list = [f_func_map[m_name] for m_name in model_name_list]

        # learn a mixture of models with a common u-vector,
        return (
            sum(
                np.log(
                    sum(
                        m_softmax[i]
                        * np.exp(s_func_user_list[i](q.item_A, q.item_B, u_vec_list[i]))
                        / (
                            np.exp(
                                s_func_user_list[i](q.item_A, q.item_B, u_vec_list[i])
                            )
                            + np.exp(
                                s_func_user_list[i](q.item_B, q.item_A, u_vec_list[i])
                            )
                            + np.exp(
                                f_func_user_list[i](
                                    q.item_A, q.item_B, u_vec_list[i], lam_list[i],
                                )
                            )
                        )
                        for i in range(num_users)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_softmax[i]
                        * np.exp(s_func_user_list[i](q.item_B, q.item_A, u_vec_list[i]))
                        / (
                            np.exp(
                                s_func_user_list[i](q.item_A, q.item_B, u_vec_list[i])
                            )
                            + np.exp(
                                s_func_user_list[i](q.item_B, q.item_A, u_vec_list[i])
                            )
                            + np.exp(
                                f_func_user_list[i](
                                    q.item_A, q.item_B, u_vec_list[i], lam_list[i],
                                )
                            )
                        )
                        for i in range(num_users)
                    )
                )
                for q in queries
                if q.response == -1
            )
            + sum(
                np.log(
                    sum(
                        m_softmax[i]
                        * np.exp(
                            f_func_user_list[i](
                                q.item_A, q.item_B, u_vec_list[i], lam_list[i]
                            )
                        )
                        / (
                            np.exp(
                                s_func_user_list[i](q.item_A, q.item_B, u_vec_list[i])
                            )
                            + np.exp(
                                s_func_user_list[i](q.item_B, q.item_A, u_vec_list[i])
                            )
                            + np.exp(
                                f_func_user_list[i](
                                    q.item_A, q.item_B, u_vec_list[i], lam_list[i],
                                )
                            )
                        )
                        for i in range(num_users)
                    )
                )
                for q in queries
                if q.response == 0
            )
        )

    elif model_name == "uniform_voter_mixture":
        # params = {
        #     "voter_model_names": model_name_list,
        #     "u_vec_list": u_vec_list,
        #     "lam_list": lam_list,
        # }
        # report the sum ll over all samples.

        num_voters = len(params["voter_model_names"])

        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, 0, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # get a list of all function handles
        s_func_user_list = [
            s_func_map[m_name] for m_name in params["voter_model_names"]
        ]
        f_func_user_list = [
            f_func_map[m_name] for m_name in params["voter_model_names"]
        ]

        u_vec_list = params["u_vec_list"]
        lam_list = params["lam_list"]

        m_list = [1.0] * num_voters
        m_list_exp = [np.exp(m) for m in m_list]

        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func_user_list[k](q.item_A, q.item_B, u_vec_list[k]))
                        / (
                            np.exp(
                                s_func_user_list[k](q.item_A, q.item_B, u_vec_list[k])
                            )
                            + np.exp(
                                s_func_user_list[k](q.item_B, q.item_A, u_vec_list[k])
                            )
                            + np.exp(
                                f_func_user_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_voters)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(s_func_user_list[k](q.item_B, q.item_A, u_vec_list[k]))
                        / (
                            np.exp(
                                s_func_user_list[k](q.item_A, q.item_B, u_vec_list[k])
                            )
                            + np.exp(
                                s_func_user_list[k](q.item_B, q.item_A, u_vec_list[k])
                            )
                            + np.exp(
                                f_func_user_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_voters)
                    )
                )
                for q in queries
                if q.response == -1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * np.exp(
                            f_func_user_list[k](
                                q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                            )
                        )
                        / (
                            np.exp(
                                s_func_user_list[k](q.item_A, q.item_B, u_vec_list[k])
                            )
                            + np.exp(
                                s_func_user_list[k](q.item_B, q.item_A, u_vec_list[k])
                            )
                            + np.exp(
                                f_func_user_list[k](
                                    q.item_A, q.item_B, u_vec_list[k], lam_list[k],
                                )
                            )
                        )
                        for k in range(num_voters)
                    )
                )
                for q in queries
                if q.response == 0
            )
        ) - len(queries) * np.log(sum(m_list_exp))

    elif model_name == "ann_classifier":
        return get_ann_log_likelihood(params, queries)

    else:
        u_vec = [params["u1"], params["u2"], params["u3"]]
        lam = params["lam"]
        s_func, f_func, _, _ = get_score_functions(model_name, 0, 0, 0, num_features)

        return (
            sum(s_func(q.item_A, q.item_B, u_vec) for q in queries if q.response == 1)
            + sum(
                s_func(q.item_B, q.item_A, u_vec) for q in queries if q.response == -1
            )
            + sum(
                f_func(q.item_A, q.item_B, u_vec, lam)
                for q in queries
                if q.response == 0
            )
            - sum(
                np.log(
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + np.exp(s_func(q.item_B, q.item_A, u_vec))
                    + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                )
                for q in queries
            )
        )


def calculate_ll_strict(queries, model_name, params, fixed_probs=None):

    num_features = len(queries[0].item_A.features)

    if model_name not in SCORE_MODEL_NAMES + OTHER_MODEL_NAMES + BASELINE_MODEL_NAMES:
        raise Exception("model not recognized")

    if model_name == "fixed_rand":
        # evaluate a randomized baseline, with fixed probabilities for each response type
        # in this case, fixed_probs must be a list s.t. fixed_probs[i] is the prob of response i
        assert fixed_probs is not None
        assert len(fixed_probs) == 2
        if not np.isclose(fixed_probs[-1] + fixed_probs[1], 1.0):
            raise Exception("fixed_probs must sum to 1")

        # number of each responses
        num_r1 = len(list(filter(lambda q: q.response == 1, queries)))
        num_rm1 = len(list(filter(lambda q: q.response == -1, queries)))
        assert num_r1 + num_rm1 == len(queries)

        return num_r1 * np.log(fixed_probs[1]) + num_rm1 * np.log(fixed_probs[-1])

    elif model_name == "simple_logit":
        # params are u [vec] + [p_flip]
        u_vec = [params["u1"], params["u2"], params["u3"]]

        # report the total (not average) ll over all samples.

        return sum(
            -np.log(1 + np.exp(-np.dot(q.z, u_vec))) for q in queries if q.response == 1
        ) + sum(
            -np.log(1 + np.exp(np.dot(q.z, u_vec))) for q in queries if q.response == -1
        )

    elif model_name == "k_mixture":

        num_models = int(params["k"])
        assert num_models > 0

        # now learn a mixture of these voter models, using regularized "importance" parameters
        # define a list of s_func and f_func handles
        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, 0, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # u vectors
        u_vec_list = [
            [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
            for k in range(num_models)
        ]

        # model thresholds
        lam_list = [params[f"lam_{k}"] for k in range(num_models)]

        # s/f functions
        s_func_list = [s_func_map[params[f"model_name_{k}"]] for k in range(num_models)]
        f_func_list = [f_func_map[params[f"model_name_{k}"]] for k in range(num_models)]

        # model importance

        lam_list = [params[f"lam_{k}"] for k in range(num_models)]
        q_param_list = [params[f"q_{k}"] for k in range(num_models)]

        # model importance
        m_list = [params[f"m_{k}"] for k in range(num_models)]
        m_list_exp = [np.exp(m) for m in m_list]

        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + 0.5
                                * np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(s_func_list[k](q.item_A, q.item_B, u_vec_list[k]))
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + 0.5
                                * np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                                + np.exp(
                                    f_func_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(s_func_list[k](q.item_B, q.item_A, u_vec_list[k]))
                            / (
                                np.exp(
                                    s_func_list[k](q.item_A, q.item_B, u_vec_list[k])
                                )
                                + np.exp(
                                    s_func_list[k](q.item_B, q.item_A, u_vec_list[k])
                                )
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == -1
            )
            - len(queries) * np.log(sum(m_list_exp))
        )

    elif model_name == "k_min_diff":

        # mixture of k min-diff models
        num_models = params["k"]

        s_func, f_func, _, _ = get_score_functions("min_diff", 0, 0, 0, num_features)

        # u vectors
        u_vec_list = [
            [params[f"u_{k}_1"], params[f"u_{k}_2"], params[f"u_{k}_3"]]
            for k in range(num_models)
        ]

        lam_list = [params[f"lam_{k}"] for k in range(num_models)]
        q_param_list = [params[f"q_{k}"] for k in range(num_models)]

        # model importance
        m_list = [params[f"m_{k}"] for k in range(num_models)]
        m_list_exp = [np.exp(m) for m in m_list]

        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + 0.5
                                * np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + 0.5
                                * np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                                + np.exp(
                                    f_func(
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            / (
                                np.exp(s_func(q.item_A, q.item_B, u_vec_list[k]))
                                + np.exp(s_func(q.item_B, q.item_A, u_vec_list[k]))
                            )
                        )
                        for k in range(num_models)
                    )
                )
                for q in queries
                if q.response == -1
            )
            - len(queries) * np.log(sum(m_list_exp))
        )

    elif model_name == "voter_mixture":

        raise NotImplemented()

    elif model_name == "uniform_voter_mixture":

        num_voters = len(params["voter_model_names"])

        s_func_map = {}
        f_func_map = {}
        for m_name in SCORE_MODEL_NAMES:
            s_func, f_func, _, _ = get_score_functions(m_name, 0, 0, 0, num_features)
            s_func_map[m_name] = s_func
            f_func_map[m_name] = f_func

        # get a list of all function handles
        s_func_user_list = [
            s_func_map[m_name] for m_name in params["voter_model_names"]
        ]
        f_func_user_list = [
            f_func_map[m_name] for m_name in params["voter_model_names"]
        ]

        u_vec_list = params["u_vec_list"]
        lam_list = params["lam_list"]
        q_param_list = params["q_param_list"]

        m_list = [1.0] * num_voters
        m_list_exp = [np.exp(m) for m in m_list]

        return (
            sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k]
                                    )
                                )
                                + 0.5
                                * np.exp(
                                    f_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    s_func_user_list[k](
                                        q.item_B, q.item_A, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    f_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(
                                s_func_user_list[k](q.item_A, q.item_B, u_vec_list[k])
                            )
                            / (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    s_func_user_list[k](
                                        q.item_B, q.item_A, u_vec_list[k]
                                    )
                                )
                            )
                        )
                        for k in range(num_voters)
                    )
                )
                for q in queries
                if q.response == 1
            )
            + sum(
                np.log(
                    sum(
                        m_list_exp[k]
                        * (
                            q_param_list[k]
                            * (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_B, q.item_A, u_vec_list[k]
                                    )
                                )
                                + 0.5
                                * np.exp(
                                    f_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            / (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    s_func_user_list[k](
                                        q.item_B, q.item_A, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    f_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k], lam_list[k]
                                    )
                                )
                            )
                            + (1.0 - q_param_list[k])
                            * np.exp(
                                s_func_user_list[k](q.item_B, q.item_A, u_vec_list[k])
                            )
                            / (
                                np.exp(
                                    s_func_user_list[k](
                                        q.item_A, q.item_B, u_vec_list[k]
                                    )
                                )
                                + np.exp(
                                    s_func_user_list[k](
                                        q.item_B, q.item_A, u_vec_list[k]
                                    )
                                )
                            )
                        )
                        for k in range(num_voters)
                    )
                )
                for q in queries
                if q.response == -1
            )
            - len(queries) * np.log(sum(m_list_exp))
        )

    elif model_name == "ann_classifier":
        return get_ann_log_likelihood_strict(params, queries)

    else:
        u_vec = [params["u1"], params["u2"], params["u3"]]
        lam = params["lam"]
        q_param = params["q"]
        s_func, f_func, _, _ = get_score_functions(model_name, 0, 0, 0, num_features)

        # report the sum ll over all samples.

        return sum(
            np.log(
                q_param
                * (
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + 0.5 * np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                )
                / (
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + np.exp(s_func(q.item_B, q.item_A, u_vec))
                    + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                )
                + (1.0 - q_param)
                * np.exp(s_func(q.item_A, q.item_B, u_vec))
                / (
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + np.exp(s_func(q.item_B, q.item_A, u_vec))
                )
            )
            for q in queries
            if q.response == 1
        ) + sum(
            np.log(
                q_param
                * (
                    np.exp(s_func(q.item_B, q.item_A, u_vec))
                    + 0.5 * np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                )
                / (
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + np.exp(s_func(q.item_B, q.item_A, u_vec))
                    + np.exp(f_func(q.item_A, q.item_B, u_vec, lam))
                )
                + (1.0 - q_param)
                * np.exp(s_func(q.item_B, q.item_A, u_vec))
                / (
                    np.exp(s_func(q.item_A, q.item_B, u_vec))
                    + np.exp(s_func(q.item_B, q.item_A, u_vec))
                )
            )
            for q in queries
            if q.response == -1
        )


def get_ann_log_likelihood(clf, queries):
    class_order = list(clf.classes_)
    class_map = {}
    for i, label in enumerate(class_order):
        class_map[label] = i

    if set(class_map.keys()) != {"-1", "1", "0"}:
        return -np.inf

    ll_sum = 0
    for q in queries:
        predict_probs = clf.predict_proba([q.z])[0]
        if q.response == 1:
            ll_sum += np.log(predict_probs[class_map["1"]])
        if q.response == -1:
            ll_sum += np.log(predict_probs[class_map["-1"]])
        if q.response == 0:
            ll_sum += np.log(predict_probs[class_map["0"]])

    return ll_sum


def get_ann_log_likelihood_strict(clf, queries):
    class_order = list(clf.classes_)
    class_map = {}
    for i, label in enumerate(class_order):
        class_map[label] = i

    if set(class_map.keys()) != {"-1", "1"}:
        return -np.inf

    ll_sum = 0
    for q in queries:
        predict_probs = clf.predict_proba([q.z])[0]
        if q.response == 1:
            ll_sum += np.log(predict_probs[class_map["1"]])
        if q.response == -1:
            ll_sum += np.log(predict_probs[class_map["-1"]])

    return ll_sum


# def calculate_flip_auc(queries, model_name, params, fixed_probs=None):
#     """
#     calculate the AUC of a simple model which predicts coin-flip vs. no-coin-flip, using sklearn
#
#     we use a binary representation : 0 = coin-flip, 1 = no-coin-flip
#     """
#     num_features = len(queries[0].item_A.features)
#
#     # correct responses: 1 if strict pref., 0 if coin-flip
#     correct_responses = [0 if q.response == 0 else 1 for q in queries]
#
#     if all([c == 0 for c in correct_responses]):
#         return 0.0
#     if all([c == 1 for c in correct_responses]):
#         return 0.0
#
#     if model_name == "fixed_rand":
#         # evaluate a randomized baseline, with fixed probabilities for each response type
#         # in this case, fixed_probs must be a list s.t. fixed_probs[i] is the prob of response i
#         assert fixed_probs is not None
#         assert len(fixed_probs) == 3
#         if not np.isclose(fixed_probs[0] + fixed_probs[-1] + fixed_probs[1], 1.0):
#             raise Exception("fixed_probs must sum to 1")
#
#         # predicted probabilities - probability of strict pref.
#         predicted_responses = [fixed_probs[-1] + fixed_probs[1]] * len(queries)
#
#     elif model_name == "simple_logit":
#         # params are u [vec] + [p_flip]
#         u_vec = params[:-1]
#         p_flip = params[-1]
#
#         # report the total (not average) ll over all samples.
#         predicted_responses = [1.0 - p_flip] * len(queries)
#
#     else:
#         # score-based models
#         # params are u [vec] + [lam]
#
#         s_func, f_func, _, _ = get_score_functions(model_name, 0, 0, 0, num_features)
#
#         predicted_responses = [
#             1.0
#             - np.exp(f_func(q.item_B, q.item_A, params[:-1], params[1]))
#             / (
#                 np.exp(
#                     s_func(q.item_A, q.item_B, params[:-1])
#                     + s_func(q.item_B, q.item_A, params[:-1])
#                     + f_func(q.item_B, q.item_A, params[:-1], params[1])
#                 )
#             )
#             for q in queries
#         ]
#
#     return sklearn.metrics.roc_auc_score(correct_responses, predicted_responses)
