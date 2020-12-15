# This file contains the following classes:
#
# Item : an item x \in X.
# Agent : a decision-maker.
# Query : a pairwise comparison between two Items.

import itertools

import numpy as np


class Item(object):
    def __init__(self, features, id):
        self.features = np.array(features)
        self.id = id

    def __eq__(self, other):
        if len(self.features) != len(other.features):
            return False
        else:
            return [
                self.features[i] == other.features[i]
                for i in range(len(other.features))
            ].all()

    def to_str(self):
        print("Item id: %d" % self.id)
        print("Features:")

        for i, f in enumerate(self.features):
            print("%d: %s" % (i, str(f)))

    @classmethod
    def random(cls, num_features, id, rs, sphere_size=1.0):
        # generate a random item, with features uniformly drawn from the num_features-dimensional sphere
        # rs (optional) : provide a random state

        x = sphere_size * generate_random_point_nsphere(num_features, rs=rs)

        return cls(x, id)


class Query(object):
    """
    'response' is:
    item_A > item_B : response = 1
    item_A < item_B : response = -1
    item_A ~ item_B : response = 0
    """

    valid_responses = [0, 1, -1]

    def __init__(self, item_A, item_B, response=None, order=None):
        self.item_A = item_A
        self.item_B = item_B
        self.order = order
        if response is not None:
            self.set_response(response)
        else:
            self.response = None

    def set_response(self, response):
        if response in self.valid_responses:
            self.response = response

    def is_answered(self):
        return self.response is not None

    def unanswered_copy(self):
        # return an identical query, with no response
        return Query(self.item_A, self.item_B, response=None)

    @property
    def z(self):
        # difference in feature vectors between A, B
        return self.item_A.features - self.item_B.features

    def dist_to_point(self, p):
        # return the distance of self.z (a hyperplane) to the point p
        # np.linalg.norm is 2-norm, by default
        return np.abs(np.dot(p, self.z)) / np.linalg.norm(self.z)


class Agent(object):
    """
    id : unique identifier
    u_set : uncertainty set over utilities (a list of intervals)
    u_true : true utility vector, if known
    """

    def __init__(
        self, id, num_feats, model_type, model_param, u_set=None, u_vector=None
    ):
        self.id = id
        self.num_feats = num_feats
        if u_set is None:
            u_set = np.array([[-1, 1] for _ in range(num_feats)])
        self.u_set = u_set
        self.u_vector = np.array(u_vector)
        self.answered_queries = []
        self.model_type = model_type
        self.model_param = model_param
        self.response_function = self.agent_response_model(
            model_type=model_type, model_param=model_param
        )

    def to_string(self):
        # return a string representation of the agent
        display_dict = {
            "id": self.id,
            "u_vector": list(np.array(self.u_vector)),
            "model_type": self.model_type,
            "model_param": self.model_param,
        }
        return str(display_dict)

    def item_utility(self, item):
        """return the individual's utility of the item, if u_true is known"""
        if self.u_vector is None:
            return None
        else:
            return np.dot(self.u_vector, item.features)

    def util_pref(self, query):
        """
        return 1 if u(query.item_A) > u(query.item_B), and -1 otherwise.
        i.e., the agent's preference function if no-preference is not allowed

        ****all ties are broken in favor of query.item_B****
        """
        if self.item_utility(query.item_A) > self.item_utility(query.item_B):
            return 1
        else:
            return -1

    def dominate_response(self, query, feature_list=None, weak=False):
        """return 1 if query.item_A dominates query.item_B, -1 if B dominates A, and 0 otherwise
        feat_list : binary vector of length num_features. if feat_list[i] = 1, then feature i is required for dominance
        weak = True: indicates weak dominance -- A can dominate B if A dominates B in *at least one featuere*, and ties
                    in the rest. *NOT CURRENTLY IMPLEMENTED*
        """

        assert len(feature_list) == self.num_feats

        if weak:
            raise Warning("weak dominance is not currently implemented")

        query_utility = self.u_vector * query.z

        feature_mask = np.array(feature_list) == 1

        # if utility for all features with feat_list[i] == 1 are >0 , return 1
        if (query_utility[feature_mask] > 0).all():
            return 1
        elif (query_utility[feature_mask] < 0).all():
            return -1
        else:
            return 0

    def true_item_rank(self, item, items):
        """find the *true* rank of item, among items"""

        # true utility of all items
        true_util = [self.item_utility(i) for i in items if (i != item)]

        # get the item's true utility -- append to the end (so use index -1 later
        true_util.append(self.item_utility(item))

        # find the rank of this item in the true utility
        order = np.array(true_util).argsort()
        ranks = order.argsort()

        return ranks[-1]

    def get_response(self, query):
        """get the agent's response to a query"""

        return self.response_function(query)

    def answer_query(self, query, response=None):
        """add an answered query to the Agent's list
        if the response is None, get the Agent's true response"""

        # don't edit the query, use a copy
        q_copy = query.unanswered_copy()

        if response is not None:
            q_copy.set_response(response)
        elif self.u_vector is not None:
            q_copy.set_response(self.get_response(query))
        else:
            raise Warning("query must have a response, unless self.u_true is defined")

        self.answered_queries.append(q_copy)

    def answer_query_list(self, query_list):
        """iterate through a list of queries.
        no responses can be provided."""

        for q in query_list:
            self.answer_query(q, response=None)

    def remove_last_query(self):
        self.answered_queries = self.answered_queries[:-1]

    def agent_response_model(self, model_type=None, model_param=None):
        """
        this function contains all possible agent response models we'll consider. the main (or... only) difference
        between these models is under what circumstances an agent indicates "no preference" (i.e., response = 0)

        inputs:
        - model_type, model_param: this is a keyword, but is a required input that indicates which preference model
                class is used. model types, and params, are:
                min-utility : min-utility threshold; model_param = epsilon (min-utility difference)
                max-utility : max-utility threshold; model_param = epsilon (max-utility difference)
                strict-dominance : strict strong dominance; model param is not used
                partial-dominance: partial strong dominance; model param is a list of length num_features where model_param[i] = 1
                    indicates that feature i is required for dominance.
                likability-threshold: likability threshold; model param is threshold: if both items have a utility above model_param,
                    "no preference", otherwise preference by utility
                max-likability-threshold: same as above, but "no preference" if both items have utility <= threshold

        outputs:
        - a *function handle* that takes a Query and returns a response (0,1,2)

        NOTE: for all threshold models, the coinflip occurs when utility is <= (or >= eps), and a strict preference
        occurs when < or >. (i.e. U == eps means coinflip.
        """

        if model_type == "min-utility":
            # 0 : min-utility threshold
            # model_param = epsilon (min-utility difference)
            if not is_number(model_param):
                raise Warning(
                    "model_type == 'min-utility': model_param must be a number"
                )
            elif model_param < 0:
                raise Warning(
                    "model_type == 'min-utility: model_param must be positive"
                )
            else:
                eps = model_param
                return (
                    lambda q: 0
                    if (
                        abs(self.item_utility(q.item_A) - self.item_utility(q.item_B))
                        <= eps
                    )
                    else self.util_pref(q)
                )

        elif model_type == "max-utility":
            # 1 : max-utility threshold
            # model_param = epsilon (max-utility difference)
            if not is_number(model_param):
                raise Warning(
                    "model_type == 'max-utility': model_param must be a number"
                )
            elif model_param < 0:
                raise Warning(
                    "model_type == 'max-utility': model_param must be positive"
                )
            else:
                eps = model_param
                return (
                    lambda q: 0
                    if (
                        abs(self.item_utility(q.item_A) - self.item_utility(q.item_B))
                        >= eps
                    )
                    else self.util_pref(q)
                )

        elif model_type == "strict-dominance":
            # 2 : strict strong dominance
            # model param is not used
            # all features of one item must be "better" than all features of the other item
            return lambda q: self.dominate_response(
                q, feature_list=np.ones(self.num_feats), weak=False
            )

        elif model_type == "partial-dominance":
            # 3: partial strong dominance
            # model param is subset of feature indices where dominance is required
            return lambda q: self.dominate_response(
                q, feature_list=model_param, weak=False
            )

        elif model_type == "likability-threshold":
            # 4: likability threshold
            # model param is threshold : if both items have a utility above model_param, "no preference", otherwise
            # preference by utility
            if not is_number(model_param):
                raise Warning(
                    "model_type == 'likability-threshold': model_param must be a number"
                )
            else:
                eps = model_param
                return (
                    lambda q: 0
                    if (
                        (self.item_utility(q.item_A) >= eps)
                        and (self.item_utility(q.item_B) >= eps)
                    )
                    else self.util_pref(q)
                )

        elif model_type == "max-likability-threshold":
            # 5: max-likability threshold
            # model param is threshold : if both items have a utility below model_param, "no preference", otherwise
            # preference by utility
            if not is_number(model_param):
                raise Warning(
                    "model_type == 'max-likability-threshold': model_param must be a number"
                )
            else:
                eps = model_param
                return (
                    lambda q: 0
                    if (
                        (self.item_utility(q.item_A) <= eps)
                        and (self.item_utility(q.item_B) <= eps)
                    )
                    else self.util_pref(q)
                )
        else:
            raise Warning("model_type is not recognized")


def is_number(x):
    """return True if x is int or float, false otherwise"""
    if type(x) in [int, float]:
        return True

    return False


def generate_items(num_features, num_items, item_sphere_size=None, seed=None):
    rs = np.random.RandomState(seed)
    return [
        Item.random(num_features, id, rs, sphere_size=item_sphere_size)
        for id in range(num_items)
    ]


def generate_items_and_queries(
    num_features, num_items, item_sphere_size=None, seed=None
):

    items = generate_items(
        num_features, num_items, item_sphere_size=item_sphere_size, seed=seed
    )

    # enumerate all queries
    all_queries = [Query(a, b) for a, b in itertools.combinations(items, 2)]

    return items, all_queries
