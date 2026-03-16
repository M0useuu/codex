from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "CQLLearner"

    config.cql_n_actions = 10
    config.cql_importance_sample = True
    config.cql_temp = 1.0
    config.cql_alpha = 5.0
    config.cql_max_target_backup = False

    config.cql_lagrange = False
    config.cql_target_action_gap = 10.0
    config.cql_alpha_lr = 3e-4

    return config
