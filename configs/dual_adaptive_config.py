from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "DualAdaptiveLearner"
    config.hidden_dims = (256, 256, 256)

    config.backup_entropy = True
    config.action_selection_temperature = 1.0
    config.ensemble_ratio = 0.5
    config.actor_bc_coef = 0.5

    return config
