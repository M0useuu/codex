from configs import sac_config


def get_config():
    config = sac_config.get_config()

    config.model_cls = "DualAdaptiveLearner"
    config.hidden_dims = (256, 256, 256)

    config.target_minmax_weight = 0.75
    config.backup_entropy = True
    config.action_selection_temperature = 1.0

    return config
