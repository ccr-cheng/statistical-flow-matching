_MODEL_DICT = {}


def register_model(name):
    def decorator(cls):
        _MODEL_DICT[name] = cls
        return cls

    return decorator


def get_model(cfg):
    m_cfg = cfg.copy()
    m_type = m_cfg.pop('type')
    return _MODEL_DICT[m_type](**m_cfg)
