from ._base import get_model, register_model
from .categorical import SimplexCategoricalFlow, SphereCategoricalFlow, LinearCategoricalFlow
from .cnn import ConvNet
from .seq_model import SeqNet
from .transformer import GPT


def get_flow_model(model_cfg, encoder_cfg):
    """
    Build the categorical flow model.
    :param model_cfg: model configs passed to the flow model, type indicates the model type
    :param encoder_cfg: encoder configs passed to the encoder model
    :return: the flow model
    """
    encoder = get_model(encoder_cfg)
    m_cfg = model_cfg.copy()
    m_type = m_cfg.pop('type')
    if m_type == 'simplex':
        return SimplexCategoricalFlow(encoder, **m_cfg)
    elif m_type == 'sphere':
        return SphereCategoricalFlow(encoder, **m_cfg)
    elif m_type == 'linear':
        return LinearCategoricalFlow(encoder, **m_cfg)
    else:
        raise ValueError(f'Unknown model type: {m_type}')
