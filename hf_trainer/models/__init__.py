from .encoder_decoder_hf.model import EncoderDecoderHFModel
from .vision_language_modernbert.model import VisionLanguageModernBertModel
import copy


registry = {
    "EncoderDecoderHF": EncoderDecoderHFModel,
    "VisionLanguageModernBert": VisionLanguageModernBertModel,
}

def create_model(config, train_dataset, logger):
    model_config = copy.deepcopy(config.model)
    proto = model_config.get('proto')
    assert proto in registry, f"Model {proto} not found"

    model_kwargs = {k: v for k, v in model_config.items() if k != 'proto'}

    model = registry[proto](
        config=model_kwargs,
        train_dataset=train_dataset,
        logger=logger
    )
    return model
