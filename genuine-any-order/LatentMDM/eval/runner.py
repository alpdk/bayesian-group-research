"""Evaluation dispatch and sampling-grid expansion."""

from copy import deepcopy

from omegaconf import ListConfig

from eval.gsm8k_eval import evaluate_ddp_gsm8k


def _is_sequence_config(value) -> bool:
    return isinstance(value, (list, tuple, ListConfig))


def _as_list(value) -> list:
    if _is_sequence_config(value):
        return list(value)
    return [value]


def _add_eval_result(out: dict, prefix: str, result) -> None:
    if isinstance(result, dict):
        for key, value in result.items():
            out[f"{prefix}_{key}"] = value
    else:
        out[prefix] = result


def evaluate_once(model, cfg, device, rank: int, world_size: int, sampling):
    """Dispatch a single sampling configuration to its dataset evaluator."""
    if cfg.data.dataset in ("tinygsm", "tinygsm_split", "tinygsm_split_v3", "apps", "taco"):
        return evaluate_ddp_gsm8k(model, cfg, device, rank, world_size, sampling)
    raise ValueError(f"Invalid dataset: {cfg.data.dataset}")


def evaluate_sampling_grid(model, cfg, device, rank: int, world_size: int) -> dict:
    """Evaluate every configured confidence/unmasking combination."""
    sampling = cfg.validation.sampling
    strategy = cfg.training.strategy

    if strategy == "latmdm":
        confidence_cfg = getattr(sampling, "confidence", None)
        unmasking_num_cfg = getattr(sampling, "unmasking_num", 1)
        confidence_is_list = _is_sequence_config(confidence_cfg)
        unmasking_num_is_list = _is_sequence_config(unmasking_num_cfg)
        confidence_values = (
            _as_list(confidence_cfg) if confidence_cfg is not None else [None]
        )
        unmasking_values = _as_list(unmasking_num_cfg)

        out = {}
        for confidence_value in confidence_values:
            for unmasking_num_value in unmasking_values:
                sampling_i = deepcopy(sampling)
                if confidence_value is not None:
                    sampling_i.confidence = confidence_value
                sampling_i.unmasking_num = int(unmasking_num_value)
                result = evaluate_once(
                    model, cfg, device, rank, world_size, sampling_i
                )
                key_parts = [strategy]
                if confidence_is_list and confidence_value is not None:
                    key_parts.append(str(confidence_value))
                if unmasking_num_is_list:
                    key_parts.append(f"unmasking_{int(unmasking_num_value)}")
                _add_eval_result(out, "_".join(key_parts), result)
        return out

    if strategy == "arm":
        confidence = getattr(sampling, "confidence", None)
        if confidence is None:
            result = evaluate_once(model, cfg, device, rank, world_size, sampling)
            out = {}
            _add_eval_result(out, strategy, result)
            return out

        out = {}
        confidence_is_list = _is_sequence_config(confidence)
        for confidence_value in _as_list(confidence):
            sampling_i = deepcopy(sampling)
            sampling_i.confidence = confidence_value
            result = evaluate_once(model, cfg, device, rank, world_size, sampling_i)
            key = (
                f"{strategy}_{confidence_value}"
                if confidence_is_list
                else strategy
            )
            _add_eval_result(out, key, result)
        return out

    base_sampling = sampling
    out = {}
    for confidence in _as_list(base_sampling.confidence):
        for unmasking_num in _as_list(base_sampling.unmasking_num):
            sampling_i = deepcopy(base_sampling)
            sampling_i.confidence = confidence
            sampling_i.unmasking_num = unmasking_num
            result = evaluate_once(model, cfg, device, rank, world_size, sampling_i)
            _add_eval_result(
                out, f"{confidence}_unmasking_{unmasking_num}", result
            )
    return out


# Compatibility aliases for existing imports from the entry-point modules.
evaluate_ddp = evaluate_once
evaluate_ddp_dict = evaluate_sampling_grid
