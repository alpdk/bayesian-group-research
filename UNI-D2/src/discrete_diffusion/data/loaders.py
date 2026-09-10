"""Top-level loader API for discrete diffusion training."""

from __future__ import annotations

import functools
import os
import re
from typing import Optional

import datasets
import tokenizers
import torch
import transformers

from .. import utils
from .datasets import (
    generate_synthetic_dataset,
    get_lambada_test_dataset,
    get_text8_dataset,
)
from .processing import (
    _apply_detokenizer,
    _group_texts,
    lm1b_detokenizer,
    lambada_detokenizer,
    ptb_detokenizer,
    scientific_papers_detokenizer,
    wt_detokenizer,
)
from .tokenizers import SyntheticTokenizer, Text8Tokenizer
from .flex_chunking import chunk_documents

LOGGER = utils.get_logger(__name__)

QA_DATASETS = frozenset({"gsm8k", "tinygsm"})

_TINYGSM_DOCSTRING_RE = re.compile(
    r'\n\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', flags=re.MULTILINE)


def _strip_tinygsm_docstring(code: str) -> str:
  return _TINYGSM_DOCSTRING_RE.sub("", code or "", count=1)


def _load_tinygsm_split(
    cache_dir: str,
    max_examples: int,
    val_ratio: float,
    seed: int,
):
  """Materialize a TinyGSM subset (HF train-only) into train/test splits."""
  raw_path = os.path.join(
    cache_dir, f"tinygsm_raw_n{max_examples}_seed{seed}.dat")
  if utils.fsspec_exists(raw_path):
    LOGGER.info("Loading TinyGSM subset from: %s", raw_path)
    return datasets.load_from_disk(raw_path)
  LOGGER.info(
    "Streaming TinyGSM/TinyGSM first %s examples (val_ratio=%s)",
    max_examples, val_ratio)
  stream = datasets.load_dataset(
    "TinyGSM/TinyGSM", split="train", streaming=True)
  rows = []
  for ex in stream:
    rows.append({
      "question": (ex.get("question") or "").strip(),
      "answer": _strip_tinygsm_docstring(ex.get("code") or "").strip(),
    })
    if len(rows) >= max_examples:
      break
  full = datasets.Dataset.from_list(rows)
  if float(val_ratio) <= 0:
    out = datasets.DatasetDict({"train": full, "test": full})
  else:
    n_val = max(1, int(round(len(full) * val_ratio)))
    n_val = min(n_val, len(full) - 1)
    split = full.train_test_split(test_size=n_val, seed=seed, shuffle=True)
    out = datasets.DatasetDict({"train": split["train"], "test": split["test"]})
  LOGGER.info("Saving TinyGSM subset to: %s", raw_path)
  out.save_to_disk(raw_path)
  return out


__all__ = [
    "get_tokenizer",
    "get_dataset",
    "get_dataloaders",
    "QA_DATASETS",
]


def get_dataset(dataset_name,
                tokenizer,
                wrap,
                mode,
                cache_dir,
                insert_eos=True,
                insert_special_tokens=True,
                block_size=1024,
                num_proc=len(os.sched_getaffinity(0)),
                streaming=False,
                revision: Optional[str] = None,
                min_length: int = 0,
                chunking: str = "none",
                max_examples: Optional[int] = None,
                val_ratio: float = 0.02,
                split_seed: int = 42):
  chunking_mode = (chunking or "none").lower()
  if chunking_mode not in {"none", "double_newline"}:
    raise ValueError(f"Unsupported chunking mode: {chunking_mode}")
  if wrap and chunking_mode != "none":
    raise ValueError("Delimiter-based chunking only applies when wrap=False.")
  eos_tag = ""
  if not insert_eos:
    eos_tag += "_eosFalse"
  if not insert_special_tokens:
    eos_tag += "_specialFalse"
  min_len_tag = f"_min{min_length}" if (min_length and not wrap) else ""
  chunk_tag = "_flexchunk" if (not wrap and chunking_mode != "none") else ""
  qa_tag = "_qa" if dataset_name in QA_DATASETS else ""
  if dataset_name == "tinygsm" and max_examples:
    qa_tag += f"_n{max_examples}"
  if wrap:
    filename = f"{dataset_name}_{mode}_bs{block_size}_wrapped{eos_tag}{qa_tag}.dat"
  else:
    filename = f"{dataset_name}_{mode}_bs{block_size}_unwrapped{chunk_tag}{eos_tag}{min_len_tag}{qa_tag}.dat"
  _path = os.path.join(cache_dir, filename)

  if utils.fsspec_exists(_path):
    LOGGER.info("Loading data from: %s", _path)
    return datasets.load_from_disk(_path).with_format("torch")
  LOGGER.info("Generating new data at: %s", _path)
  LOGGER.info("streaming=%s", streaming)

  crop_train = dataset_name == "text8-crop"
  if mode == "train" and crop_train:
    block_size *= 2

  if dataset_name == "wikitext103":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-103-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "wikitext2":
    dataset = datasets.load_dataset(
      "wikitext",
      name="wikitext-2-raw-v1",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "ptb":
    dataset = datasets.load_dataset(
      "ptb_text_only",
      cache_dir=cache_dir,
      revision=revision)
  elif dataset_name == "lambada":
    dataset = get_lambada_test_dataset()
  elif dataset_name == "text8":
    assert wrap
    assert revision is None
    dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
  elif dataset_name == "text8-crop":
    assert revision is None
    dataset = get_text8_dataset(
      cache_dir, max_seq_length=block_size, crop_train=True)
  elif dataset_name == "openwebtext-train":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[:-100000]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
  elif dataset_name == "openwebtext-valid":
    dataset = datasets.load_dataset(
      "openwebtext",
      split="train[-100000:]",
      cache_dir=cache_dir,
      revision=revision,
      streaming=False,
      num_proc=num_proc,
      trust_remote_code=True)
  elif dataset_name == "scientific_papers_arxiv":
    dataset = datasets.load_dataset(
      "scientific_papers", "arxiv",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "scientific_papers_pubmed":
    dataset = datasets.load_dataset(
      "scientific_papers", "pubmed",
      trust_remote_code=True,
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "ag_news":
    dataset = datasets.load_dataset(
      "ag_news",
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "gsm8k":
    dataset = datasets.load_dataset(
      "openai/gsm8k",
      name="main",
      cache_dir=cache_dir,
      streaming=streaming,
      revision=revision)
  elif dataset_name == "tinygsm":
    n_ex = int(max_examples) if max_examples else 100000
    dataset = _load_tinygsm_split(
      cache_dir, n_ex, val_ratio=val_ratio, seed=split_seed)
  elif dataset_name == "synthetic":
    assert streaming
    assert wrap
    dataset = generate_synthetic_dataset(
      train_dataset_size=100000,
      validation_dataset_size=1024,
      seq_len=32,
      vocab_size=256,
    )
  else:
    dataset = datasets.load_dataset(
      dataset_name,
      cache_dir=cache_dir,
      streaming=streaming,
      trust_remote_code=True,
      revision=revision)

  if dataset_name in ["lambada", "openwebtext-train",
                      "openwebtext-valid"]:
    data = dataset
  else:
    data = dataset[mode]
    if dataset_name == "synthetic":
      return data

  if dataset_name.startswith("wikitext"):
    detokenizer = wt_detokenizer
  elif dataset_name == "lm1b":
    detokenizer = lm1b_detokenizer
  elif dataset_name == "ptb":
    detokenizer = ptb_detokenizer
  elif dataset_name == "lambada":
    detokenizer = lambada_detokenizer
  elif dataset_name.startswith("scientific_papers"):
    detokenizer = scientific_papers_detokenizer
  else:
    detokenizer = None

  EOS = tokenizer.eos_token_id
  BOS = tokenizer.bos_token_id

  tokenizer.padding_side = "right"
  tokenizer.truncation_side = "right"

  use_chunking = chunking_mode != "none"
  if use_chunking:
    if chunking_mode == "double_newline":
      delimiter_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
    else:
      delimiter_tokens = []
    if not delimiter_tokens:
      raise ValueError(
        "Tokenizer did not produce any tokens for the specified chunking delimiter.")
  else:
    delimiter_tokens = []

  def preprocess_and_tokenize(example):
    if dataset_name == "ptb":
      text = example["sentence"]
    elif dataset_name in QA_DATASETS:
      text = [q + "\n" + a for q, a in
              zip(example["question"], example["answer"])]
    elif "scientific_papers" in dataset_name:
      text = example["article"]
    else:
      text = example["text"]
    if detokenizer is not None:
      text = _apply_detokenizer(detokenizer)(text)
    if use_chunking:
      return chunk_documents(
        tokenizer,
        text,
        max_length=block_size,
        delimiter_tokens=delimiter_tokens,
        add_special_tokens=insert_special_tokens)
    if wrap:
      tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False)
      if insert_eos:
        tokens = {'input_ids': [t + [EOS] for t in tokens['input_ids']]}
    else:
      tokens = tokenizer(
        text,
        max_length=block_size,
        padding="max_length",
        truncation=True,
        add_special_tokens=insert_special_tokens,
        return_attention_mask=True,
        return_token_type_ids=True)
    if dataset_name in QA_DATASETS and not wrap:
      newline_ids = tokenizer.encode("\n", add_special_tokens=False)
      prompt_lens = []
      for question in example["question"]:
        q_ids = tokenizer.encode(question, add_special_tokens=False)
        plen = len(q_ids) + len(newline_ids)
        if insert_special_tokens:
          plen += 1  # BOS
        prompt_lens.append(min(plen, block_size))
      tokens["prompt_len"] = prompt_lens
    return tokens

  map_kwargs = {
    "batched": True,
  }
  if use_chunking:
    map_kwargs["remove_columns"] = ["text"]
  if not streaming:
    map_kwargs.update(load_from_cache_file=True, desc="Tokenizing")
    if num_proc is not None and int(num_proc) > 1:
      map_kwargs["num_proc"] = int(num_proc)
  tokenized_dataset = data.map(
    preprocess_and_tokenize,
    **map_kwargs)
  if dataset_name == "ptb":
    tokenized_dataset = tokenized_dataset.remove_columns("sentence")
  elif dataset_name in QA_DATASETS:
    drop = [c for c in ("question", "answer", "input_output")
            if c in tokenized_dataset.column_names]
    if drop:
      tokenized_dataset = tokenized_dataset.remove_columns(drop)
  elif "scientific_papers" in dataset_name:
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["article", "abstract", "section_names"])
  elif dataset_name == "ag_news":
    tokenized_dataset = tokenized_dataset.remove_columns(
      ["text", "label"])
  elif "text" in tokenized_dataset.column_names:
    tokenized_dataset = tokenized_dataset.remove_columns("text")

  if (not wrap) and min_length > 0 and (not streaming):
    def _has_min_length(example):
      mask = example.get("attention_mask", None)
      if mask is None:
        return True
      return sum(mask) >= min_length

    tokenized_dataset = tokenized_dataset.filter(
      _has_min_length,
      load_from_cache_file=True,
      desc="Filtering min length",
      **({"num_proc": int(num_proc)} if num_proc is not None and int(num_proc) > 1 else {}))

  if not wrap:
    if not streaming:
      tokenized_dataset.save_to_disk(_path)
    return tokenized_dataset.with_format("torch")

  group_texts = functools.partial(
    _group_texts,
    block_size=block_size,
    bos=BOS,
    eos=EOS,
    insert_special_tokens=insert_special_tokens)
  if streaming:
    chunked_dataset = tokenized_dataset.map(group_texts, batched=True)
  else:
    chunked_dataset = tokenized_dataset.map(
      group_texts,
      batched=True,
      load_from_cache_file=True,
      desc="Grouping",
      **({"num_proc": int(num_proc)} if num_proc is not None and int(num_proc) > 1 else {}))
    chunked_dataset.save_to_disk(_path)
  chunked_dataset = chunked_dataset.with_format("torch")
  return chunked_dataset


def get_tokenizer(config):
  if config.data.tokenizer_name_or_path == "text8":
    tokenizer = Text8Tokenizer()
  elif config.data.tokenizer_name_or_path == "bert-base-uncased":
    tokenizer = transformers.BertTokenizer.from_pretrained(
      "bert-base-uncased")
  elif config.data.tokenizer_name_or_path == "synthetic":
    tokenizer = SyntheticTokenizer(vocab_size=256)
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      config.data.tokenizer_name_or_path)
  if isinstance(tokenizer, (transformers.GPT2TokenizerFast,
                            transformers.GPT2Tokenizer)):
    tokenizer._tokenizer.post_processor = (
      tokenizers.processors.BertProcessing(
        (tokenizer.bos_token, tokenizer.bos_token_id),
        (tokenizer.eos_token, tokenizer.eos_token_id)))
  if tokenizer.bos_token is None:
    if tokenizer.cls_token is None:
      raise AttributeError(
        "Tokenizer must have a bos_token or "
        f"cls_token: {tokenizer}")
    tokenizer.bos_token = tokenizer.cls_token
  if tokenizer.eos_token is None:
    if tokenizer.sep_token is None:
      raise AttributeError(
        "Tokenizer must have a eos_token "
        f"or sep_token: {tokenizer}")
    tokenizer.eos_token = tokenizer.sep_token
  if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
  if getattr(tokenizer, 'mask_token', None) is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})
  return tokenizer


def get_dataloaders(config, tokenizer, skip_train=False,
                    skip_valid=False, valid_seed=None):
  num_gpus = torch.cuda.device_count()
  assert (config.loader.global_batch_size
          == (config.loader.batch_size
              * config.trainer.num_nodes
              * num_gpus
              * config.trainer.accumulate_grad_batches))
  if config.loader.global_batch_size % (
    num_gpus * config.trainer.accumulate_grad_batches) != 0:
    raise ValueError(
      f"Train Batch Size {config.loader.batch_size} "
      f"not divisible by {num_gpus} gpus with accumulation "
      f"{config.trainer.accumulate_grad_batches}.")
  if config.loader.eval_global_batch_size % num_gpus != 0:
    raise ValueError(
      f"Eval Batch Size for {config.eval.batch_size} "
      f"not divisible by {num_gpus}.")
  default_chunking = config.data.get("chunking", "none")
  train_chunking = config.data.get("train_chunking", default_chunking)
  valid_chunking = config.data.get("valid_chunking", default_chunking)
  max_examples = config.data.get("max_examples", None)
  val_ratio_raw = config.data.get("val_ratio", 0.02)
  val_ratio = 0.02 if val_ratio_raw is None else float(val_ratio_raw)
  split_seed = int(config.data.get("split_seed", config.get("seed", 42)) or 42)
  extra_ds = dict(
    max_examples=max_examples, val_ratio=val_ratio, split_seed=split_seed)
  if skip_train:
    train_set = None
  else:
    train_min_length = config.data.get(
      "train_min_length", config.data.get("min_length", 0))
    train_set = get_dataset(
      config.data.train,
      tokenizer,
      mode="train",
      wrap=config.data.wrap,
      insert_eos=config.data.insert_train_eos,
      insert_special_tokens=getattr(
        config.data, "insert_train_special", True),
      cache_dir=config.data.cache_dir,
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("train_revision", None),
      min_length=train_min_length,
      chunking=train_chunking,
      **extra_ds)

  if config.data.valid in ["text8", "lm1b", "ag_news", "gsm8k", "tinygsm"]:
    validation_split = "test"
  else:
    validation_split = "validation"
  if skip_valid:
    valid_set = None
  else:
    valid_min_length = config.data.get(
      "valid_min_length", config.data.get("min_length", 0))
    valid_set = get_dataset(
      config.data.valid,
      tokenizer,
      wrap=config.data.wrap,
      mode=validation_split,
      cache_dir=config.data.cache_dir,
      insert_eos=config.data.insert_valid_eos,
      insert_special_tokens=getattr(
        config.data, "insert_valid_special", True),
      block_size=config.model.length,
      streaming=config.data.streaming,
      num_proc=config.loader.num_workers,
      revision=config.data.get("valid_revision", None),
      min_length=valid_min_length,
      chunking=valid_chunking,
      **extra_ds)

  if skip_train:
    train_loader = None
  else:
    train_loader = torch.utils.data.DataLoader(
      train_set,
      batch_size=config.loader.batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=not config.data.streaming,
      persistent_workers=True)
    train_loader.tokenizer = tokenizer
  if skip_valid:
    valid_loader = None
  else:
    if valid_seed is None:
      shuffle_valid = False
      generator = None
    else:
      shuffle_valid = True
      generator = torch.Generator().manual_seed(valid_seed)
    valid_loader = torch.utils.data.DataLoader(
      valid_set,
      batch_size=config.loader.eval_batch_size,
      num_workers=config.loader.num_workers,
      pin_memory=config.loader.pin_memory,
      shuffle=shuffle_valid,
      generator=generator,
      persistent_workers=True)
    valid_loader.tokenizer = tokenizer

  return train_loader, valid_loader
