"""
Benchmark evaluation for the nanochat autoresearch workstream.

This module adapts the in-memory nanochat GPT model to EleutherAI's
lm-evaluation-harness so train.py can optimize against a broad benchmark suite
instead of only validation BPB.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from lm_eval import evaluator
from lm_eval.api.model import LM


ScoreTransform = Callable[[float], float]
EvalProgressCallback = Callable[[str, int, int], None]


def _identity(value: float) -> float:
    return value


def _inverse_one_plus(value: float) -> float:
    return 1.0 / (1.0 + value)


@dataclass(frozen=True)
class BenchmarkSpec:
    label: str
    tasks: tuple[str, ...]
    metrics: tuple[str, ...]
    transform: ScoreTransform = _identity
    raw_metric_label: str | None = None


DEFAULT_BENCHMARKS: "OrderedDict[str, BenchmarkSpec]" = OrderedDict(
    (
        (
            "MMLU",
            BenchmarkSpec("MMLU", ("mmlu",), ("acc,none",)),
        ),
        (
            "MMLU-stem",
            BenchmarkSpec("MMLU-stem", ("mmlu_stem",), ("acc,none",)),
        ),
        (
            "MMLU-exclude-stem",
            BenchmarkSpec(
                "MMLU-exclude-stem",
                ("mmlu_humanities", "mmlu_social_sciences", "mmlu_other"),
                ("acc,none",),
            ),
        ),
        (
            "GSM8k",
            BenchmarkSpec(
                "GSM8k",
                ("gsm8k",),
                ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"),
            ),
        ),
        (
            "MATH_6k",
            BenchmarkSpec("MATH_6k", ("minerva_math",), ("exact_match,none", "math_verify,none")),
        ),
        (
            "MATH_6k_zeroshot",
            BenchmarkSpec("MATH_6k_zeroshot", ("hendrycks_math",), ("exact_match,none",)),
        ),
        (
            "BBH",
            BenchmarkSpec("BBH", ("bbh_zeroshot",), ("exact_match,flexible-extract", "exact_match,none")),
        ),
        (
            "IFEval-nile",
            BenchmarkSpec(
                "IFEval-nile",
                ("ifeval",),
                ("prompt_level_strict_acc,none", "inst_level_strict_acc,none"),
            ),
        ),
        (
            "HumanEval",
            BenchmarkSpec("HumanEval", ("humaneval",), ("pass@1,create_test", "pass_at_1,create_test")),
        ),
        (
            "hellaswag",
            BenchmarkSpec("hellaswag", ("hellaswag",), ("acc_norm,none", "acc,none")),
        ),
        (
            "arc-challenge",
            BenchmarkSpec("arc-challenge", ("arc_challenge",), ("acc_norm,none", "acc,none")),
        ),
        (
            "winogrande",
            BenchmarkSpec("winogrande", ("winogrande",), ("acc,none",)),
        ),
        (
            "TruthfulQA",
            BenchmarkSpec("TruthfulQA", ("truthfulqa_mc1",), ("acc,none",)),
        ),
        (
            "lambada",
            BenchmarkSpec("lambada", ("lambada_openai",), ("acc,none",)),
        ),
        (
            "wikitext2",
            BenchmarkSpec(
                "wikitext2",
                ("wikitext",),
                ("bits_per_byte,none",),
                transform=_inverse_one_plus,
                raw_metric_label="bits_per_byte",
            ),
        ),
    )
)


def sanitize_metric_name(name: str) -> str:
    chars = [c.lower() if c.isalnum() else "_" for c in name]
    return "_".join("".join(chars).split("_"))


def _normalize_selector(name: str) -> str:
    return "".join(c.lower() for c in name if c.isalnum())


def _parse_limit(value: str | None) -> int | float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if parsed <= 0:
        return None
    if parsed < 1:
        return parsed
    return int(parsed)


def _parse_int(value: str | None, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


def _select_benchmarks(names: str | None) -> list[BenchmarkSpec]:
    if not names:
        return list(DEFAULT_BENCHMARKS.values())

    by_selector = {
        _normalize_selector(spec.label): spec
        for spec in DEFAULT_BENCHMARKS.values()
    }
    by_selector.update({
        _normalize_selector(key): spec
        for key, spec in DEFAULT_BENCHMARKS.items()
    })

    specs: list[BenchmarkSpec] = []
    for raw_name in names.split(","):
        name = raw_name.strip()
        if not name:
            continue
        spec = by_selector.get(_normalize_selector(name))
        if spec is None:
            spec = BenchmarkSpec(name, (name,), _default_metric_order())
        specs.append(spec)
    return specs


def _default_metric_order() -> tuple[str, ...]:
    return (
        "acc_norm,none",
        "acc,none",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "pass@1,create_test",
        "pass_at_1,create_test",
        "prompt_level_strict_acc,none",
    )


def _bucket_length(length: int) -> int:
    return max(1, 2 ** math.ceil(math.log2(max(1, length))))


class NanochatLM(LM):
    """lm-evaluation-harness adapter for the local nanochat GPT."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        device: torch.device,
        max_length: int,
        batch_size: int,
        autocast_ctx=None,
        max_gen_toks: int | None = None,
        progress_callback: EvalProgressCallback | None = None,
        progress_interval: int = 100,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.autocast_ctx = autocast_ctx or nullcontext()
        self.max_gen_toks = max_gen_toks
        self.progress_callback = progress_callback
        self.progress_interval = max(1, progress_interval)
        self.bos_token_id = tokenizer.get_bos_token_id()

    def _report_progress(self, phase: str, completed: int, total: int, *, force: bool = False) -> None:
        if self.progress_callback is None or total <= 0:
            return
        if force or completed >= total or completed % self.progress_interval == 0:
            self.progress_callback(phase, completed, total)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def _decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.tokenizer.decode(token_ids)

    def _continuation_tokens(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        context_tokens = self._encode(context)
        whole_tokens = self._encode(context + continuation)
        if whole_tokens[:len(context_tokens)] == context_tokens:
            continuation_tokens = whole_tokens[len(context_tokens):]
            return whole_tokens, continuation_tokens
        continuation_tokens = self._encode(continuation)
        return context_tokens + continuation_tokens, continuation_tokens

    def _make_loglikelihood_example(
        self,
        context: str,
        continuation: str,
    ) -> tuple[list[int], list[int], int]:
        whole_tokens, continuation_tokens = self._continuation_tokens(context, continuation)
        if not continuation_tokens:
            return [self.bos_token_id], [self.bos_token_id], 1

        seq = [self.bos_token_id] + whole_tokens
        input_ids = seq[:-1]
        target_ids = seq[1:]
        score_start = len(seq) - len(continuation_tokens) - 1

        if len(input_ids) > self.max_length:
            offset = len(input_ids) - self.max_length
            input_ids = input_ids[offset:]
            target_ids = target_ids[offset:]
            score_start = max(0, score_start - offset)

        return input_ids, target_ids, score_start

    def _score_examples(
        self,
        examples: list[tuple[list[int], list[int], int]],
        *,
        progress_phase: str | None = None,
    ) -> list[tuple[float, bool]]:
        scored: list[tuple[float, bool]] = []
        if progress_phase is not None:
            self._report_progress(progress_phase, 0, len(examples), force=True)
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start:start + self.batch_size]
            max_len = max(len(input_ids) for input_ids, _, _ in batch)
            seq_len = min(self.max_length, _bucket_length(max_len))
            input_batch = torch.full(
                (len(batch), seq_len),
                self.bos_token_id,
                dtype=torch.long,
                device=self.device,
            )
            for row, (input_ids, _, _) in enumerate(batch):
                input_batch[row, :len(input_ids)] = torch.tensor(
                    input_ids,
                    dtype=torch.long,
                    device=self.device,
                )

            with torch.no_grad(), self.autocast_ctx:
                logits = self.model(input_batch).float()
                logprobs = F.log_softmax(logits, dim=-1)

            for row, (input_ids, target_ids, score_start) in enumerate(batch):
                if score_start >= len(input_ids):
                    scored.append((0.0, True))
                    continue
                positions = torch.arange(score_start, len(input_ids), device=self.device)
                targets = torch.tensor(target_ids[score_start:], dtype=torch.long, device=self.device)
                token_logprobs = logprobs[row, positions, targets]
                greedy_tokens = logits[row, positions].argmax(dim=-1)
                scored.append((token_logprobs.sum().item(), bool(torch.equal(greedy_tokens, targets))))
            if progress_phase is not None:
                self._report_progress(progress_phase, min(start + len(batch), len(examples)), len(examples))

        return scored

    def loglikelihood(self, requests, disable_tqdm: bool = False):
        examples = [
            self._make_loglikelihood_example(*request.arguments)
            for request in requests
        ]
        return self._score_examples(examples, progress_phase="loglikelihood")

    def _rolling_examples(self, text: str) -> list[tuple[list[int], list[int], int]]:
        tokens = self._encode(text)
        if not tokens:
            return []

        examples = []
        cursor = 0
        while cursor < len(tokens):
            chunk = tokens[cursor:cursor + self.max_length]
            prefix = [self.bos_token_id] if cursor == 0 else [tokens[cursor - 1]]
            seq = prefix + chunk
            examples.append((seq[:-1], seq[1:], 0))
            cursor += self.max_length
        return examples

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False):
        request_list = list(requests)
        self._report_progress("loglikelihood_rolling", 0, len(request_list), force=True)
        results = []
        for index, request in enumerate(request_list, start=1):
            examples = self._rolling_examples(request.arguments[0])
            if not examples:
                results.append(0.0)
                self._report_progress("loglikelihood_rolling", index, len(request_list))
                continue
            results.append(sum(logprob for logprob, _ in self._score_examples(examples)))
            self._report_progress("loglikelihood_rolling", index, len(request_list))
        return results

    def _generate_one(self, context: str, gen_kwargs: dict) -> str:
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            until = [until]
        max_gen_toks = int(gen_kwargs.get("max_gen_toks", 64))
        if self.max_gen_toks is not None:
            max_gen_toks = min(max_gen_toks, self.max_gen_toks)

        context_tokens = self._encode(context)
        all_tokens = context_tokens if context_tokens else [self.bos_token_id]
        generated: list[int] = []

        for _ in range(max_gen_toks):
            input_ids = all_tokens[-self.max_length:]
            seq_len = min(self.max_length, _bucket_length(len(input_ids)))
            input_batch = torch.full(
                (1, seq_len),
                self.bos_token_id,
                dtype=torch.long,
                device=self.device,
            )
            input_batch[0, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long, device=self.device)

            with torch.no_grad(), self.autocast_ctx:
                logits = self.model(input_batch)
            next_token = int(logits[0, len(input_ids) - 1].argmax(dim=-1).item())
            all_tokens.append(next_token)
            generated.append(next_token)

            generated_text = self._decode(generated)
            for stop in until:
                if stop and stop in generated_text:
                    return generated_text.split(stop, 1)[0]

        return self._decode(generated)

    def generate_until(self, requests, disable_tqdm: bool = False):
        request_list = list(requests)
        self._report_progress("generate_until", 0, len(request_list), force=True)
        results = []
        for index, request in enumerate(request_list, start=1):
            results.append(self._generate_one(request.arguments[0], request.arguments[1]))
            self._report_progress("generate_until", index, len(request_list))
        return results


def _metric_value(result: dict, preferred_metrics: tuple[str, ...]) -> tuple[str, float] | None:
    for metric in preferred_metrics:
        value = result.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return metric, float(value)

    for metric, value in result.items():
        if metric == "alias" or metric.endswith("_stderr,none"):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return metric, float(value)
    return None


def _summarize_scores(
    specs: list[BenchmarkSpec],
    raw_results: dict,
) -> tuple[float, dict[str, float], dict[str, dict[str, float | str]]]:
    task_scores: dict[str, float] = {}
    task_metrics: dict[str, dict[str, float | str]] = {}

    for spec in specs:
        values = []
        raw_metrics: dict[str, float | str] = {}
        for task in spec.tasks:
            result = raw_results.get(task)
            if result is None:
                continue
            metric_value = _metric_value(result, spec.metrics)
            if metric_value is None:
                continue
            metric, raw_value = metric_value
            score_value = spec.transform(raw_value)
            values.append(score_value)
            metric_name = spec.raw_metric_label or metric.split(",", 1)[0]
            raw_metrics[task] = raw_value
            raw_metrics[f"{task}:{metric_name}"] = raw_value

        if values:
            task_scores[spec.label] = sum(values) / len(values)
            task_metrics[spec.label] = raw_metrics

    benchmark_score = sum(task_scores.values()) / len(task_scores) if task_scores else 0.0
    return benchmark_score, task_scores, task_metrics


def evaluate_benchmark_suite(
    model,
    tokenizer,
    *,
    device: torch.device,
    max_length: int,
    autocast_ctx=None,
    progress_callback: EvalProgressCallback | None = None,
) -> dict:
    specs = _select_benchmarks(os.environ.get("NANOCHAT_BENCHMARKS"))
    tasks = sorted({task for spec in specs for task in spec.tasks})
    limit = _parse_limit(os.environ.get("NANOCHAT_BENCHMARK_LIMIT"))
    batch_size = _parse_int(os.environ.get("NANOCHAT_BENCHMARK_BATCH_SIZE"), 8)
    progress_interval = _parse_int(os.environ.get("NANOCHAT_BENCHMARK_PROGRESS_INTERVAL"), 100)
    max_gen_toks = _parse_limit(os.environ.get("NANOCHAT_BENCHMARK_MAX_GEN_TOKS"))
    max_gen_toks = int(max_gen_toks) if max_gen_toks is not None else None

    lm = NanochatLM(
        model,
        tokenizer,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        autocast_ctx=autocast_ctx,
        max_gen_toks=max_gen_toks,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )

    if any(task.startswith("humaneval") for task in tasks):
        os.environ.setdefault("HF_ALLOW_CODE_EVAL", "1")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=limit,
        log_samples=False,
        bootstrap_iters=0,
        confirm_run_unsafe_code=True,
        verbosity=os.environ.get("NANOCHAT_BENCHMARK_VERBOSITY", "WARNING"),
    )

    benchmark_score, task_scores, task_metrics = _summarize_scores(specs, results["results"])
    return {
        "benchmark_score": benchmark_score,
        "task_scores": task_scores,
        "task_metrics": task_metrics,
        "tasks": tasks,
        "limit": limit,
        "batch_size": batch_size,
        "lm_eval": results,
    }
