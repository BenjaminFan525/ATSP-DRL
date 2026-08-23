#!/usr/bin/env python3
"""Generate replay-verified Stage-2 resource IGA teachers.

One closed Stage-1 plane policy is evaluated deterministically while a
real-coded GA evolves dispatch priorities for both ordinary mobile resources
and R014.  Like the Stage-1 IGA runner, the unit of CPU parallelism is an
independent dataset case.  The frozen plane forwards from those cases are
combined into one GPU batch, avoiding both per-process model copies and the
old mistake of spending a case's wall-time budget on a parallel population.

IGA-1800 is implemented as a nested refinement of IGA-180.  Its first
incumbent is the verified IGA-180 chromosome and only the remaining wall-time
budget is charged in the second phase, so the nominal cumulative budget is
1800 seconds rather than 1980 seconds.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from onpolicy.envs.HKBZ.environment import AircraftScheduleEnv
from onpolicy.envs.HKBZ.experiment.eval_common import list_case_folders
from onpolicy.envs.HKBZ.experiment.evaluate_resource_iga_ablation import (
    ARM_BACKENDS,
    BatchedCaseEvaluator,
    CrossCaseBatchedEvaluator,
    FrozenPlaneEvaluator,
    GenomeLayout,
    _atomic_json,
    _derived_seed,
    _metadata,
    _model_digest,
    _next_population,
    _sha256_file,
    _validate_stage1_handoff,
    load_frozen_policy,
)

TEACHER_SCOPE = "stage2_resource_policy"
TEACHER_METHOD = "resource_iga_all"
TEACHER_SCHEMA_VERSION = 1
SEARCH_CONTRACT_VERSION = 3
BACKENDS = ARM_BACKENDS["iga_all"]


def _initial_population(
    layout: GenomeLayout,
    population: int,
    seed: int,
    warm_chromosome: np.ndarray | None,
) -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    values = rng.random((int(population), layout.n_var))
    heuristic = np.full(layout.n_var, 0.5, dtype=np.float64)
    heuristic[-4:] = np.asarray([1.0, 1.0, 0.5, 1.0])
    if warm_chromosome is None:
        values[0] = heuristic
    else:
        warm = np.asarray(warm_chromosome, dtype=np.float64).reshape(-1)
        if warm.size != layout.n_var:
            raise ValueError(
                f"Warm-start chromosome has {warm.size} variables; "
                f"expected {layout.n_var}."
            )
        if not np.isfinite(warm).all() or np.any(warm < 0.0) or np.any(warm > 1.0):
            raise ValueError("Warm-start chromosome is outside [0, 1].")
        values[0] = warm
        if population > 1:
            values[1] = heuristic
    return rng, values


def _optimize_teacher_population_parallel_legacy(
    evaluator: FrozenPlaneEvaluator,
    case_path: Path,
    *,
    population: int,
    max_generations: int,
    time_budget_seconds: float,
    cumulative_budget_seconds: float,
    seed: int,
    warm_chromosome: np.ndarray | None,
    warm_start: Mapping | None,
) -> tuple[dict, dict]:
    """Retained only for forensic comparison with the stopped v2 flow."""

    layout = evaluator.build_layout(case_path, BACKENDS)
    if not layout.selected_device_indices:
        raise RuntimeError(f"{case_path.name} contains no IGA-controlled devices.")
    rng, chromosomes = _initial_population(
        layout, population, seed, warm_chromosome
    )
    best_objective = math.inf
    best_chromosome = None
    evaluated = 0
    generation_seconds: list[float] = []
    completed_per_generation: list[int] = []
    admission_estimates: list[float] = []
    stopped_with_seconds_remaining = 0.0
    batch = BatchedCaseEvaluator(evaluator, case_path, int(population))
    started = time.monotonic()
    deadline = started + float(time_budget_seconds)
    generation = 0
    try:
        while generation < int(max_generations):
            # Always evaluate one complete generation.  A later generation is
            # admitted only when the slowest of the three most recent
            # generations is expected to fit.  Checking merely that the
            # deadline has not passed can turn an IGA-180 run into 300+ s when
            # only a few seconds remain at a population boundary.
            if generation > 0:
                remaining = deadline - time.monotonic()
                estimate = max(generation_seconds[-3:])
                admission_estimates.append(float(estimate))
                if remaining < estimate:
                    stopped_with_seconds_remaining = max(0.0, float(remaining))
                    break
            specs = [
                {
                    "backends": BACKENDS,
                    "layout": layout,
                    "chromosome": chromosome,
                    "record_trace": False,
                }
                for chromosome in chromosomes
            ]
            generation_started = time.monotonic()
            episodes = batch.run(specs)
            generation_seconds.append(time.monotonic() - generation_started)
            fitness = np.full(int(population), 1e9, dtype=np.float64)
            completed_count = 0
            for index, episode in enumerate(episodes):
                completed = bool(episode["completed"])
                completed_count += int(completed)
                objective = (
                    float(episode["makespan"])
                    if completed
                    else 1e8 + float(episode["makespan"])
                )
                fitness[index] = objective
                evaluated += 1
                if objective < best_objective:
                    best_objective = objective
                    best_chromosome = chromosomes[index].copy()
            completed_per_generation.append(completed_count)
            generation += 1
            if generation < int(max_generations) and time.monotonic() < deadline:
                chromosomes = _next_population(rng, chromosomes, fitness)

        optimization_wall = time.monotonic() - started
        if best_chromosome is None or best_objective >= 1e8:
            raise RuntimeError(
                f"{case_path.name} produced no completed resource IGA candidate "
                f"after {evaluated} evaluations."
            )
        final = batch.run(
            [
                {
                    "backends": BACKENDS,
                    "layout": layout,
                    "chromosome": best_chromosome,
                    "record_trace": True,
                }
            ]
        )[0]
    finally:
        batch.close()

    if not final["completed"]:
        raise RuntimeError(
            f"Best resource IGA incumbent failed replay for {case_path.name}: "
            f"{final.get('error')}"
        )
    if not math.isclose(
        float(final["makespan"]),
        float(best_objective),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            f"Non-deterministic resource incumbent replay for {case_path.name}: "
            f"search={best_objective} replay={final['makespan']}"
        )

    search = {
        "search_contract_version": 2,
        "algorithm": (
            "real_coded_ga_sbx15_pm20_elitist_tournament2_batched_"
            "deadline_fit_v2"
        ),
        "population": int(population),
        "max_generations": int(max_generations),
        "completed_generations": int(generation),
        "seed": int(seed),
        "n_var": int(layout.n_var),
        "iga_device_count": len(layout.selected_device_indices),
        "evaluated_candidates": int(evaluated),
        "completed_candidates_per_generation": completed_per_generation,
        "generation_wall_seconds": [float(value) for value in generation_seconds],
        "generation_admission_estimate_seconds": admission_estimates,
        "stopped_with_seconds_remaining": float(stopped_with_seconds_remaining),
        "configured_additional_budget_seconds": float(time_budget_seconds),
        "nominal_cumulative_budget_seconds": float(cumulative_budget_seconds),
        "optimization_wall_seconds": float(optimization_wall),
        "budget_overshoot_seconds": float(
            max(0.0, optimization_wall - float(time_budget_seconds))
        ),
        "warm_start": dict(warm_start) if warm_start else None,
        "chromosome": best_chromosome.tolist(),
    }
    return search, final


class _AnytimeCaseGA:
    """One generational GA whose incumbent survives partial generations."""

    def __init__(
        self,
        layout: GenomeLayout,
        *,
        population: int,
        max_generations: int,
        seed: int,
        warm_chromosome: np.ndarray | None,
        warm_start: Mapping | None,
    ):
        self.layout = layout
        self.population_size = int(population)
        self.max_generations = int(max_generations)
        self.seed = int(seed)
        self.rng, self.population = _initial_population(
            layout, self.population_size, self.seed, warm_chromosome
        )
        self.fitness = np.full(self.population_size, 1e9, dtype=np.float64)
        self.generation = 0
        self.cursor = 0
        self.evaluated_candidates = 0
        self.completed_candidates = 0
        self.completed_per_generation: list[int] = []
        self._completed_this_generation = 0
        self.candidate_history: list[dict] = []
        self.best_objective = math.inf
        self.best_chromosome = None
        self.warm_start = dict(warm_start) if warm_start else None
        self.warm_objective: float | None = None
        self.warm_chromosome: np.ndarray | None = None
        self.replay_verification: dict | None = None
        self.inherited_candidates = 0
        if warm_chromosome is not None:
            source_makespan = float(self.warm_start["source_makespan"])
            if not math.isfinite(source_makespan):
                raise ValueError("Warm-start makespan must be finite.")
            self.warm_objective = source_makespan
            self.warm_chromosome = self.population[0].copy()
            self.fitness[0] = source_makespan
            self.best_objective = source_makespan
            self.best_chromosome = self.population[0].copy()
            self.cursor = 1
            self.inherited_candidates = 1

    @property
    def can_evaluate(self) -> bool:
        return self.generation < self.max_generations

    @property
    def candidate(self) -> np.ndarray:
        if not self.can_evaluate:
            raise RuntimeError("IGA has already reached max_generations.")
        return self.population[self.cursor]

    def observe(self, episode: Mapping) -> None:
        if not self.can_evaluate:
            raise RuntimeError("Cannot observe after GA termination.")
        index = int(self.cursor)
        completed = bool(episode.get("completed"))
        makespan = float(episode.get("makespan", 0.0))
        objective = makespan if completed else 1e8 + max(0.0, makespan)
        self.fitness[index] = objective
        self.evaluated_candidates += 1
        self.completed_candidates += int(completed)
        self._completed_this_generation += int(completed)
        self.candidate_history.append(
            {
                "generation": int(self.generation),
                "population_index": index,
                "completed": completed,
                "makespan": makespan,
                "error": episode.get("error"),
                "deadline_interrupted": bool(
                    episode.get("deadline_interrupted", False)
                ),
            }
        )
        if completed and objective < self.best_objective:
            self.best_objective = objective
            self.best_chromosome = self.population[index].copy()
        self.cursor += 1
        if self.cursor == self.population_size:
            self.completed_per_generation.append(
                int(self._completed_this_generation)
            )
            self.generation += 1
            if self.generation < self.max_generations:
                self.population = _next_population(
                    self.rng, self.population, self.fitness
                )
                self.fitness = np.full(
                    self.population_size, 1e9, dtype=np.float64
                )
                self.cursor = 0
                self._completed_this_generation = 0

    def reconcile_verified_replay(self, episode: Mapping) -> str:
        """Make the trace replay, not its screening score, authoritative.

        A frozen graph policy is mathematically case independent, but CUDA/PyG
        reductions and padded GEMM shapes can move a near-tied deterministic
        argmax when the other active cases in a cross-case batch change.  A
        small plane-action difference can then amplify into a different Cmax.

        Search episodes are therefore screening measurements.  The final
        trace replay is the verified measurement.  A verified warm incumbent
        remains the safety floor: if the replayed search incumbent is worse or
        incomplete, restore the already replay-verified warm chromosome.
        """

        searched_objective = float(self.best_objective)
        completed = bool(episode.get("completed"))
        replay_objective = (
            float(episode.get("makespan")) if completed else None
        )
        if replay_objective is not None and not math.isfinite(replay_objective):
            completed = False
            replay_objective = None

        matches_search = bool(
            completed
            and math.isclose(
                replay_objective,
                searched_objective,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        )
        if matches_search:
            status = "matched_search_incumbent"
            selected_source = "replayed_search_incumbent"
        elif (
            self.warm_chromosome is not None
            and self.warm_objective is not None
            and (
                replay_objective is None
                or self.warm_objective < replay_objective
            )
        ):
            self.best_objective = float(self.warm_objective)
            self.best_chromosome = self.warm_chromosome.copy()
            status = (
                "fallback_after_incomplete_replay"
                if replay_objective is None
                else "fallback_to_verified_warm_start"
            )
            selected_source = "verified_warm_start"
        elif completed:
            # The trace itself is complete and is the value consumed by BC.
            # Keep the searched chromosome but replace its screening score by
            # the replay-verified score.
            self.best_objective = float(replay_objective)
            status = "accepted_replay_adjustment"
            selected_source = "replayed_search_incumbent"
        else:
            raise RuntimeError(
                "Search incumbent replay was incomplete and no verified "
                "warm-start incumbent is available."
            )

        self.replay_verification = {
            "status": status,
            "search_screening_makespan": searched_objective,
            "trace_replay_makespan": replay_objective,
            "selected_makespan": float(self.best_objective),
            "selected_source": selected_source,
            "batch_context_sensitive": bool(not matches_search),
        }
        return selected_source

    def search_record(
        self,
        *,
        optimization_wall_seconds: float,
        time_budget_seconds: float,
        cumulative_budget_seconds: float,
        batch_wall_seconds: Sequence[float],
        parallel_case_count: int,
    ) -> dict:
        partial_evaluations = (
            int(self.cursor)
            if self.generation < self.max_generations
            else 0
        )
        if self.inherited_candidates and self.generation == 0:
            partial_evaluations = max(
                0, partial_evaluations - self.inherited_candidates
            )
        return {
            "search_contract_version": SEARCH_CONTRACT_VERSION,
            "algorithm": (
                "real_coded_ga_sbx15_pm20_elitist_tournament2_"
                "case_parallel_anytime_v3"
            ),
            "parallel_axis": "independent_cases",
            "gpu_inference": "one_cross_case_batch_per_step",
            "population": self.population_size,
            "max_generations": self.max_generations,
            "completed_generations": int(self.generation),
            "partial_generation_evaluations": partial_evaluations,
            "seed": self.seed,
            "n_var": int(self.layout.n_var),
            "iga_device_count": len(self.layout.selected_device_indices),
            "evaluated_candidates": int(self.evaluated_candidates),
            "inherited_verified_candidates": int(self.inherited_candidates),
            "completed_candidates": int(self.completed_candidates),
            "completed_candidates_per_generation": list(
                self.completed_per_generation
            ),
            "candidate_history": list(self.candidate_history),
            "cross_case_batch_wall_seconds": [
                float(value) for value in batch_wall_seconds
            ],
            "parallel_case_count": int(parallel_case_count),
            "configured_additional_budget_seconds": float(
                time_budget_seconds
            ),
            "nominal_cumulative_budget_seconds": float(
                cumulative_budget_seconds
            ),
            "optimization_wall_seconds": float(optimization_wall_seconds),
            "budget_overshoot_seconds": float(
                max(0.0, optimization_wall_seconds - time_budget_seconds)
            ),
            "incumbent_preserved": True,
            "replay_verification": dict(self.replay_verification or {}),
            "warm_start": self.warm_start,
            "chromosome": self.best_chromosome.tolist(),
        }


_VERIFIED_EPISODE_KEYS = (
    "completed",
    "makespan",
    "wall_seconds",
    "error",
    "completion",
    "decision_trace",
    "plane_trajectory",
    "resource_trajectory",
    "resource_decision_log",
    "batch_step_count",
    "batch_policy_forwards",
    "batch_policy_rows",
    "deadline_interrupted",
)


def _load_verified_warm_episode(
    state: _AnytimeCaseGA,
) -> dict:
    """Load only evaluator-result fields from a verified warm teacher."""

    evidence = state.warm_start
    if (
        not evidence
        or state.warm_chromosome is None
        or state.warm_objective is None
    ):
        raise RuntimeError("Verified warm replay requested without evidence.")
    source = Path(str(evidence["path"]))
    if not source.is_file() or _sha256_file(source) != evidence.get("sha256"):
        raise RuntimeError(f"Verified warm teacher changed on disk: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    source_chromosome = np.asarray(
        payload.get("search", {}).get("chromosome", []), dtype=np.float64
    )
    if (
        not payload.get("completion_verified")
        or not payload.get("completed")
        or not payload.get("completion", {}).get("completed")
        or not math.isclose(
            float(payload.get("makespan", math.inf)),
            float(state.warm_objective),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not np.array_equal(source_chromosome, state.warm_chromosome)
    ):
        raise RuntimeError(f"Invalid verified warm teacher evidence: {source}")
    missing = [
        key
        for key in (
            "completed",
            "makespan",
            "completion",
            "decision_trace",
            "plane_trajectory",
            "resource_trajectory",
            "resource_decision_log",
        )
        if key not in payload
    ]
    if missing:
        raise RuntimeError(
            f"Verified warm teacher lacks replay fields {missing}: {source}"
        )
    episode = {
        key: payload[key]
        for key in _VERIFIED_EPISODE_KEYS
        if key in payload
    }
    episode["reused_verified_warm_start"] = True
    return episode


def optimize_teachers_case_parallel(
    evaluator: FrozenPlaneEvaluator,
    case_paths: Sequence[Path],
    *,
    population: int,
    max_generations: int,
    time_budget_seconds: float,
    cumulative_budget_seconds: float,
    base_seed: int,
    warm_starts: Mapping[str, tuple[np.ndarray | None, Mapping | None]],
) -> dict[str, tuple[dict, dict]]:
    """Run one anytime IGA per case with one shared frozen-policy GPU batch."""

    case_paths = tuple(Path(path) for path in case_paths)
    states: dict[str, _AnytimeCaseGA] = {}
    for case_path in case_paths:
        layout = evaluator.build_layout(case_path, BACKENDS)
        if not layout.selected_device_indices:
            raise RuntimeError(
                f"{case_path.name} contains no IGA-controlled devices."
            )
        warm_chromosome, warm_evidence = warm_starts[case_path.name]
        states[case_path.name] = _AnytimeCaseGA(
            layout,
            population=population,
            max_generations=max_generations,
            seed=_derived_seed(
                base_seed,
                case_path.name,
                f"stage2_iga_{int(cumulative_budget_seconds)}",
            ),
            warm_chromosome=warm_chromosome,
            warm_start=warm_evidence,
        )

    batch = CrossCaseBatchedEvaluator(evaluator, case_paths)
    started = time.monotonic()
    deadline = started + float(time_budget_seconds)
    batch_walls: list[float] = []
    try:
        while all(state.can_evaluate for state in states.values()):
            # The first IGA-180 heuristic must finish once so every case has a
            # feasible incumbent.  All later candidates obey the same
            # step-level deadline used by the Stage-1 IGA implementation.
            has_incumbents = all(
                state.best_chromosome is not None for state in states.values()
            )
            if has_incumbents and time.monotonic() >= deadline:
                break
            specs = [
                {
                    "backends": BACKENDS,
                    "layout": states[path.name].layout,
                    "chromosome": states[path.name].candidate,
                    "record_trace": False,
                }
                for path in case_paths
            ]
            episodes = batch.run(
                specs,
                deadline=deadline,
                abort_at_deadline=has_incumbents,
            )
            batch_walls.append(float(episodes[0]["wall_seconds"]))
            for path, episode in zip(case_paths, episodes):
                states[path.name].observe(episode)
            if any(
                episode.get("deadline_interrupted", False)
                for episode in episodes
            ):
                break
            if time.monotonic() >= deadline:
                break

        optimization_wall = time.monotonic() - started
        missing = [
            case for case, state in states.items()
            if state.best_chromosome is None
        ]
        if missing:
            raise RuntimeError(
                "No completed case-parallel IGA incumbent for: "
                + ", ".join(missing[:5])
            )
        finals = batch.run(
            [
                {
                    "backends": BACKENDS,
                    "layout": states[path.name].layout,
                    "chromosome": states[path.name].best_chromosome,
                    "record_trace": True,
                }
                for path in case_paths
            ]
        )
    finally:
        batch.close()

    verified_finals = []
    for case_path, final in zip(case_paths, finals):
        state = states[case_path.name]
        try:
            selected_source = state.reconcile_verified_replay(final)
        except RuntimeError as error:
            raise RuntimeError(
                f"Best resource IGA incumbent failed replay for "
                f"{case_path.name}: {final.get('error')}"
            ) from error
        if selected_source == "verified_warm_start":
            final = _load_verified_warm_episode(state)
        if not final["completed"]:
            raise RuntimeError(
                f"Best resource IGA incumbent failed replay for "
                f"{case_path.name}: {final.get('error')}"
            )
        if not math.isclose(
            float(final["makespan"]),
            float(state.best_objective),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(
                f"Internal replay reconciliation failure for "
                f"{case_path.name}: selected={state.best_objective} "
                f"replay={final['makespan']}"
            )
        verified_finals.append(final)

    results = {}
    for case_path, final in zip(case_paths, verified_finals):
        state = states[case_path.name]
        search = state.search_record(
            optimization_wall_seconds=optimization_wall,
            time_budget_seconds=time_budget_seconds,
            cumulative_budget_seconds=cumulative_budget_seconds,
            batch_wall_seconds=batch_walls,
            parallel_case_count=len(case_paths),
        )
        results[case_path.name] = (search, final)
    return results


def _load_warm_start(
    warm_start_dir: Path | None,
    case: str,
    *,
    checkpoint_sha: str,
    command_sha: str,
    command_key: str,
    case_sha: str | None,
    expected_budget: float,
) -> tuple[np.ndarray | None, dict | None]:
    if warm_start_dir is None:
        return None, None
    source = warm_start_dir / "teachers" / f"{case}.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing required IGA warm-start teacher: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    required = {
        "status": "completed",
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "frozen_plane_checkpoint_sha256": checkpoint_sha,
        "frozen_plane_source_command_sha256": command_sha,
        "frozen_plane_source_command_key": command_key,
        "case": case,
    }
    mismatched = {
        key: {"observed": payload.get(key), "expected": value}
        for key, value in required.items()
        if payload.get(key) != value
    }
    if mismatched:
        raise ValueError(f"Incompatible warm-start teacher {source}: {mismatched}")
    if case_sha and payload.get("case_sha256") != case_sha:
        raise ValueError(f"Warm-start case hash mismatch: {source}")
    if not payload.get("completion_verified"):
        raise ValueError(f"Warm-start teacher is not replay verified: {source}")
    source_budget = float(
        payload.get("search", {}).get("nominal_cumulative_budget_seconds", -1.0)
    )
    if not math.isclose(source_budget, float(expected_budget), abs_tol=1e-9):
        raise ValueError(
            f"Warm-start budget mismatch in {source}: "
            f"{source_budget} != {expected_budget}"
        )
    chromosome = payload.get("search", {}).get("chromosome")
    if not isinstance(chromosome, list) or not chromosome:
        raise ValueError(f"Warm-start teacher has no chromosome: {source}")
    return np.asarray(chromosome, dtype=np.float64), {
        "path": str(source.resolve()),
        "sha256": _sha256_file(source),
        "source_nominal_budget_seconds": source_budget,
        "source_makespan": float(payload["makespan"]),
    }


def _teacher_contract(
    *,
    case: str,
    case_sha: str | None,
    checkpoint_sha: str,
    command_sha: str,
    command_key: str,
    cumulative_budget: float,
) -> dict:
    return {
        "status": "completed",
        "schema_version": TEACHER_SCHEMA_VERSION,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "search_contract_version": SEARCH_CONTRACT_VERSION,
        "resource_policy": "drl",
        "arm": "iga_all",
        "backends": BACKENDS,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "case": case,
        "case_sha256": case_sha,
        "frozen_plane_checkpoint_sha256": checkpoint_sha,
        "frozen_plane_source_command_sha256": command_sha,
        "frozen_plane_source_command_key": command_key,
        "nominal_cumulative_budget_seconds": float(cumulative_budget),
        "completion_verified": True,
    }


def _teacher_is_reusable(path: Path, expected: Mapping) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if any(payload.get(key) != value for key, value in expected.items()):
        return False
    completion = payload.get("completion", {})
    return bool(
        payload.get("completion_verified")
        and payload.get("completed")
        and completion.get("completed")
        and isinstance(payload.get("decision_trace"), list)
        and payload.get("decision_trace")
        and isinstance(payload.get("search", {}).get("chromosome"), list)
    )


def _validate_semantic_chain(handoff: Mapping, policy_args, checkpoint: Mapping) -> dict:
    expected = {
        "plane_order_mode": str(policy_args.plane_order_mode),
        "plane_pair_decoder": str(policy_args.plane_pair_decoder),
        "global_feature_mode": str(policy_args.global_feature_mode),
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
    }
    observed = handoff.get("semantic_contract", {})
    mismatched = {
        key: {"handoff": observed.get(key), "runtime": value}
        for key, value in expected.items()
        if str(observed.get(key)) != value
    }
    validation = checkpoint.get("_resource_iga_validation")
    if mismatched or not isinstance(validation, Mapping):
        raise ValueError(
            f"Frozen plane semantic chain validation failed: {mismatched}"
        )
    observation = validation.get("observation_contract", {})
    handoff_validation = validation.get("handoff_contract", {})
    source_summary = handoff_validation.get("source_summary", {})
    loaded_summary = validation.get("loaded_protected_summary", {})
    return {
        "runtime": expected,
        "checkpoint_validation": {
            "strict_metadata": bool(observation.get("strict_metadata")),
            "observation_schema_id": observation.get("observation_schema_id"),
            "environment_semantics_version": observation.get(
                "environment_semantics_version"
            ),
            "protected_parameter_count": source_summary.get("count"),
            "protected_parameter_numel": source_summary.get("numel"),
            "protected_parameter_sha256": source_summary.get("sha256"),
            "loaded_protected_parameter_sha256": loaded_summary.get("sha256"),
        },
    }


def run_shard(args) -> None:
    checkpoint_path = args.checkpoint.resolve()
    command_path = args.source_command.resolve()
    handoff = _validate_stage1_handoff(
        args.handoff.resolve(),
        checkpoint_path,
        command_path,
        args.source_command_key,
    )
    command_key = str(handoff.get("source_command_key") or "")
    if not command_key:
        raise ValueError("Authoritative Stage1 handoff has no source_command_key.")
    all_cases = list_case_folders(str(args.dataset_dir), args.max_cases)
    cases = [
        case
        for index, case in enumerate(all_cases)
        if index % args.shard_count == args.shard_index
    ]
    device = torch.device(args.device)
    policy, policy_args, checkpoint = load_frozen_policy(
        checkpoint_path,
        command_path,
        device,
        args.evaluation_tau,
        command_key,
    )
    semantic_chain = _validate_semantic_chain(handoff, policy_args, checkpoint)
    evaluator = FrozenPlaneEvaluator(
        policy,
        policy_args,
        max_steps=args.max_steps,
        device_lookahead_dispatch=True,
        device_lookahead_safety_margin=args.device_lookahead_safety_margin,
    )
    digest_before = _model_digest(policy)
    checkpoint_sha = _sha256_file(checkpoint_path)
    command_sha = _sha256_file(command_path)
    output_dir = args.output_dir.resolve()
    shard_started = time.monotonic()
    print(
        f"[Stage2IGA] shard={args.shard_index}/{args.shard_count} "
        f"device={device} cases={len(cases)} case_batch={args.case_batch_size} "
        f"population={args.population} "
        f"additional_budget={args.time_budget_seconds:.0f}s "
        f"cumulative_budget={args.cumulative_budget_seconds:.0f}s "
        f"model={digest_before[:12]}",
        flush=True,
    )

    completed_cases = []
    pending = []
    entries = {}
    for case in cases:
        case_path = args.dataset_dir / case
        metadata = _metadata(case_path)
        teacher_path = output_dir / "teachers" / f"{case}.json"
        result_path = output_dir / "cases" / f"{case}.json"
        expected = _teacher_contract(
            case=case,
            case_sha=metadata.get("case_sha256"),
            checkpoint_sha=checkpoint_sha,
            command_sha=command_sha,
            command_key=command_key,
            cumulative_budget=args.cumulative_budget_seconds,
        )
        if _teacher_is_reusable(teacher_path, expected) and result_path.is_file():
            completed_cases.append(case)
            print(
                f"[Stage2IGA] shard={args.shard_index} {case} reusable "
                f"({len(completed_cases)}/{len(cases)})",
                flush=True,
            )
            continue
        with (case_path / "flights.json").open("r", encoding="utf-8") as source:
            plane_count = len(json.load(source))
        entries[case] = {
            "path": case_path,
            "metadata": metadata,
            "teacher_path": teacher_path,
            "result_path": result_path,
            "expected": expected,
            "plane_count": plane_count,
        }
        pending.append(case)

    # Similar-sized cases share a batch so terminal padding is small.  The
    # original shard assignment remains unchanged, preserving deterministic
    # resume and keeping both GPUs balanced over the complete dataset.
    pending.sort(key=lambda case: (-entries[case]["plane_count"], case))
    waves = [
        pending[index : index + args.case_batch_size]
        for index in range(0, len(pending), args.case_batch_size)
    ]
    for wave_index, wave in enumerate(waves, start=1):
        wave_started = time.monotonic()
        warm_starts = {}
        for case in wave:
            entry = entries[case]
            warm_starts[case] = _load_warm_start(
                args.warm_start_dir,
                case,
                checkpoint_sha=checkpoint_sha,
                command_sha=command_sha,
                command_key=command_key,
                case_sha=entry["metadata"].get("case_sha256"),
                expected_budget=args.warm_start_budget_seconds,
            )
        print(
            f"[Stage2IGA] shard={args.shard_index} wave="
            f"{wave_index}/{len(waves)} cases={len(wave)} "
            f"planes={entries[wave[-1]]['plane_count']}-"
            f"{entries[wave[0]]['plane_count']}",
            flush=True,
        )
        wave_results = optimize_teachers_case_parallel(
            evaluator,
            [entries[case]["path"] for case in wave],
            population=args.population,
            max_generations=args.max_generations,
            time_budget_seconds=args.time_budget_seconds,
            cumulative_budget_seconds=args.cumulative_budget_seconds,
            base_seed=args.seed,
            warm_starts=warm_starts,
        )
        wave_wall = float(time.monotonic() - wave_started)
        for case in wave:
            entry = entries[case]
            metadata = entry["metadata"]
            search, final = wave_results[case]
            teacher = {
                **entry["expected"],
                "dataset_dir": str(args.dataset_dir.resolve()),
                "case_id": metadata.get("case_id"),
                "profile": metadata.get("profile"),
                "distribution": metadata.get("distribution"),
                "frozen_plane_checkpoint": str(checkpoint_path),
                "frozen_plane_source_command": str(command_path),
                "stage1_handoff": handoff,
                "semantic_chain": semantic_chain,
                "frozen_model_digest": digest_before,
                "frozen_model_unchanged": True,
                "plane_actions_deterministic": True,
                "evaluation_tau": float(args.evaluation_tau),
                "device_lookahead_dispatch": True,
                "device_lookahead_safety_margin": float(
                    args.device_lookahead_safety_margin
                ),
                "search": search,
                **final,
                "completion_verified": True,
                "generated_unix_time": time.time(),
            }
            _atomic_json(entry["teacher_path"], teacher)
            _atomic_json(
                entry["result_path"],
                {
                    **entry["expected"],
                    "case_id": metadata.get("case_id"),
                    "profile": metadata.get("profile"),
                    "distribution": metadata.get("distribution"),
                    "makespan": float(final["makespan"]),
                    "completion": final["completion"],
                    "search": {
                        key: value
                        for key, value in search.items()
                        if key != "chromosome"
                    },
                    "teacher": str(entry["teacher_path"]),
                    "case_wall_seconds": wave_wall,
                    "case_parallel_wave": int(wave_index),
                },
            )
            completed_cases.append(case)
            print(
                f"[Stage2IGA] shard={args.shard_index} {case} "
                f"cmax={final['makespan']:.1f} generations="
                f"{search['completed_generations']} evals="
                f"{search['evaluated_candidates']} search_wall="
                f"{search['optimization_wall_seconds']:.1f}s "
                f"({len(completed_cases)}/{len(cases)})",
                flush=True,
            )

    digest_after = _model_digest(policy)
    if digest_after != digest_before:
        raise RuntimeError("Frozen plane/model parameters changed during Stage2 IGA.")
    _atomic_json(
        output_dir / "workers" / f"shard_{args.shard_index:02d}.json",
        {
            "status": "completed",
            "schema_version": 1,
            "teacher_scope": TEACHER_SCOPE,
            "teacher_method": TEACHER_METHOD,
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "device": str(device),
            "population": int(args.population),
            "case_batch_size": int(args.case_batch_size),
            "parallel_axis": "independent_cases",
            "configured_additional_budget_seconds": float(
                args.time_budget_seconds
            ),
            "nominal_cumulative_budget_seconds": float(
                args.cumulative_budget_seconds
            ),
            "cases": completed_cases,
            "case_count": len(completed_cases),
            "model_digest_before": digest_before,
            "model_digest_after": digest_after,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "source_command": str(command_path),
            "source_command_key": command_key,
            "source_command_sha256": command_sha,
            "stage1_handoff": handoff,
            "semantic_chain": semantic_chain,
            "shard_wall_seconds": float(time.monotonic() - shard_started),
            "completed_unix_time": time.time(),
        },
    )


def summarize(args) -> dict:
    cases = list_case_folders(str(args.dataset_dir), args.max_cases)
    records = []
    missing = []
    for case in cases:
        path = args.output_dir / "cases" / f"{case}.json"
        teacher = args.output_dir / "teachers" / f"{case}.json"
        if not path.is_file() or not teacher.is_file():
            missing.append(case)
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        teacher_payload = json.loads(teacher.read_text(encoding="utf-8"))
        if (
            record.get("status") != "completed"
            or teacher_payload.get("status") != "completed"
            or not teacher_payload.get("completion_verified")
            or teacher_payload.get("teacher_scope") != TEACHER_SCOPE
            or teacher_payload.get("search_contract_version")
            != SEARCH_CONTRACT_VERSION
            or teacher_payload.get("environment_semantics_version")
            != AircraftScheduleEnv.SEMANTICS_VERSION
            or not teacher_payload.get("decision_trace")
        ):
            raise RuntimeError(f"Invalid Stage2 IGA teacher for {case}.")
        records.append(record)
    if missing:
        raise RuntimeError(
            f"Cannot summarize: {len(missing)} cases are missing; first={missing[0]}"
        )

    makespans = [float(record["makespan"]) for record in records]
    search_walls = [
        float(record["search"]["optimization_wall_seconds"])
        for record in records
    ]
    payload = {
        "status": "completed",
        "schema_version": 1,
        "teacher_scope": TEACHER_SCOPE,
        "teacher_method": TEACHER_METHOD,
        "resource_policy": "drl",
        "backends": BACKENDS,
        "environment_semantics_version": AircraftScheduleEnv.SEMANTICS_VERSION,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "case_count": len(records),
        "completed_count": len(records),
        "replay_verified_count": len(records),
        "mean_makespan": statistics.mean(makespans),
        "std_makespan": statistics.pstdev(makespans),
        "median_makespan": statistics.median(makespans),
        "mean_optimization_wall_seconds": statistics.mean(search_walls),
        "max_optimization_wall_seconds": max(search_walls),
        "total_evaluated_candidates": sum(
            int(record["search"]["evaluated_candidates"])
            for record in records
        ),
        "nominal_cumulative_budget_seconds": float(
            args.cumulative_budget_seconds
        ),
        "completed_unix_time": time.time(),
    }
    _atomic_json(args.output_dir / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "onpolicy/envs/HKBZ/dataset/fjsp_v3_t600_v120_test60/train",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "onpolicy/scripts/results/HKBZ/simple/gnn_mappo/"
        "stage1_departure_reward_formal_dual_20260816_r1_formal_"
        "P5_team_time_potential_fixed_seed3/run1/models/checkpoint_Best.pt",
    )
    parser.add_argument(
        "--source-command",
        type=Path,
        default=ROOT
        / "result/hkbz_train_logs/stage1_departure_reward_formal_dual_"
        "20260816_r1/commands/formal_P5.json",
    )
    parser.add_argument(
        "--source-command-key",
        default="P5_team_time_potential_fixed_seed3",
    )
    parser.add_argument(
        "--handoff",
        type=Path,
        default=ROOT / "onpolicy/config/stage1_m2_handoff.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-tau", type=float, default=0.3)
    parser.add_argument("--device-lookahead-safety-margin", type=float, default=60.0)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument(
        "--case-batch-size",
        type=int,
        default=72,
        help="independent cases advanced together by one shared GPU policy",
    )
    parser.add_argument("--max-generations", type=int, default=100000)
    parser.add_argument("--time-budget-seconds", type=float, required=True)
    parser.add_argument("--cumulative-budget-seconds", type=float, required=True)
    parser.add_argument("--warm-start-dir", type=Path, default=None)
    parser.add_argument("--warm-start-budget-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.warm_start_dir is not None:
        args.warm_start_dir = args.warm_start_dir.resolve()
    if args.population < 2:
        parser.error("population must be >= 2")
    if args.case_batch_size < 1:
        parser.error("case-batch-size must be positive")
    if args.max_generations < 1:
        parser.error("max-generations must be positive")
    if args.time_budget_seconds <= 0.0:
        parser.error("time-budget-seconds must be positive")
    if args.cumulative_budget_seconds < args.time_budget_seconds:
        parser.error("cumulative budget cannot be smaller than additional budget")
    if args.warm_start_dir is None and args.warm_start_budget_seconds != 0.0:
        parser.error("warm-start-budget requires --warm-start-dir")
    if args.warm_start_dir is not None:
        expected = args.time_budget_seconds + args.warm_start_budget_seconds
        if not math.isclose(
            args.cumulative_budget_seconds, expected, abs_tol=1e-9
        ):
            parser.error(
                "cumulative budget must equal warm-start plus additional budget"
            )
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    if args.device_lookahead_safety_margin < 0.0:
        parser.error("device-lookahead-safety-margin must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summarize_only:
        summarize(args)
        return 0
    mp.set_start_method("spawn", force=True)
    run_shard(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
