"""Opt-in research engine with explicit representation checkpoint identity."""
from pathlib import Path
import torch
from onpolicy.runner.shared.stage3_research_engine import ResearchEngine
from onpolicy.algorithms.utils.stage3_encoder import Stage3Encoder, FullStage3Encoder, install_encoder
from onpolicy.utils.stage3_research import SOURCE


class RepresentationEngine(ResearchEngine):
    def __init__(self, variant, checkpoint=None, *, source=SOURCE, performance=None, **kwargs):
        self.representation = variant
        self._representation_ready = False
        kwargs.pop("freeze_shared", None)
        super().__init__(checkpoint=source, freeze_shared=variant == "E0", **kwargs)
        self.representation_report = install_encoder(self.policy, variant)
        self._representation_ready = True
        self.configure_exploration(self.exploration)
        self.policy.ac.eval()
        if checkpoint is not None and Path(checkpoint).resolve() != Path(source).resolve():
            self.load(checkpoint)
        self.assert_optimizer_ownership()
        self.performance = dict(performance or {})
        self.defer_statistics = bool(self.performance.get("defer_statistics", False))
        if self.performance.get("disable_activation_checkpoint", False):
            self.policy.ac.shared_encoder_activation_checkpoint = False
            self.args.shared_encoder_activation_checkpoint = False
        self.execution_cache = None
        frozen_cache = self.performance.get("cache_frozen_features", False)
        if frozen_cache or self.performance.get("cache_inputs", False):
            if frozen_cache and isinstance(self.policy.ac.encoder, FullStage3Encoder):
                raise ValueError("Full-depth trainable encoders cannot cache detached graph features")
            from onpolicy.utils.stage3_performance import GroupExecutionCache
            self.execution_cache = GroupExecutionCache(self.policy, self.performance.get("cache_mib", 1024),
                                                       frozen_features=frozen_cache)
            self.policy.stage3_execution_cache = self.execution_cache
            self.policy.ac.stage3_execution_cache = self.execution_cache

    def configure_exploration(self, config):
        super().configure_exploration(config)
        if isinstance(self.policy.ac.encoder, (Stage3Encoder, FullStage3Encoder)):
            self.policy.ac.encoder.set_trainability()

    def load(self, checkpoint):
        if self._representation_ready:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("representation") != self.representation:
                raise ValueError("Checkpoint representation differs; migrate original C0 at construction only")
            del payload
        super().load(checkpoint)

    def save(self, path, **metadata):
        return super().save(path, representation=self.representation,
                            representation_report=self.representation_report, **metadata)

    def assert_optimizer_ownership(self):
        actor = [p for group in self.policy.actor_optimizer.param_groups for p in group["params"]]
        critic = [p for group in self.policy.critic_optimizer.param_groups for p in group["params"]]
        if len({id(p) for p in actor}) != len(actor) or {id(p) for p in actor} & {id(p) for p in critic}:
            raise RuntimeError("Duplicate actor/critic optimizer ownership")
        owned = {id(p) for p in actor + critic}
        if any(id(p) not in owned for p in self.policy.ac.parameters() if p.requires_grad):
            raise RuntimeError("Trainable parameter has no optimizer")
        pointers = [p.data_ptr() for p in self.policy.ac.encoder.parameters()]
        if len(pointers) != len(set(pointers)):
            raise RuntimeError("Private encoder parameter storage aliases")

    def update(self, trajectories, mode, source_cost=None, **kwargs):
        kwargs.setdefault("allow_multi_case", True)
        if self.execution_cache is None:
            return super().update(trajectories, mode, source_cost, **kwargs)
        with self.execution_cache.group():
            result = super().update(trajectories, mode, source_cost, **kwargs)
        result["execution_cache"] = self.execution_cache.last_report
        return result
