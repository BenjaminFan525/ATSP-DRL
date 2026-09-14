"""Function-preserving graph-tail interventions, opt-in for the Stage3 study.

E0: frozen shared graph; E1: shared trainable tail; E2: role-private tails;
E3: shared tail with zero-output residual capacity matched to E2. All role
decisions still execute in the existing single autoregressive decoder.
"""
from copy import deepcopy

import torch
from torch import nn
from onpolicy.algorithms.utils.gnn import HeteroGraphEncoder


class ZeroResidual(nn.Module):
    def __init__(self, dim, width, reference):
        super().__init__()
        kw = {"device": reference.device, "dtype": reference.dtype}
        self.input = nn.Linear(dim, width, **kw)
        self.output = nn.Linear(width, dim, **kw)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, value):
        return value + self.output(torch.tanh(self.input(value)))


class Stage3Encoder(nn.Module):
    variants = ("E0", "E1", "E2", "E3")
    outputs = ("global_emb", "op_nodes", "site_nodes", "device_nodes", "request_nodes")

    def __init__(self, original, variant):
        super().__init__()
        if type(original) is not HeteroGraphEncoder or variant not in self.variants:
            raise ValueError("Stage3 tails require the original heterogeneous graph encoder")
        if len(original.convs) < 2:
            raise ValueError("A frozen lower graph block is required")
        self.variant = variant
        # Deep copies share no Parameter/storage with C0 or other role tails.
        self.prefix = deepcopy(original)
        template = deepcopy(original)
        for name in ("op_embedding", "site_embedding", "dev_embedding", "req_embedding"):
            delattr(template, name)
        for name in ("convs", "norms_op", "norms_site", "norms_dev", "norms_req"):
            setattr(template, name, nn.ModuleList([getattr(template, name)[-1]]))
            setattr(self.prefix, name, nn.ModuleList(list(getattr(self.prefix, name))[:-1]))
        for name in ("global_embedding", "global_context_embedding", "global_context_scale"):
            delattr(self.prefix, name)
        self.tails = nn.ModuleList([deepcopy(template) for _ in range(3 if variant == "E2" else 1)])
        self.residuals = nn.ModuleDict()
        self.tail_parameters = sum(p.numel() for p in template.parameters())
        self.residual_width = 0
        if variant == "E3":
            dim = original.embed_dim
            # Five independent residual MLPs: count = 5 * ((2*d+1)*h+d).
            self.residual_width = max(1, round((2*self.tail_parameters/5 - dim)/(2*dim+1)))
            ref = next(original.parameters())
            # Stable and local initialization: never consume the training RNG.
            with torch.random.fork_rng(devices=[ref.device.index] if ref.is_cuda else []):
                torch.manual_seed(2026090801)
                self.residuals.update({key: ZeroResidual(dim, self.residual_width, ref)
                                       for key in self.outputs})
        self.set_trainability()

    def set_trainability(self):
        self.prefix.requires_grad_(False)
        self.tails.requires_grad_(self.variant != "E0")
        self.residuals.requires_grad_(True)

    def encode_prefix(self, data):
        with torch.no_grad():
            nodes, edges, attributes = self.prefix._prepare_graph(data)
            nodes = self.prefix._message_pass(nodes, edges, attributes)
        return nodes, edges, attributes

    def encode_tail(self, data, prefix):
        nodes, edges, attributes = prefix
        outputs = [tail._readout(data, tail._message_pass(nodes, edges, attributes))
                   for tail in self.tails]
        if self.residuals:
            outputs[0] = {key: self.residuals[key](value) if key in self.residuals else value
                          for key, value in outputs[0].items()}
        result = dict(outputs[0])
        if self.variant == "E2":
            result["role_encodings"] = {str(role): value for role, value in enumerate(outputs)}
        return result

    def forward(self, data):
        return self.encode_tail(data, self.encode_prefix(data))

    def report(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        expected_e2 = 3*self.tail_parameters
        return {"variant": self.variant, "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": trainable, "tail_parameters": self.tail_parameters,
                "residual_width": self.residual_width,
                "e3_vs_e2_trainable_relative_error": abs(trainable/expected_e2 - 1) if self.variant == "E3" else None}


class FullStage3Encoder(nn.Module):
    """C0-preserving full-depth controls, without a detached/frozen prefix.

    F_SHARED trains one entire graph encoder; F_PRIVATE trains three disjoint
    entire encoders. Role routing and the joint autoregressive decoder are
    unchanged. No trained feature is eligible for the frozen-prefix cache.
    """
    variants = ("F_SHARED", "F_PRIVATE")

    def __init__(self, original, variant):
        super().__init__()
        if type(original) is not HeteroGraphEncoder or variant not in self.variants:
            raise ValueError("Full-depth study requires the original C0 graph encoder")
        self.variant = variant
        self.encoders = nn.ModuleList([
            deepcopy(original) for _ in range(3 if variant == "F_PRIVATE" else 1)
        ])
        self.set_trainability()

    def set_trainability(self):
        self.encoders.requires_grad_(True)

    def forward(self, data):
        outputs = [encoder(data) for encoder in self.encoders]
        result = dict(outputs[0])
        if self.variant == "F_PRIVATE":
            result["role_encodings"] = {str(role): value for role, value in enumerate(outputs)}
        return result

    def report(self):
        return {"variant": self.variant,
                "total_parameters": sum(p.numel() for p in self.parameters()),
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "full_encoder_copies": len(self.encoders), "frozen_prefix": False,
                "e3_vs_e2_trainable_relative_error": None}


def install_encoder(policy, variant):
    """Invoke only AFTER loading original C0, then rebuild optimizer ownership."""
    from onpolicy.algorithms.gnn_mappo.algorithm.gnn_actor_critic import _ModuleGroup
    ac = policy.ac
    if (ac.resource_residual_adapter is not None or ac.plane_order_actor is not None
            or ac.device_policy_head_mode != "shared" or ac.request_ready_prediction):
        raise ValueError("Representation study is restricted to the frozen C0 head contract")
    encoder_class = FullStage3Encoder if variant in FullStage3Encoder.variants else Stage3Encoder
    ac.encoder = encoder_class(ac.encoder, variant)
    ac.shared_actor_param = _ModuleGroup([ac.encoder])
    policy.reset_optimizers()
    return ac.encoder.report()
