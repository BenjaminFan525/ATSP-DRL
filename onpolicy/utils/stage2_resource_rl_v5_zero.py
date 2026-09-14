"""Numerically matched, strictly no-grad B0 *initialization audit* only.

PyTorch 2.6 matmul's should_fold checks the matrix's requires_grad even inside
no_grad. A strided MHA K/V projection therefore uses mm for the trainable actor
but bmm for the frozen reference. Equal FP32 weights need not give bit-identical
probabilities across those kernels. Align only resource-actor metadata during
the audit; never alter actor behavior, PPO/reference-KL, weights, or tolerances.
"""
from contextlib import contextmanager
import ast
import hashlib

import torch

from onpolicy.utils.stage2_resource_rl import RESOURCE_PREFIXES

VERSION = 'v5_initial_audit_matched_autograd_metadata_v1'


def non_audit_code_sha(source):
    """Prove the old full engineering/learning path is unaffected by this fix."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.module == 'onpolicy.utils.stage2_resource_rl_v5_zero')]
    matches = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'SplitEncoderLearner':
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name == 'collect_forward':
                    kept = []
                    for statement in method.body:
                        if isinstance(statement, ast.If) and ast.unparse(statement.test) == 'behavior and self.verify_initial_reference':
                            matches += 1
                        else:
                            kept.append(statement)
                    method.body = kept
    if matches != 1:
        raise ValueError('Exactly one isolated initial-reference audit block is required.')
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@contextmanager
def initial_reference_kernel_context(policy):
    reference = policy.bc_reference_ac
    if torch.is_grad_enabled():
        raise RuntimeError('Initial reference kernel matching requires torch.no_grad().')
    if reference is None or policy.ac.training or reference.training:
        raise RuntimeError('Initial reference kernel matching requires two eval-mode models.')
    reference_params = dict(reference.named_parameters())
    current_params = dict(policy.ac.named_parameters())
    if any(p.requires_grad or p.grad is not None for p in reference_params.values()):
        raise RuntimeError('B0 reference must be fully frozen with no gradients on audit entry.')
    if not set(reference_params).issubset(current_params):
        raise ValueError('Missing B0 reference parameters in current architecture.')
    changed = []
    try:
        for name, p in reference_params.items():
            if name.startswith(RESOURCE_PREFIXES) and current_params[name].requires_grad:
                p.requires_grad_(True)
                changed.append(p)
        yield
    finally:
        for p in changed:
            p.requires_grad_(False)
        if any(p.requires_grad or p.grad is not None for p in reference_params.values()):
            raise RuntimeError('B0 reference did not return to its fully frozen state.')
