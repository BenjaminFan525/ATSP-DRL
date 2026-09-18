"""Require effective memory settings, not just an unenforced manifest flag."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from onpolicy.utils.stage3_memory_runtime import verify_unprotected_memory


def cgroups(tmp_path, monkeypatch, *, uid=0, omit=b'1'):
    root = tmp_path/'cgroup'
    parent = root/'system.slice'
    leaf = parent/'experiment.service'
    leaf.mkdir(parents=True)
    for group in (parent, leaf):
        for name in ('memory.high','memory.max','memory.swap.max'):
            (group/name).write_text('max\n')
    original = Path.stat
    monkeypatch.setattr(Path, 'stat', lambda self, *a, **kw:
                        SimpleNamespace(st_uid=uid) if self == leaf else original(self, *a, **kw))
    monkeypatch.setattr('os.getxattr', lambda *_: omit)
    return root, parent, leaf


def test_effective_root_owned_oomd_exemption_and_unlimited_ancestors(tmp_path, monkeypatch):
    root, _, leaf = cgroups(tmp_path, monkeypatch)
    result = verify_unprotected_memory(cgroup_root=root,
                                       membership='0::/system.slice/experiment.service\n')
    assert result['cgroup'] == str(leaf) and result['oomd_omit'] == '1'
    assert len(result['limits']) == 2
    assert result['kernel_global_oom_behavior'] == 'unchanged'


@pytest.mark.parametrize('uid,omit', [(1001,b'1'), (0,b'0')])
def test_user_owned_or_missing_effective_omit_is_rejected(tmp_path, monkeypatch, uid, omit):
    root, _, _ = cgroups(tmp_path, monkeypatch, uid=uid, omit=omit)
    with pytest.raises(RuntimeError):
        verify_unprotected_memory(cgroup_root=root,
                                  membership='0::/system.slice/experiment.service\n')


@pytest.mark.parametrize('name', ['memory.high','memory.max','memory.swap.max'])
def test_parent_limit_is_rejected_even_when_leaf_is_unlimited(tmp_path, monkeypatch, name):
    root, parent, _ = cgroups(tmp_path, monkeypatch)
    (parent/name).write_text('176093659136\n')
    with pytest.raises(RuntimeError, match='remains active'):
        verify_unprotected_memory(cgroup_root=root,
                                  membership='0::/system.slice/experiment.service\n')
