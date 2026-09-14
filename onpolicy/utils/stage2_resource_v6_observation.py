"""Opt-in observable resource view and semantic dispatch history (V6).

No transition, mask, reward or legacy graph tensor is modified here. Historical
IDs are audit metadata, never ordinal policy features. Only raw observations
are retained: no learned embedding or future completion timestamp is cached.
"""
from __future__ import annotations

import numpy as np
import torch

VERSION = 'resource_semantic_pair_view_v6_1'
TYPES = tuple(f'R{i:03d}' for i in range(1, 15))
REQUEST_DIM = 46
DEVICE_DIM = 21
HISTORY_DIM = REQUEST_DIM + 2
GLOBAL_DIM = REQUEST_DIM * 2 + DEVICE_DIM * 2 + 4
PAIR_DIM = DEVICE_DIM + REQUEST_DIM + GLOBAL_DIM + 6


def enabled(env):
    return bool(getattr(env, 'config', {}).get('stage2_resource_v6_observations', False))


def capture_dispatches(env, plan):
    """Snapshot the entire validated joint plan BEFORE any device is moved."""
    if not enabled(env):
        return {}
    current = getattr(env, '_v6_request_snapshots', {})
    result = {}
    for device, request, _ in plan:
        identity = tuple(env._request_identity(request))
        if identity not in current:
            raise RuntimeError('V6 dispatch lacks a pre-action semantic observation.')
        result[device.code] = dict(identity=identity,
            case=str(getattr(env, 'current_case_path', '')),
            time=float(env.total_time), features=current[identity].copy())
    return result


def bind_teacher_identities(env, decisions):
    """Independent live-pool identity for validating saved observation labels."""
    if not enabled(env):
        return
    sites = list(env.sites)
    for decision in decisions:
        idx = int(decision['selected_request_id'])
        req = env.request_pool.get(idx) if idx else None
        decision['selected_request_semantic_key'] = (None if req is None else
            [int(req.get('plane_idx', -1)), env.job_code_list.index(req['job_code']), sites.index(req['site_code'])])


def attach_resource_view(env, graph, coordinate_scale):
    """Attach fixed-shape metadata; the existing graph remains byte-identical."""
    if not enabled(env):
        return
    r, d = env.max_request_num, env.max_device_num
    requests = np.zeros((r, REQUEST_DIM), np.float32)
    devices = np.zeros((d, DEVICE_DIM), np.float32)
    pairs = np.zeros((d, r, 2), np.float32)
    history = np.zeros((d, HISTORY_DIM), np.float32)
    keys = np.full((r, 3), -1, np.int64)
    snapshots = {}
    sites = list(env.sites)
    n_jobs = max(1, len(env.job_code_list))
    kind_names = ('bounded_mobile_frontier_h1', 'bounded_mobile_frontier_h2',
                  'blocking_wait', 'departure_pickup')
    for req in env.request_list:
        idx = int(req['id'])
        if req.get('is_noop', False):
            requests[idx, 1] = 1.
            continue
        plane = env.planes.get(req.get('plane_id'))
        job_code, site_code = req['job_code'], req['site_code']
        job = (plane.jobs.get(job_code, env.jobs[job_code]) if plane is not None
               else env.jobs[job_code])
        site = env.sites[site_code]
        remaining = [] if plane is None else list(plane.left_jobs) + list(plane.current_jobs)
        remaining_work = sum(max(0., float(getattr(
            plane.jobs.get(code, env.jobs.get(code)), 'time', 0.) or 0.))
            for code in remaining) if plane is not None else 0.
        finished = set() if plane is None else set(plane.finished_jobs)
        unfinished = [code for code in getattr(job, 'predecessor', ()) or () if code not in finished]
        kind = str(req.get('request_kind', ''))
        if kind == 'bounded_mobile_frontier':
            kind = 'bounded_mobile_frontier_h1'
        kinds = [float(kind == name) for name in kind_names]
        kinds.append(float(kind not in kind_names))
        required = set(getattr(job, 'resources', ()))
        if kind == 'departure_pickup':
            required.add(env.TRANSPORTER_RESOURCE_TYPE)
        base = [1., 0., float(bool(req.get('is_lookahead', False))),
            min(max(0., float(req.get('waiting_time', 0.))) / 3600., 10.),
            min(max(0., float(req.get('lead_time', 0.))) / 3600., 10.),
            float(site.pos[0]) / coordinate_scale, float(site.pos[1]) / coordinate_scale,
            min(max(0., float(getattr(job, 'time', 0.) or 0.)) / 3600., 10.),
            len(remaining) / n_jobs, min(remaining_work / 3600., 100.),
            float(req.get('dependency_depth', 0) or 0) / 4., len(unfinished) / n_jobs,
            float(site.is_occupied), float(site.is_interfered),
            min(max(float(site.left_job_time), float(site.left_rec_time), 0.) / 3600., 10.),
            float(env.total_time) / 10000.]
        op_idx = int(graph.request_operation_indices[idx])
        op = graph['operation'].x[op_idx].numpy().tolist() if op_idx >= 0 else [0.] * 11
        requests[idx] = base + op + [float(t in required) for t in TYPES] + kinds
        identity = tuple(env._request_identity(req))
        if identity in snapshots:
            raise RuntimeError('Duplicate semantic request identity in V6 observation.')
        snapshots[identity] = requests[idx].copy()
        keys[idx] = [int(req.get('plane_idx', -1)), env.job_code_list.index(job_code), sites.index(site_code)]
    env._v6_request_snapshots = snapshots
    ledger = getattr(env, '_v6_dispatch_history', {})
    for idx, dev in enumerate(env.device_list[:d]):
        if dev.resource.type not in TYPES:
            raise ValueError(f'Unversioned V6 resource type: {dev.resource.type}')
        old = graph['device'].x[idx].numpy()
        devices[idx] = ([float(t == dev.resource.type) for t in TYPES]
            + old[1:5].tolist() + [float(dev.velocity) / 1000.,
                float(env._device_observation_available(dev)), 1.])
        for req in env.request_list[1:]:
            target = env.sites[req['site_code']]
            distance = sum(abs(float(a) - float(b)) for a, b in zip(dev.site.pos, target.pos))
            pairs[idx, int(req['id'])] = [min(distance / max(float(dev.velocity), 1.) / 600., 20.),
                                         float(dev.site.code == target.code)]
        prior = ledger.get(dev.code)
        if prior is not None:
            if prior['case'] != str(getattr(env, 'current_case_path', '')):
                raise RuntimeError('V6 history leaked across cases.')
            age = float(env.total_time) - prior['time']
            if age < -1e-6:
                raise RuntimeError('V6 history has a future timestamp.')
            history[idx] = np.concatenate((prior['features'], [1., max(0., age) / 3600.]))
    if not all(np.isfinite(x).all() for x in (requests, devices, pairs, history)):
        raise FloatingPointError('Non-finite V6 observation; no sanitizing is allowed.')
    graph.v6_requests = torch.from_numpy(requests).unsqueeze(0)
    graph.v6_devices = torch.from_numpy(devices).unsqueeze(0)
    graph.v6_pairs = torch.from_numpy(pairs).unsqueeze(0)
    graph.v6_history = torch.from_numpy(history).unsqueeze(0)
    graph.v6_request_keys = torch.from_numpy(keys).unsqueeze(0)
    graph.v6_counts = torch.tensor([[float(env.total_time) / 10000.,
        (len(env.request_list) - 1) / max(1, r - 1),
        len(env.device_list[:d]) / max(1, d), len(env.planes) / max(1, env.n_plane_agents)]])
    graph.v6_schema = torch.tensor([1], dtype=torch.long)
