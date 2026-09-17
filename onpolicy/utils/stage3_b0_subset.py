"""Deterministic Train240 continuation, retaining the real Train600 history."""
from collections import Counter
from pathlib import Path

from onpolicy.utils.stage3_research import atomic_json, digest_file, digest_json, read_json

SCHEMA = 'stage3_b0_train240_subset_v1'
COUNTS = {'iid': 192, 'ood_stress': 43, 'ood_scale': 5}
WEIGHTS = {'iid': 1, 'ood_stress': 4, 'ood_scale': 4}
EPOCH_VISITS = 384
EPOCHS = 8


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest_file(path))


def bound(record):
    if digest_file(record['path']) != record['sha256']:
        raise ValueError('Training subset evidence changed')
    return read_json(record['path'])


def select_cases(cases, seed):
    """Allocate within distributions by profile, then select by seeded hashes."""
    if (Counter(c['distribution'] for c in cases) != dict(iid=480, ood_stress=108, ood_scale=12)
            or len({c['path'] for c in cases}) != 600):
        raise ValueError('Train240 selection requires the original unique Train600')
    selected = []
    for distribution, count in COUNTS.items():
        pool = [c for c in cases if c['distribution'] == distribution]
        profiles = Counter(c['profile'] for c in pool)
        quotas = {p: count*n//len(pool) for p, n in profiles.items()}
        order = sorted(profiles, key=lambda p: (-(count*profiles[p] % len(pool)), p))
        for p in order[:count-sum(quotas.values())]:
            quotas[p] += 1
        for profile in sorted(profiles):
            members = sorted((c for c in pool if c['profile'] == profile),
                key=lambda c: (digest_json([seed, SCHEMA, c['content_sha256']]), c['path']))
            selected.extend(members[:quotas[profile]])
    return sorted(selected, key=lambda c: c['path'])


def create_selection(parent_path, output, seed):
    parent = read_json(parent_path)
    cases = select_cases(parent['splits']['train'], seed)
    value = dict(schema=SCHEMA, seed=seed, parent_manifest=binding(parent_path),
        parent_train_sha256=digest_json(parent['splits']['train']), cases=cases,
        cases_sha256=digest_json(cases), counts=COUNTS,
        profiles=dict(Counter(c['profile'] for c in cases)), weights=WEIGHTS,
        epoch_visits=EPOCH_VISITS, epochs=EPOCHS, new_training_visits=EPOCH_VISITS*EPOCHS,
        rule='distribution quotas, largest-remainder profile quotas, seeded content hashes; no scores')
    atomic_json(output, value, overwrite=False)
    return value


def verify_selection(parent, record):
    value = bound(record)
    original = bound(value['parent_manifest'])
    if (value.get('schema') != SCHEMA or value.get('counts') != COUNTS
            or value.get('weights') != WEIGHTS or value.get('epoch_visits') != EPOCH_VISITS
            or value.get('epochs') != EPOCHS or value.get('new_training_visits') != EPOCH_VISITS*EPOCHS
            or original['manifest_sha256'] != parent['manifest_sha256']
            or original['splits'] != parent['splits']
            or value['parent_train_sha256'] != digest_json(parent['splits']['train'])
            or value['cases'] != select_cases(parent['splits']['train'], value['seed'])
            or value['cases_sha256'] != digest_json(value['cases'])
            or value['profiles'] != dict(Counter(c['profile'] for c in value['cases']))):
        raise ValueError('Train240 selection or parent identity changed')
    excluded = [c for split, rows in parent['splits'].items() if split != 'train' for c in rows]
    if ({c['path'] for c in value['cases']} & {c['path'] for c in excluded}
            or {c['content_sha256'] for c in value['cases']} & {c['content_sha256'] for c in excluded}):
        raise ValueError('Training subset overlaps held-out cases')
    return value


def epoch_rows(selection, epoch):
    seed = selection['seed']
    visits = [(c, repetition) for c in selection['cases']
              for repetition in range(WEIGHTS[c['distribution']])]
    visits.sort(key=lambda item: digest_json([seed, SCHEMA, 'order', epoch,
                                              item[0]['content_sha256'], item[1]]))
    return [(c, int(digest_json([seed, SCHEMA, 'trajectory', epoch, c['content_sha256'], rep])[:15], 16) % (2**31-1),
             f'train240:{seed}:{epoch}:{i}') for i, (c, rep) in enumerate(visits)]


def make_amendment(parent, prefix, selection_record, output):
    selection = verify_selection(parent, selection_record)
    path = Path(output) / 'training_subset_prefix.json'
    atomic_json(path, prefix, overwrite=False)
    return dict(schema=SCHEMA, selection=selection_record, prefix_schedule=binding(path),
        inherited_batches=len(prefix), inherited_visits=prefix[-1]['training_episodes'],
        inherited_actor_updates=2*len(prefix), epoch_visits=EPOCH_VISITS, epochs=EPOCHS,
        new_training_visits=selection['new_training_visits'],
        normalization_scope='original frozen Train600 statistics retained exactly',
        initialization_scope='Train600-trained checkpoint; subsequent visits use only fixed Train240')


def subset_schedule(m):
    a = m['training_subset']
    selection = bound(a['selection'])
    prefix = bound(a['prefix_schedule'])
    n = a['inherited_batches']
    if (a.get('schema') != SCHEMA or not prefix or len(prefix) != n
            or prefix[-1]['training_episodes'] != a['inherited_visits']
            or a['inherited_actor_updates'] != 2*n
            or m['splits']['train'] != selection['cases']
            or a['epoch_visits'] != EPOCH_VISITS or a['epochs'] != EPOCHS
            or a['new_training_visits'] != EPOCH_VISITS*EPOCHS):
        raise ValueError('Invalid Train240 schedule identity')
    resume = m['resume_amendment']
    completed = resume['completed_batch_sizes']
    if (completed[:n] != [len(row['cases']) for row in prefix]
            or len(completed) != resume['completed_batches']
            or sum(completed) != resume['completed_visits']
            or 2*len(completed) != resume['completed_actor_updates']):
        raise ValueError('Train240 continuation cursor changed')
    sizes, done = list(completed[n:]), 0
    for size in sizes:
        if type(size) is not int or not 0 < size <= EPOCH_VISITS-done % EPOCH_VISITS:
            raise ValueError('Committed subset batch crosses an epoch boundary')
        done += size
    total = EPOCH_VISITS*EPOCHS
    if done > total:
        raise ValueError('Subset training budget exceeded')
    width = m['recipe']['global_batch']
    if width not in (32,64,128,240,256,512):
        raise ValueError('Unregistered subset batch size')
    while done < total:
        size = min(width, EPOCH_VISITS-done % EPOCH_VISITS)
        sizes.append(size)
        done += size
    visits = [row for epoch in range(EPOCHS) for row in epoch_rows(selection, epoch)]
    plan, start = list(prefix), 0
    for size in sizes:
        group = visits[start:start+size]
        epoch = start//EPOCH_VISITS
        plan.append(dict(cases=[r[0] for r in group], seeds=[r[1] for r in group],
            visit_ids=[r[2] for r in group], visit=epoch, data_epoch=epoch+1,
            group=len(plan)+1, training_episodes=a['inherited_visits']+start+size))
        start += size
    return plan


def verify_subset(m):
    from onpolicy.utils.stage3_b_shared_b0 import training_schedule
    a = m['training_subset']
    selection = bound(a['selection'])
    parent = bound(selection['parent_manifest'])
    verify_selection(parent, a['selection'])
    prefix = bound(a['prefix_schedule'])
    if (prefix != training_schedule(parent)[:a['inherited_batches']]
            or any(m['splits'][s] != parent['splits'][s] for s in parent['splits'] if s != 'train')):
        raise ValueError('Subset changed historical visits or held-out data')
    subset_schedule(m)
