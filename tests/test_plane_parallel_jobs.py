"""Regression coverage for pairwise exclusion within a parallel job group."""

from itertools import combinations, permutations, product
from types import SimpleNamespace

from onpolicy.envs.HKBZ.core.plane import Plane


def make_plane(order, exclusions=None, durations=None):
    exclusions = exclusions or {}
    durations = durations or {}
    site = SimpleNamespace(res_avail={}, started_jobs=[])
    site.update_resources = lambda: None
    site.start_jobs = lambda jobs: site.started_jobs.extend(job.code for job in jobs)
    jobs = [
        SimpleNamespace(
            code=code,
            group="保障",
            time=durations.get(code, 10),
            exclusive=set(exclusions.get(code, ())),
            predecessor=set(),
            resources=set(),
        )
        for code in order
    ]
    return Plane("test-plane", {"velocity": 1, "site": site, "jobs": jobs})


def test_mutually_exclusive_secondary_jobs_do_not_start_together():
    plane = make_plane(
        ("A", "B", "C"),
        exclusions={"B": {"C"}, "C": {"B"}},
        durations={"A": 20},
    )

    assert plane.choose_job("A") == 20

    assert plane.current_jobs == ["A", "B"]
    assert plane.site.started_jobs == ["A", "B"]
    assert plane.left_jobs == ["C"]
    assert plane.job_time == 20
    assert plane.total_job_time == 30


def test_anchor_only_exclusion_blocks_candidate_before_anchor():
    plane = make_plane(("A", "B", "C"), exclusions={"B": {"A"}})

    assert plane.get_parallel_jobs("B") == ["B", "C"]


def test_pairwise_exclusion_for_all_three_job_directed_graphs():
    nodes = ("A", "B", "C")
    edges = tuple(permutations(nodes, 2))
    for bits in range(1 << len(edges)):
        exclusions = {node: set() for node in nodes}
        for bit, (first, second) in enumerate(edges):
            if bits & (1 << bit):
                exclusions[first].add(second)
        for order in permutations(nodes):
            for anchor in nodes:
                plane = make_plane(order, exclusions)
                selected = plane.get_parallel_jobs(anchor)
                context = (bits, order, anchor, selected)

                assert anchor in selected, context
                assert len(selected) == len(set(selected)), context
                assert selected == [job for job in order if job in selected], context
                for first, second in combinations(selected, 2):
                    assert second not in exclusions[first], context
                    assert first not in exclusions[second], context
                for omitted in set(nodes) - set(selected):
                    assert any(
                        other in exclusions[omitted] or omitted in exclusions[other]
                        for other in selected
                    ), context


def test_no_conflict_preserves_order_and_duration_filter():
    nodes = ("A", "B", "C")
    for order in permutations(nodes):
        for values in product((0, 1, 2), repeat=3):
            durations = dict(zip(nodes, values))
            plane = make_plane(order, durations=durations)
            for anchor in nodes:
                expected = [job for job in order if durations[job] <= durations[anchor]]
                assert plane.get_parallel_jobs(anchor) == expected, (order, values, anchor)


def test_predecessor_and_resource_availability_still_apply():
    plane = make_plane(("A", "B", "C"))
    plane.jobs["B"].predecessor = {"unfinished"}
    plane.jobs["C"].resources = {"required-resource"}
    assert plane.get_parallel_jobs("A") == ["A"]

    plane.finished_jobs = ["unfinished"]
    plane.site.res_avail = {"required-resource": 1}
    assert plane.get_parallel_jobs("A") == ["A", "B", "C"]
