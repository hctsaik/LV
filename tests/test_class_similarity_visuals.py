from __future__ import annotations

from types import SimpleNamespace

from class_similarity_visuals import jitter_x, sample_scatter_rows


def test_scatter_sampling_is_bounded_deterministic_and_keeps_lowest_margins():
    items = [SimpleNamespace(
        record_id=f"r{index:03d}", label="A",
        margin=float(index), same=float(index) / 300, other=0.25,
    ) for index in reversed(range(250))]
    sampled = sample_scatter_rows(items, "same", "analysis", per_class_cap=25, margin_keep=5)
    sampled_again = sample_scatter_rows(items, "same", "analysis", per_class_cap=25, margin_keep=5)

    assert len(sampled) == 25
    assert [item.record_id for item in sampled] == [item.record_id for item in sampled_again]
    assert {item.record_id for item in sampled[:5]} == {f"r{index:03d}" for index in range(5)}
    assert jitter_x("analysis", "r010") == jitter_x("analysis", "r010")
    assert abs(jitter_x("analysis", "r010")) <= 0.18
