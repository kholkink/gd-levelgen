"""Level metrics and the comparison against a reference set (gdgen/evaluate.py)."""
import pytest

from gdgen.evaluate import SCALARS, compare, js_distance, level_metrics


def level(n_blocks=40, gap_at=None, deco=10):
    objs = [{"1": "1", "2": str(15 + 30 * i), "3": "15"} for i in range(n_blocks) if i != gap_at]
    objs += [{"1": "8", "2": "615", "3": "45"}, {"1": "36", "2": "915", "3": "75"}]   # spike, yellow orb
    objs += [{"1": "211", "2": str(15 + 60 * i), "3": "200", "57": "3.7"} for i in range(deco)]
    objs += [{"1": "901", "2": "300", "3": "300", "51": "3"}]                          # move trigger
    return {"kA4": "0", "kS38": "1_0_2_0_3_0_7_1|2_0_2_0_3_0_7_1"}, objs


def test_level_metrics_counts_streams_and_density():
    m = level_metrics(*level())
    assert m["gameplay"] == 42 and m["deco"] == 11 and m["objects"] == 53
    assert m["triggers"] == 1 and m["groups"] == 2 and m["color_channels"] == 2
    assert m["hazard_share"] == pytest.approx(1 / 42)
    assert m["gameplay_segments"] == 4 and m["empty_share"] == 0          # x 15 .. 1185 -> segments 0..3
    assert m["ids_gameplay"]["1"] == 40 and m["ids_deco"]["211"] == 10
    assert set(SCALARS) <= set(m)


def test_gap_and_empty_segments():
    _, objs = level(n_blocks=80)
    objs = [o for o in objs if not (o["1"] == "1" and 1200 < float(o["2"]) < 1800)]  # remove two segments
    m = level_metrics({}, objs)
    assert m["max_gap"] >= 600 and m["empty_share"] > 0


def test_js_distance_and_compare():
    assert js_distance({"a": 1}, {"a": 2}) == pytest.approx(0, abs=1e-6)
    assert js_distance({"a": 1}, {"b": 1}) == pytest.approx(1, abs=1e-3)
    ref = [level_metrics(*level(n_blocks=n)) for n in (30, 40, 50, 60)]
    gen = [level_metrics(*level(n_blocks=n, deco=0)) for n in (35, 45)]
    out = compare({"ref": ref, "gen": gen}, "ref")
    assert out["ref"]["n"] == 4 and "js" not in out["ref"]
    assert out["gen"]["in_ref_range"]["gameplay"] == 1.0
    assert out["gen"]["in_ref_range"]["deco"] == 0.0 and out["gen"]["js"]["ids_deco"] > 0.5
