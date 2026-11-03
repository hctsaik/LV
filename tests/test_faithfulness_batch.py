"""Batch evidence keeps fixed scores, missing targets and image support distinct."""
import copy

import pytest

from yolo_attribution import AttributionError, summarize_faithfulness


def report(drop, controls, rematch):
    return {"status": "measured", "rows": [
        {"fraction": .2, "role": "attribution", "fixed_score_drop": drop,
         "rematched_detection": rematch},
        *[{"fraction": .2, "role": "random_control", "fixed_score_drop": x,
           "rematched_detection": None} for x in controls]]}


def test_replacement_boxes_and_missing_detections_do_not_replace_fixed_scores():
    rows = [report(.5, [.1, .3], {"output_index": 999, "score": .99}),
            report(.1, [.2, .4], None), {"status": "unavailable"}]
    result = summarize_faithfulness(rows)
    assert result["requested_images"] == 3 and result["measured_images"] == 2
    fraction = result["fractions"]["0.2"]
    assert fraction["support"] == 2
    assert fraction["mean"] == pytest.approx(.05)
    assert fraction["std"] == pytest.approx(2 ** .5 * .25)
    changed = copy.deepcopy(rows)
    for row in changed[0]["rows"]:
        row["rematched_detection"] = None
    assert summarize_faithfulness(changed) == result


def test_missing_controls_and_unavailable_targets_are_inconclusive():
    result = summarize_faithfulness([report(.5, [], None), {"status": "unavailable"}])
    assert result["status"] == "inconclusive" and not result["fractions"]
    one = summarize_faithfulness([report(.5, [.1], None)])
    assert one["fractions"]["0.2"]["std"] is None
    for reports in ([], [{"status": "unavailable"}] * 201):
        with pytest.raises(AttributionError):
            summarize_faithfulness(reports)
