from seek.components.search_graph.nodes.utils import get_characteristic_context


def test_get_characteristic_context_accepts_single_mission_dict():
    mission = {
        "name": "mac_ai_corpus_v1",
        "goals": [
            {
                "characteristic": "Primary Data Collection",
                "context": "Collect original data via measurements, surveys, or experiments.",
            }
        ],
    }
    task = {"characteristic": "Primary Data Collection"}
    ctx = get_characteristic_context(task, mission)
    assert ctx
    assert "original data" in ctx
