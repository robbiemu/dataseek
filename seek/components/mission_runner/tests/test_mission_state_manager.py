import uuid

from seek.components.mission_runner.mission_runner import MissionStateManager


def test_mission_state_manager_in_memory_db_roundtrip():
    """Using ':memory:' should work across create/get/update on one manager instance.

    This guards against regressions where each method opens a new sqlite
    connection, which breaks in-memory DBs and causes 'no such table'.
    """
    mgr = MissionStateManager(":memory:")

    mission_id = f"mission_{uuid.uuid4()}"
    initial = {"samples_generated": 0, "progress": {}, "mission_name": "t"}

    # Create should not raise and should be retrievable
    mgr.create_mission(mission_id, initial)
    fetched = mgr.get_mission_state(mission_id)
    assert fetched == initial

    # Update should persist and be retrievable
    updated = dict(initial)
    updated["samples_generated"] = 1
    mgr.update_mission_state(mission_id, updated)
    fetched2 = mgr.get_mission_state(mission_id)
    assert fetched2 == updated
