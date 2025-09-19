from unittest.mock import MagicMock

from seek.components.mission_runner.mission_runner import MissionRunner


class DummyApp:
    def __init__(self):
        self.last_config = None

    def update_state(self, _config, _state):
        pass

    def stream(self, _inputs, config=None):
        # Capture the config provided by the runner
        self.last_config = config
        # Yield a single empty event to satisfy the iterator usage
        yield {}


def test_recursion_limit_is_top_level_in_config():
    # Arrange: mission runner with a dummy app and mock checkpointer
    dummy_app = DummyApp()
    mock_checkpointer = MagicMock()
    mission_config = {"name": "test_mission", "target_size": 1, "goals": []}
    runner = MissionRunner(
        checkpointer=mock_checkpointer,
        app=dummy_app,
        mission_config=mission_config,
        seek_config={},
    )

    # Act: run a single cycle
    runner.run_full_cycle(thread_id="thread_test", recursion_limit=14)

    # Assert: recursion_limit is at the top level of the config
    assert dummy_app.last_config is not None
    assert dummy_app.last_config.get("recursion_limit") == 14
    # And thread_id remains under configurable
    assert dummy_app.last_config.get("configurable", {}).get("thread_id") == "thread_test"
