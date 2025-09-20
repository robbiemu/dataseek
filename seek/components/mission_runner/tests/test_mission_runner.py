import unittest
from unittest.mock import MagicMock, patch

from seek.components.mission_runner.mission_runner import MissionRunner


class TestMissionRunner(unittest.TestCase):
    def test_get_progress(self):
        """Test that get_progress correctly retrieves and parses the state."""
        # 1. Setup Mocks
        mock_app = MagicMock()
        mock_checkpointer = MagicMock()

        # 2. Mock the return value of app.get_state
        mock_state = MagicMock()
        mock_state.values = {
            "samples_generated": 10,
            "total_samples_target": 100,
        }
        mock_app.get_state.return_value = mock_state

        # 3. Instantiate the MissionRunner
        with patch("seek.components.mission_runner.mission_runner.MissionStateManager"):
            mission_runner = MissionRunner(
                checkpointer=mock_checkpointer,
                app=mock_app,
                mission_config={"name": "test_mission", "target_size": 0, "goals": []},
                seek_config={},
            )

            # 4. Call the method to be tested
            progress = mission_runner.get_progress("some_thread_id")

            # 5. Assert the results
            self.assertEqual(progress["samples_generated"], 10)
            self.assertEqual(progress["total_target"], 100)
            self.assertEqual(progress["progress_pct"], 10.0)

            # 6. Verify that the correct method was called on the mock
            mock_app.get_state.assert_called_once_with(
                {"configurable": {"thread_id": "some_thread_id"}}
            )


if __name__ == "__main__":
    unittest.main()
