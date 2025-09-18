from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("langsmith")

from seek.main import run
from seek.models import (
    SeekAgentMissionPlanConfig,
    SeekAgentWriterConfig,
)


@patch("seek.main.input")
@patch("seek.main.MissionRunner")
@patch("seek.main.load_seek_config")
def test_seek_agent_runnable_new_mission(mock_load_config, mock_mission_runner, mock_input):
    """
    Tests that the seek agent can start a new mission.
    """
    # Mock the input to exit immediately
    mock_input.return_value = "exit"

    # Mock the MissionRunner
    mock_runner_instance = mock_mission_runner.return_value
    mock_runner_instance.get_mission_names.return_value = ["test_mission"]
    mock_runner_instance.run_mission.return_value = None

    # Mock the configuration
    mock_mission_plan = SeekAgentMissionPlanConfig(goal="Test Goal", nodes=[], tools={})
    mock_writer_config = SeekAgentWriterConfig()
    # Minimal dict-like seek config for the current code
    mock_load_config.return_value = {
        "model_defaults": {"model": "openai/gpt-5-mini", "temperature": 0.1, "max_tokens": 2000},
        "mission_plan": {"nodes": []},
        "writer": mock_writer_config.model_dump(),
        "nodes": {"research": {"max_iterations": 7}},
        "use_robots": True,
        "observability": {},
    }

    # Ensure the mission exists in the mission plan and name list
    with (
        patch("seek.main.load_mission_plan") as mock_load_mission_plan,
        patch("seek.main.get_mission_names") as mock_get_mission_names,
    ):
        mock_get_mission_names.return_value = ["test_mission"]
        mock_load_mission_plan.return_value = {
            "missions": [{"name": "test_mission", "target_size": 1, "goals": []}],
            "total_samples_target": 1,
            "mission_plan_path": "settings/mission_config.yaml",
        }
        # We expect this to run without errors
        run(mission_name="test_mission", resume_from=None)

    # Check that the config was loaded and MissionRunner was used
    assert mock_load_config.call_count >= 1
    assert mock_mission_runner.call_count >= 1
    mock_runner_instance.run_mission.assert_called_once()


@patch("seek.main.input")
@patch("seek.main.MissionRunner")
@patch("seek.main.load_seek_config")
def test_seek_agent_runnable_resume_mission(mock_load_config, mock_mission_runner, mock_input):
    """
    Tests that the seek agent can resume a mission.
    """
    # Mock the input to exit immediately
    mock_input.return_value = "exit"

    # Mock the MissionRunner instance
    mock_runner_instance = MagicMock()
    mock_mission_runner.return_value = mock_runner_instance

    # Mock the seek config to a minimal dict-like
    mock_load_config.return_value = {
        "model_defaults": {"model": "openai/gpt-5-mini", "temperature": 0.1, "max_tokens": 2000},
        "mission_plan": {"nodes": []},
        "nodes": {},
        "use_robots": True,
        "observability": {},
    }

    # We expect this to run without errors
    run(mission_name="test_mission", resume_from="existing-thread-id")

    # Check that the config was loaded and MissionRunner was used
    assert mock_load_config.call_count >= 1
    assert mock_mission_runner.call_count >= 1
