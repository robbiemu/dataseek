from unittest.mock import patch, MagicMock

import pytest

pytest.importorskip("langsmith")

from seek.main import run
from seek.models import (
    SeekAgentConfig,
    SeekAgentMissionPlanConfig,
    SeekAgentWriterConfig,
)


@patch("seek.main.input")
@patch("seek.main.MissionRunner")
@patch("seek.main.load_seek_config")
def test_seek_agent_runnable_new_mission(
    mock_load_config, mock_mission_runner, mock_input
):
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
    mock_mission_plan = SeekAgentMissionPlanConfig(
        goal="Test Goal", nodes=[], tools={}
    )
    mock_writer_config = SeekAgentWriterConfig()
    mock_seek_config = SeekAgentConfig(
        mission_plan=mock_mission_plan, writer=mock_writer_config
    )
    mock_load_config.return_value = mock_seek_config

    # We expect this to run without errors
    run(mission_name="test_mission", resume_from=None)

    # Check that the config was loaded and MissionRunner was used
    mock_load_config.assert_called_once()
    mock_mission_runner.assert_called_once()
    mock_runner_instance.run_mission.assert_called_once()


@patch("seek.main.input")
@patch("seek.main.MissionRunner")
@patch("seek.main.load_seek_config")
def test_seek_agent_runnable_resume_mission(
    mock_load_config, mock_mission_runner, mock_input
):
    """
    Tests that the seek agent can resume a mission.
    """
    # Mock the input to exit immediately
    mock_input.return_value = "exit"

    # Mock the MissionRunner instance
    mock_runner_instance = MagicMock()
    mock_mission_runner.return_value = mock_runner_instance

    # Mock the claimify config
    mock_seek_config = MagicMock()
    mock_load_config.return_value = mock_seek_config

    # We expect this to run without errors
    run(mission_name="test_mission", resume_from="existing-thread-id")

    # Check that the config was loaded and MissionRunner was used
    mock_load_config.assert_called_once()
    mock_mission_runner.assert_called_once()
