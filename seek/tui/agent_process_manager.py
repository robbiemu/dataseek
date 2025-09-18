import asyncio
import os
from typing import Optional


class AgentProcessManager:
    """Manages the lifecycle of the agent subprocess."""

    def __init__(self, mission_name: str, recursion_limit: int, use_robots: bool = True, seek_config_path: Optional[str] = None, mission_plan_path: Optional[str] = None):
        self.mission_name = mission_name
        self.recursion_limit = recursion_limit
        self.use_robots = use_robots
        self.seek_config_path = seek_config_path
        self.mission_plan_path = mission_plan_path
        self.process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> asyncio.subprocess.Process:
        """Starts the agent subprocess."""
        command = [
            "dataseek",
            "--mission",
            self.mission_name,
            "--recursion-limit",
            str(self.recursion_limit),
        ]
        
        if self.seek_config_path:
            command.extend(["--config", self.seek_config_path])
        
        if self.mission_plan_path:
            command.extend(["--mission-config", self.mission_plan_path])

        # Add --no-robots flag if use_robots is False
        if not self.use_robots:
            command.append("--no-robots")
        
        # Use the current environment without forcing any variables
        env = os.environ.copy()
        
        self.process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
        )
        return self.process

    def terminate(self):
        """Terminates the agent subprocess."""
        if self.process and self.process.returncode is None:
            self.process.terminate()
