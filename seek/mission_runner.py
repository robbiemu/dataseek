# mission_runner.py
"""
Mission Runner for the Data Seek Agent.
This class orchestrates the mission execution and handles checkpointing for resumable missions.
"""

import json
import logging
import math
import sqlite3
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MissionStateManager:
    """Manages the persistent state of a mission."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        """Create the mission_state table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mission_state (
                    mission_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL
                )
            """
            )
            conn.commit()

    def create_mission(self, mission_id: str, initial_state: dict[str, Any]):
        """Create a new mission entry in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO mission_state (mission_id, state) VALUES (?, ?)",
                (mission_id, json.dumps(initial_state)),
            )
            conn.commit()
        logging.info(f"Mission {mission_id} created in the database.")

    def get_mission_state(self, mission_id: str) -> dict[str, Any] | None:
        """Retrieve the state of a mission."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT state FROM mission_state WHERE mission_id = ?", (mission_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def update_mission_state(self, mission_id: str, new_state: dict[str, Any]):
        """Update the state of an existing mission."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE mission_state SET state = ? WHERE mission_id = ?",
                (json.dumps(new_state), mission_id),
            )
            conn.commit()
        logging.info(f"Mission {mission_id} state updated.")


class MissionRunner:
    """A high-level orchestrator for Data Seek missions with checkpointing support."""

    def __init__(
        self,
        checkpointer: SqliteSaver,
        app,
        mission_config: dict[str, Any],
        seek_config: dict[str, Any],
        resume_from_mission_id: str | None = None,
    ):
        """Initialize the MissionRunner with a mission configuration."""
        self.mission_config = mission_config
        self.seek_config = seek_config
        self.checkpointer = checkpointer
        self.app = app
        self.state_manager = MissionStateManager("checkpoints/mission_checkpoints.db")
        self.resume_from_mission_id = resume_from_mission_id

    def _initialize_mission_state(self) -> dict[str, Any]:
        """Prepare the initial state for a new mission."""
        mission_name = self.mission_config.get("name")
        target_size = self.mission_config.get("target_size", 0)
        goals = self.mission_config.get("goals", [])

        # Load mission config for output paths and recursion limit
        output_paths = self.mission_config.get("output_paths", {})
        pedigree_path = output_paths.get("audit_trail_path", "examples/PEDIGREE.md")
        samples_path = output_paths.get("samples_path", "examples/data/datasets/samples")
        max_recursion_steps = self.seek_config.get("recursion_per_sample", 30)

        progress = {
            mission_name: {
                goal["characteristic"]: {
                    "target": target_size,
                    "collected": 0,
                    "topics": {
                        topic: {
                            "collected": 0,
                            "target": math.ceil(target_size / len(goal["topics"])),
                        }
                        for topic in goal["topics"]
                    },
                }
                for goal in goals
            }
        }

        return {
            "mission_name": mission_name,
            "total_samples_target": target_size * len(goals),
            "samples_generated": 0,
            "synthetic_samples_generated": 0,
            "research_samples_generated": 0,
            "progress": progress,
            "synthetic_budget": self.mission_config.get("synthetic_budget", 0.2),
            "pedigree_path": pedigree_path,
            "samples_path": samples_path,
            "last_thread_id": None,
            "excluded_urls": [],
            "next_cycle_cached_reuse": {"active": False},
            "max_recursion_steps": max_recursion_steps,
        }

    def run_mission(self, recursion_limit: int = 1000, max_samples: int | None = None):
        """
        Runs a full mission from start to finish, managing state across multiple sample generation cycles.
        """
        if self.resume_from_mission_id:
            mission_id = self.resume_from_mission_id
            mission_state = self.state_manager.get_mission_state(mission_id)
            if not mission_state:
                logging.error(f"❌ Could not find mission with ID '{mission_id}' to resume.")
                return
            logging.info(f"🔄 Resuming mission with ID: {mission_id}")
        else:
            mission_id = f"mission_{uuid.uuid4()}"
            logging.info(f"🚀 Starting new mission with ID: {mission_id}")
            mission_state = self._initialize_mission_state()
            self.state_manager.create_mission(mission_id, mission_state)

        total_target = mission_state.get("total_samples_target", 0)

        while mission_state.get("samples_generated", 0) < total_target:
            if max_samples is not None and mission_state["samples_generated"] >= max_samples:
                logging.info(f"Reached sample limit of {max_samples}. Ending mission.")
                break

            thread_id = f"thread_{uuid.uuid4()}"
            logging.info(f"  -> Starting sample cycle with thread ID: {thread_id}")

            # Prepare the initial state for this specific cycle
            cycle_state = self._prepare_cycle_state(mission_id, thread_id, mission_state)
            thread_config = {"configurable": {"thread_id": thread_id}}
            self.app.update_state(thread_config, cycle_state)

            # Run the agent for one full cycle
            self.run_full_cycle(thread_id, recursion_limit)

            # After the cycle, update the master mission state
            mission_state = self._update_mission_progress(mission_id, thread_id)

            # Log progress
            samples_done = mission_state.get("samples_generated", 0)
            progress_pct = (samples_done / total_target) * 100 if total_target > 0 else 0
            logging.info(
                f"📊 Mission Progress: {samples_done}/{total_target} samples ({progress_pct:.1f}%)"
            )

        logging.info(f"🏁 Mission {mission_id} completed.")

    def _prepare_cycle_state(
        self, mission_id: str, thread_id: str, mission_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Creates the initial state for a single agent cycle."""
        # Check if we should enter cached-only mode
        carryover = mission_state.get("next_cycle_cached_reuse", {"active": False})
        is_cached_reuse = carryover.get("active", False)

        base_state = {
            "run_id": thread_id,
            "mission_id": mission_id,
            "messages": [],
            "current_task": None,
            "research_findings": [],
            "decision_history": [],
            "tool_execution_failures": 0,
            "research_attempts": 0,
            "consecutive_failures": 0,
            "last_action_status": "success",
            "last_action_agent": "",
            "fitness_report": None,
            # Carry over mission-level info
            "current_mission": mission_state.get("mission_name"),
            "total_samples_target": mission_state.get("total_samples_target"),
            "synthetic_budget": mission_state.get("synthetic_budget"),
            "pedigree_path": mission_state.get("pedigree_path"),
            "samples_path": mission_state.get("samples_path"),
            "samples_generated": mission_state.get("samples_generated", 0),
            "progress": mission_state.get("progress", {}),
            # Always seed excluded URLs
            "excluded_urls": mission_state.get("excluded_urls", []),
            # Carry over sample counters
            "synthetic_samples_generated": mission_state.get("synthetic_samples_generated", 0),
            "research_samples_generated": mission_state.get("research_samples_generated", 0),
            # Add recursion tracking
            "step_count": 0,
            "max_recursion_steps": mission_state.get("max_recursion_steps", 30),
        }

        if is_cached_reuse:
            # Enter cached-only mode
            base_state.update(
                {
                    "cached_only_mode": True,
                    "no_search_tools": True,
                    "allowed_url_whitelist": carryover.get("allowed_url_whitelist", []),
                    "current_task": carryover.get("current_task"),
                    "research_session_cache": carryover.get("research_session_cache", []),
                    "cached_exhausted": False,
                }
            )
        else:
            # Normal mode
            base_state.update(
                {
                    "cached_only_mode": False,
                    "no_search_tools": False,
                    "allowed_url_whitelist": [],
                    "cached_exhausted": False,
                }
            )

        return base_state

    def run_full_cycle(self, thread_id: str, recursion_limit: int):
        """Runs the agent for a single cycle with a given thread_id."""
        # LangGraph expects recursion_limit at the top level of the config,
        # while dynamic values like thread_id belong under "configurable".
        thread_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": recursion_limit,
        }
        inputs = {"messages": [HumanMessage(content="Start the data generation cycle.")]}

        print(f"[DEBUG] Starting cycle for thread {thread_id}")

        try:
            for event in self.app.stream(inputs, config=thread_config):
                print(f"[DEBUG] Received event from app.stream: {list(event.keys())}")
                for _node_name, node_output in event.items():
                    if "messages" in node_output:
                        # Minimal logging to avoid clutter
                        pass
            logging.info(f"  -> Cycle for thread {thread_id} finished.")
        except Exception as e:
            import traceback

            print(f"[DEBUG] Exception in run_full_cycle: {e}")
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            raise

    def _update_mission_progress(self, mission_id: str, thread_id: str) -> dict[str, Any]:
        """
        Retrieves the final state from a cycle, updates the master mission state,
        and persists it.
        """
        thread_config = {"configurable": {"thread_id": thread_id}}
        final_cycle_state = self.app.get_state(thread_config).values

        mission_state = self.state_manager.get_mission_state(mission_id)
        if not mission_state:
            logging.error(f"Could not retrieve mission state for {mission_id} to update progress.")
            return {}

        # Logic to determine which characteristic/topic was generated
        current_task = final_cycle_state.get("current_task")
        if current_task:
            generated_characteristic = current_task.get("characteristic")
            generated_topic = current_task.get("topic")
        else:
            generated_characteristic = None
            generated_topic = None

        # Update the detailed breakdown
        if generated_characteristic and generated_topic:
            if generated_characteristic in mission_state["progress"][mission_state["mission_name"]]:
                mission_state["progress"][mission_state["mission_name"]][generated_characteristic][
                    "collected"
                ] += 1
                if (
                    generated_topic
                    in mission_state["progress"][mission_state["mission_name"]][
                        generated_characteristic
                    ]["topics"]
                ):
                    mission_state["progress"][mission_state["mission_name"]][
                        generated_characteristic
                    ]["topics"][generated_topic]["collected"] += 1

        # Update total counts
        mission_state["samples_generated"] = final_cycle_state.get(
            "samples_generated", mission_state["samples_generated"]
        )
        mission_state["synthetic_samples_generated"] = final_cycle_state.get(
            "synthetic_samples_generated", mission_state["synthetic_samples_generated"]
        )
        mission_state["research_samples_generated"] = final_cycle_state.get(
            "research_samples_generated", mission_state["research_samples_generated"]
        )

        # Update the last thread ID
        mission_state["last_thread_id"] = thread_id

        # Persist cached reuse state
        mission_state["excluded_urls"] = final_cycle_state.get(
            "excluded_urls", mission_state.get("excluded_urls", [])
        )
        mission_state["next_cycle_cached_reuse"] = final_cycle_state.get(
            "next_cycle_cached_reuse", {"active": False}
        )

        # If cached is exhausted or quotas reached, force carryover inactive
        if final_cycle_state.get("cached_exhausted"):
            mission_state["next_cycle_cached_reuse"] = {"active": False}

        self.state_manager.update_mission_state(mission_id, mission_state)
        logging.info(f"Master state for mission {mission_id} updated after cycle {thread_id}.")

        return mission_state

    def get_progress(self, thread_id: str) -> dict[str, Any]:
        """Return simple progress metrics for a given thread.

        Computes samples generated, total target, and percentage complete
        based on the current app state for the provided thread ID.
        """
        thread_config = {"configurable": {"thread_id": thread_id}}
        state = self.app.get_state(thread_config).values
        samples_generated = state.get("samples_generated", 0)
        total_target = state.get("total_samples_target", 0)
        progress_pct = (samples_generated / total_target * 100) if total_target else 0.0
        return {
            "samples_generated": samples_generated,
            "total_target": total_target,
            "progress_pct": progress_pct,
        }
