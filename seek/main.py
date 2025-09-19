"""
CLI entry point for the Data Seek Agent.
"""

import os
import sys
from typing import Any

import typer
import yaml
from langgraph.checkpoint.sqlite import SqliteSaver

from .config import get_active_seek_config, load_seek_config, set_active_seek_config
from .graph import build_graph
from .mission_runner import MissionRunner
from .tools import _HAVE_LIBCRAWLER

# Force unbuffered output for live progress updates when supported
_reconf = getattr(sys.stdout, "reconfigure", None)
if callable(_reconf):
    _reconf(line_buffering=True)


def get_mission_names(mission_plan_path: str) -> list[str]:
    """Loads the mission YAML and returns a list of mission names."""
    try:
        with open(mission_plan_path) as f:
            content = f.read()
            if content.startswith("#"):
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_plan = yaml.safe_load(content)
        return [mission.get("name") for mission in mission_plan.get("missions", [])]
    except Exception:
        return []


def load_mission_plan(
    mission_plan_path: str = "settings/mission_config.yaml",
    mission_name: str | None = None,
) -> dict:
    """
    Load and parse the mission plan YAML file to calculate target sample counts.

    Args:
        mission_plan_path: Path to the mission plan YAML file

    Returns:
        Dictionary with mission plan information including total target samples
    """
    try:
        with open(mission_plan_path) as f:
            # Skip the first line which is a comment
            content = f.read()
            if content.startswith("#"):
                # Find the first newline and skip to after it
                first_newline = content.find("\n")
                if first_newline != -1:
                    content = content[first_newline + 1 :]
            mission_plan = yaml.safe_load(content)

        # Calculate total target samples
        total_samples = 0
        missions = mission_plan.get("missions", [])

        if mission_name:
            for mission in missions:
                if mission.get("name") == mission_name:
                    target_size = mission.get("target_size", 0)
                    goals = mission.get("goals", [])
                    total_samples += target_size * len(goals)
                    break
        else:
            for mission in missions:
                target_size = mission.get("target_size", 0)
                goals = mission.get("goals", [])
                # Each goal is a characteristic with topics
                # Total samples = target_size * number_of_characteristics
                total_samples += target_size * len(goals)

        return {
            "missions": missions,
            "total_samples_target": total_samples,
            "mission_plan_path": mission_plan_path,
        }
    except Exception as e:
        print(f"Warning: Could not load mission plan from {mission_plan_path}: {e}")
        # Return default values
        return {
            "missions": [],
            "total_samples_target": 1200,  # Default from documentation
            "mission_plan_path": mission_plan_path,
        }


def setup_observability(seek_config: dict[str, Any]) -> None:
    """Configure LangSmith tracing via LangChain/LangGraph environment variables.

    Avoids LiteLLM's custom logger (which requires a running event loop in-thread)
    and instead relies on LANGCHAIN_TRACING_V2 for observability.
    """
    # Read env vars (support both LANGCHAIN_* and LANGSMITH_* forms)
    env_api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    env_tracing = os.getenv("LANGCHAIN_TRACING_V2") or os.getenv("LANGSMITH_TRACING")
    env_endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
    env_project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")

    # Read config overrides
    observability_config = seek_config.get("observability", {})
    cfg_api_key = observability_config.get("api_key")
    cfg_tracing = observability_config.get("tracing")
    cfg_endpoint = observability_config.get("endpoint")
    cfg_project = observability_config.get("project")

    # Resolve effective settings (env takes precedence)
    api_key = env_api_key or cfg_api_key
    tracing = env_tracing if env_tracing is not None else cfg_tracing
    endpoint = env_endpoint or cfg_endpoint
    project = env_project or cfg_project

    # Respect explicit disable
    if tracing and str(tracing).lower() in {"false", "0", "no", "off"}:
        print("LangSmith tracing explicitly disabled.")
        return

    if api_key:
        # Normalize both namespaces for compatibility
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGSMITH_API_KEY", api_key)
        if endpoint:
            os.environ.setdefault("LANGCHAIN_ENDPOINT", endpoint)
            os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
        if project:
            os.environ.setdefault("LANGCHAIN_PROJECT", project)
            os.environ.setdefault("LANGSMITH_PROJECT", project)
        # Enable tracing with LangChain v2
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        print("Observability configured (LangChain/LangGraph tracing enabled).")
    else:
        if tracing and str(tracing).lower() in {"true", "1", "yes", "on"}:
            print(
                "Warning: Tracing requested but no API key provided (LANGCHAIN_API_KEY/LANGSMITH_API_KEY)."
            )
        else:
            print("Note: No LangSmith configuration found. Skipping observability setup.")


def run_agent_process(
    mission_name: str | None = None,
    recursion_limit: int | None = None,
    max_samples: int | None = None,
    resume_from: str | None = None,
    seek_config_path: str | None = None,
    mission_plan_path: str = "settings/mission_config.yaml",
    use_robots: bool = True,
) -> None:
    """Main function to set up and run the seek agent mission."""
    with SqliteSaver.from_conn_string("checkpoints/mission_checkpoints.db") as checkpointer:
        if resume_from:
            # If resuming, mission_name is not required from the command line
            mission_config: dict[str, Any] = (
                {}
            )  # The runner will load the config from the saved state
        else:
            # If starting a new mission, mission_name is required
            available_missions = get_mission_names(mission_plan_path)
            if not mission_name:
                print("❌ Error: No mission specified. Use --mission or --resume-from.")
                print("Available missions are:")
                for name in available_missions:
                    print(f"  - {name}")
                return

            if mission_name not in available_missions:
                print(f"❌ Error: Mission '{mission_name}' not found in mission plan.")
                return

            mission_plan = load_mission_plan(mission_plan_path, mission_name)
            mission_config = next(
                (m for m in mission_plan.get("missions", []) if m.get("name") == mission_name),
                {},
            )

            if not mission_config:
                print(f"❌ Error: Could not load configuration for mission '{mission_name}'.")
                return

        # Load mission-specific configuration once and set active
        seek_config = load_seek_config(seek_config_path, use_robots=use_robots)
        set_active_seek_config(seek_config)

        # Set up observability with the seek configuration
        # Use dict form for observability
        setup_observability(seek_config.to_dict())
        sys.stdout.flush()

        if not _HAVE_LIBCRAWLER:
            print("\033[93mWarning: `libcrawler` is not installed.\033[0m")
            print("Crawler functionality will be disabled.")
            print("To enable, run: pip install -e .[crawler]\n")
            sys.stdout.flush()

        # For a new mission, calculate the target. For a resumed one, this will be loaded.
        if not resume_from:
            mission_plan = load_mission_plan(mission_plan_path, mission_name)
            total_samples_target = mission_plan.get("total_samples_target", 0)
            print(
                f"📊 Mission Plan: Targeting {total_samples_target} samples for mission '{mission_name}'"
            )
            sys.stdout.flush()

        # Determine the recursion limit for each sample generation cycle
        if recursion_limit is None:
            # Use active configuration established earlier
            seek_config = get_active_seek_config()
            recursion_limit = seek_config.get("recursion_per_sample", 27)
            print(f"🔐 Using recursion limit from mission config: {recursion_limit}")
        else:
            print(f"🔐 Using CLI-provided recursion limit: {recursion_limit}")
        sys.stdout.flush()

        try:
            app = build_graph(checkpointer=checkpointer, seek_config=seek_config.to_dict())
            mission_runner = MissionRunner(
                checkpointer=checkpointer,
                app=app,
                mission_config=mission_config,
                seek_config=seek_config.to_dict(),
                resume_from_mission_id=resume_from,
            )
            mission_runner.run_mission(recursion_limit=recursion_limit, max_samples=max_samples)

        except Exception as e:
            import traceback

            print(f"\n\033[91mAn error occurred during the mission: {e}\033[0m")
            print(f"Error type: {type(e).__name__}")
            print(f"Traceback: {traceback.format_exc()}")
        finally:
            print("\nConnection closed")


# This function is the actual entry point called by the console script
def run(
    mission_name: str | None = typer.Option(
        None, "--mission", "-m", help="The name of the mission to run."
    ),
    recursion_limit: int = typer.Option(None, "--recursion-limit"),
    max_samples: int = typer.Option(
        None, "--max-samples", help="Maximum number of samples to generate."
    ),
    resume_from: str | None = typer.Option(
        None, "--resume-from", help="The mission ID to resume from."
    ),
    config: str = typer.Option(
        "config/seek_config.yaml",
        "--config",
        "-c",
        help="Path to the agent configuration file.",
    ),
    mission_config: str = typer.Option(
        "settings/mission_config.yaml",
        "--mission-config",
        help="Path to the mission configuration file.",
    ),
    no_robots: bool = typer.Option(False, "--no-robots", help="Ignore robots.txt rules"),
) -> None:
    # Handle --no-robots flag
    use_robots = not no_robots

    # Set the global use_robots setting
    from .config import set_global_use_robots

    set_global_use_robots(use_robots)

    print(f"[DEBUG] Setting global use_robots to {use_robots}")

    # Load mission-specific configuration
    seek_config = load_seek_config(config, use_robots=use_robots)
    set_active_seek_config(seek_config)

    print(f"[DEBUG] Loaded seek config with use_robots={seek_config.get('use_robots')}")

    # Set up observability with the seek configuration
    setup_observability(seek_config.to_dict())

    run_agent_process(
        mission_name=mission_name,
        recursion_limit=recursion_limit,
        max_samples=max_samples,
        resume_from=resume_from,
        seek_config_path=config,
        mission_plan_path=mission_config,
        use_robots=use_robots,  # Pass the use_robots parameter
    )


# Create the CLI app that will be used by the entry point
cli_app = typer.Typer()
cli_app.command()(run)


def main() -> None:
    """Entry point for the dataseek console script."""
    cli_app()


if __name__ == "__main__":
    main()
