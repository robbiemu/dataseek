import logging
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class MissionDetailsParser:
    """Parser for mission configuration files."""

    @staticmethod
    def get_mission_details_from_file(
        mission_plan_path: str,
    ) -> dict[str, Any] | None:
        """
        Parse mission_config.yaml to extract mission names and target sizes.

        Args:
            mission_plan_path: Path to the mission configuration file

        Returns:
            Dictionary with mission names and targets, or None if parsing fails
        """
        try:
            with open(mission_plan_path, encoding="utf-8") as f:
                content = f.read()

                # Remove comment header if present
                if content.startswith("#"):
                    first_newline = content.find("\n")
                    if first_newline != -1:
                        content = content[first_newline + 1 :]

                mission_plan = yaml.safe_load(content)

                if not mission_plan or "missions" not in mission_plan:
                    logger.warning(f"No missions found in {mission_plan_path}")
                    return None

                return MissionDetailsParser._extract_mission_info(mission_plan)

        except FileNotFoundError:
            logger.error(f"Mission plan file not found: {mission_plan_path}")
            return None
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {mission_plan_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {mission_plan_path}: {e}")
            return None

    @staticmethod
    def _extract_mission_info(mission_plan: dict[str, Any]) -> dict[str, Any]:
        """Extract mission information from parsed YAML."""
        mission_names = []
        mission_targets = {}

        for mission in mission_plan.get("missions", []):
            name = mission.get("name")
            if not name:
                logger.warning("Found mission without name, skipping")
                continue

            mission_names.append(name)

            # Calculate total samples
            target_size = mission.get("target_size", 0)
            goals = mission.get("goals", [])
            total_samples = target_size * len(goals)
            mission_targets[name] = total_samples

        return {"mission_names": mission_names, "mission_targets": mission_targets}


# Legacy function for backward compatibility
def get_mission_details_from_file(mission_plan_path: str) -> dict[str, Any] | None:
    """Legacy wrapper for mission details parsing."""
    return MissionDetailsParser.get_mission_details_from_file(mission_plan_path)
