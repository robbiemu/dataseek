#!/usr/bin/env python3
"""
Local CI Runner Script

This script reads the GitHub Actions workflow file and executes the steps
directly without generating intermediate scripts.
"""

import subprocess
import sys
from pathlib import Path

import yaml


def load_workflow(workflow_path):
    """Load and parse the GitHub Actions workflow YAML file."""
    with open(workflow_path) as file:
        return yaml.safe_load(file)


def extract_steps_for_job(workflow, job_name):
    """Extract steps for a specific job from the workflow."""
    jobs = workflow.get("jobs", {})
    if job_name not in jobs:
        raise ValueError(f"Job '{job_name}' not found in workflow")

    return jobs[job_name].get("steps", [])


def execute_step(step, step_num, total_steps):
    """Execute a single step from the workflow."""
    name = step.get("name", f"Step {step_num}")
    print(f"\n[{step_num}/{total_steps}] {name}")
    print("-" * (len(name) + 12))

    if "run" in step:
        run_command = step["run"]
        print(f"Executing: {run_command}")

        # Handle the virtual environment creation step specially
        if "uv venv" in run_command and ".venv" in run_command:
            # Check if virtual environment already exists
            if Path(".venv").exists():
                print("✓ Virtual environment already exists, skipping creation")
                return
            else:
                print("Creating virtual environment...")

        # Execute the command
        try:
            # For multi-line commands, we need to handle them properly
            if "\n" in run_command:
                # Create a temporary script to execute multi-line commands
                subprocess.run(
                    ["bash", "-c", run_command], capture_output=False, text=True, check=True
                )
            else:
                # Single line command
                subprocess.run(run_command, shell=True, capture_output=False, text=True, check=True)
            print("✓ Step completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Step failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    elif "uses" in step and "actions/checkout" in step["uses"]:
        print("✓ Checkout step - not needed for local execution")
    elif "uses" in step and "actions/setup-python" in step["uses"]:
        print("✓ Python setup step - assuming correct Python version is already installed")
    elif "uses" in step and "astral-sh/setup-uv" in step["uses"]:
        print("✓ uv setup step - assuming uv is already installed")
    else:
        print(f"Skipping step: {name}")


def run_job(job_name):
    """Run all steps for a specific job."""
    # Paths
    workflow_path = Path(".github/workflows/ci.yml")

    # Check if workflow exists
    if not workflow_path.exists():
        print(f"Error: Workflow file not found at {workflow_path}")
        sys.exit(1)

    # Load workflow
    workflow = load_workflow(workflow_path)

    # Extract steps for the job
    try:
        steps = extract_steps_for_job(workflow, job_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Running local CI checks for job: {job_name}")
    print("=" * (35 + len(job_name)))

    # Execute each step
    for i, step in enumerate(steps, 1):
        execute_step(step, i, len(steps))

    print(f"\n🎉 All checks for job '{job_name}' completed successfully!")


def list_jobs():
    """List all available jobs in the workflow."""
    workflow_path = Path(".github/workflows/ci.yml")
    if workflow_path.exists():
        workflow = load_workflow(workflow_path)
        jobs = workflow.get("jobs", {})
        print("Available jobs:")
        for job_name in jobs:
            print(f"  - {job_name}")
    else:
        print("ci.yml not found")


def main():
    if len(sys.argv) < 2:
        print("Usage: python local_ci_runner.py <job_name>")
        list_jobs()
        sys.exit(1)

    job_name = sys.argv[1]
    run_job(job_name)


if __name__ == "__main__":
    main()
