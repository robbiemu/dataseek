import re
from collections.abc import Generator
from dataclasses import dataclass


@dataclass
class ProgressUpdate:
    completed: int
    target: int


@dataclass
class NewMessage:
    role: str
    content: str


@dataclass
class ErrorMessage:
    message: str


@dataclass
class SyntheticSampleUpdate:
    """Event for when a synthetic sample is generated."""

    count: int


@dataclass
class RecursionStepUpdate:
    """Event for tracking recursion steps."""

    current_step: int
    total_steps: int


class AgentOutputParser:
    """Parses agent log output and emits structured events."""

    def __init__(self) -> None:
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\\[0-?]*[ -/]*[@-~])")
        self.synthetic_count = 0  # Track cumulative synthetic samples
        # Suppress immediate duplicate lines that often occur from dual loggers
        self._last_clean_line: str | None = None

    def parse_line(
        self,
        line: str,
    ) -> Generator[
        ProgressUpdate | NewMessage | ErrorMessage | SyntheticSampleUpdate | RecursionStepUpdate
    ]:
        """Parses a single line of agent output."""
        clean_line = self.ansi_escape.sub("", line).strip()
        if not clean_line:
            return

        # Deduplicate immediate repeats of the exact same content
        if clean_line == self._last_clean_line:
            return
        self._last_clean_line = clean_line

        # 1. Progress patterns: "📊 Mission Progress: 123/1200 samples (10.3%)"
        progress_match = re.search(
            r"📊 Mission Progress: (\d+)/(\d+) samples \((\d+\.\d+)%\)", clean_line
        )
        if progress_match:
            yield ProgressUpdate(
                completed=int(progress_match.group(1)),
                target=int(progress_match.group(2)),
            )
            return

        # 2. Alternative progress pattern: "📊 Progress: 123/1200 samples (10.3%)"
        progress_match2 = re.search(r"📊 Progress: (\d+)/(\d+) samples \((\d+\.\d+)%\)", clean_line)
        if progress_match2:
            yield ProgressUpdate(
                completed=int(progress_match2.group(1)),
                target=int(progress_match2.group(2)),
            )
            return

        # 3. Sample archived: "📊 Sample #123 archived (10.3% complete) - Source: research/synthetic"
        sample_match = re.search(r"📊 Sample #(\d+) archived.*Source: (\w+)", clean_line)
        if sample_match:
            completed = int(sample_match.group(1))
            source_type = sample_match.group(2)
            yield ProgressUpdate(
                completed=completed, target=completed
            )  # Update with current progress
            # If it's a synthetic sample, emit a synthetic sample update
            if source_type == "synthetic":
                self.synthetic_count += 1
                yield SyntheticSampleUpdate(count=self.synthetic_count)
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 4. Mission plan info: "📊 Mission Plan: Targeting 1200 samples"
        if clean_line.startswith("📊 Mission Plan:"):
            target_match = re.search(r"Targeting (\d+) samples", clean_line)
            if target_match:
                target = int(target_match.group(1))
                yield ProgressUpdate(completed=0, target=target)
            yield NewMessage(role="info", content=clean_line)
            return

        # 5. Recursion limit info: "🔐 Using CLI-provided recursion limit: 30"
        recursion_match = re.search(r"recursion limit: (\d+)", clean_line)
        if recursion_match:
            total_steps = int(recursion_match.group(1))
            yield RecursionStepUpdate(current_step=0, total_steps=total_steps)
            yield NewMessage(role="system", content=clean_line)
            return

        # 6. Recursion step info: "🔄 Supervisor: Recursion step 1/30"
        recursion_step_match = re.search(r"🔄 Supervisor: Recursion step (\d+)/(\d+)", clean_line)
        if recursion_step_match:
            current_step = int(recursion_step_match.group(1))
            total_steps = int(recursion_step_match.group(2))
            yield RecursionStepUpdate(
                current_step=current_step,
                total_steps=total_steps,
            )
            # Don't yield NewMessage here to avoid cluttering the conversation with recursion steps
            return

        # 6. Agent status: "🤖 Data Seek Agent is ready..."
        if clean_line.startswith("🤖 "):
            yield NewMessage(role="system", content=clean_line)
            return

        # 7. Supervisor messages: "🔍 Supervisor: ..." (excluding end decisions)
        if clean_line.startswith("🔍 Supervisor:") and "Decided on 'end'" not in clean_line:
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 8. Warning/error messages: "⚠️ Supervisor: ..."
        if clean_line.startswith("⚠️ "):
            # Check if this is a filtering warning that should not be treated as an error
            if "Filtering out" in clean_line:
                yield NewMessage(role="debug", content=clean_line)
            else:
                yield ErrorMessage(message=clean_line)
            return

        # 9. Success messages: "✅ Supervisor: ..." (excluding end decisions)
        if clean_line.startswith("✅ Supervisor:") and "Decided on 'end'" not in clean_line:
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 10. Research node messages: "🔍 RESEARCH NODE DEBUG:"
        if "RESEARCH NODE DEBUG:" in clean_line or "RESEARCH NODE" in clean_line:
            yield NewMessage(role="tool", content=clean_line)
            return

        # 11. Node execution: "-> Executing Node: RESEARCH"
        if clean_line.startswith("-> Executing Node:"):
            yield NewMessage(role="system", content=clean_line)
            return

        # 12. Tool calls: "- Decided to call tools: web_search(...)"
        if "Decided to call tools:" in clean_line:
            yield NewMessage(role="tool", content=clean_line)
            return

        # 13. Agent responses: "- Responded: ..."
        if clean_line.strip().startswith("- Responded:"):
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 14. Final completion: "🏁 Agent has finished the task."
        if clean_line.startswith("🏁 "):
            yield NewMessage(role="system", content=clean_line)
            return

        # 15. State creation messages
        if "Creating new state for this thread" in clean_line:
            yield NewMessage(role="system", content=clean_line)
            return

        # 16. Warning messages (without emoji)
        if clean_line.startswith("Warning:"):
            yield NewMessage(role="debug", content=clean_line)
            return

        # 17. Demote pydantic help line to debug, not error
        if "errors.pydantic.dev" in clean_line:
            yield NewMessage(role="debug", content=clean_line)
            return

        # 18. Error patterns
        clean_lower = clean_line.lower()
        if any(keyword in clean_lower for keyword in ["error", "failed", "exception", "traceback"]):
            yield ErrorMessage(message=clean_line)
            return

        # 19. Tool operation messages
        if any(
            keyword in clean_lower
            for keyword in ["search", "found", "fetching", "downloading", "crawling"]
        ):
            yield NewMessage(role="tool", content=clean_line)
            return

        # 20. Creation/generation messages
        if any(
            keyword in clean_lower
            for keyword in ["generating", "created", "completed", "writing", "saved"]
        ):
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 21. Thread/connection messages
        if any(keyword in clean_line for keyword in ["thread", "Thread", "config"]):
            yield NewMessage(role="system", content=clean_line)
            return

        # 22. Supervisor end decision - detect when agent decides to end
        if "Decided on 'end'" in clean_line or "Routing to END" in clean_line:
            yield NewMessage(role="system", content="Agent decided to end conversation")
            return

        # 23. Agent responses that contain synthetic sample information
        if "synthetic sample" in clean_line.lower() or "source: synthetic" in clean_line.lower():
            # Only increment if we haven't already tracked this from the archive pattern
            if "archived" not in clean_line.lower():
                self.synthetic_count += 1
                yield SyntheticSampleUpdate(count=self.synthetic_count)
            yield NewMessage(role="assistant", content=clean_line)
            return

        # 24. Default: system message for anything else
        yield NewMessage(role="system", content=clean_line)
