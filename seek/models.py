# models.py
"""
Data models for the Data Seek agent.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SeekAgentMissionPlanNodeConfig(BaseModel):
    """Configuration for a single node in the Seek Agent's mission plan."""

    name: str = Field(..., description="The name of the node.")
    model: str = Field(..., description="The LLM model to use for this node.")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature for the model."
    )
    max_tokens: int = Field(
        default=4096, ge=1, description="Maximum tokens for the model's output."
    )


class SeekAgentResearchNodeConfig(BaseModel):
    """Configuration for the Research Node's ReAct loop."""

    max_iterations: int = Field(
        default=7,
        ge=1,
        le=50,
        description="Maximum number of iterations for the ReAct loop in the research node.",
    )


class SeekAgentNodesConfig(BaseModel):
    """Container for node-specific configurations."""

    research: SeekAgentResearchNodeConfig = Field(
        default_factory=SeekAgentResearchNodeConfig
    )
    # We can add configs for other nodes here later if needed


class SeekAgentMissionPlanToolConfig(BaseModel):
    """Configuration for individual tools used by the Data Seek Agent."""

    pre_fetch_pages: bool = Field(
        default=False,
        description="Whether to pre-fetch and validate URLs returned by search results.",
    )
    pre_fetch_limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of pages to pre-fetch from search results.",
    )
    validate_urls: bool = Field(
        default=True,
        description="Whether to validate URLs for accessibility before using them.",
    )
    retry_on_failure: bool = Field(
        default=True,
        description="Whether to retry failed operations with modified parameters.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum number of retries for failed operations.",
    )


class SeekAgentMissionPlanConfig(BaseModel):
    """Defines the mission plan for the Data Seek Agent."""

    goal: str = Field(
        ..., description="The high-level goal for the data seeking mission."
    )
    nodes: List[SeekAgentMissionPlanNodeConfig] = Field(
        default_factory=list,
        description="Configuration for each node in the agent graph.",
    )
    tools: Dict[str, SeekAgentMissionPlanToolConfig] = Field(
        default_factory=dict,
        description="Configuration for individual tools used in the mission.",
    )

    def get_node_config(self, name: str) -> Optional[SeekAgentMissionPlanNodeConfig]:
        """Retrieve the configuration for a specific node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_tool_config(
        self, tool_name: str
    ) -> Optional[SeekAgentMissionPlanToolConfig]:
        """Retrieve the configuration for a specific tool by name."""
        return self.tools.get(tool_name)


class SeekAgentWriterConfig(BaseModel):
    """Configuration for the agent's output writers."""

    samples_path: str = Field(
        default="./seek_samples",
        description="Path to store raw, cleaned text dumps.",
    )
    tier2_path: str = Field(
        default="./seek_tier2_curated",
        description="Path to store curated, fitness-checked subsets.",
    )
    audit_trail_path: str = Field(
        default="./PEDIGREE.md", description="Path to the audit trail file."
    )


class SeekAgentConfig(BaseModel):
    """Main configuration for the Data Seek Agent."""

    search_provider: Optional[str] = Field(
        default="duckduckgo/search",
        description="The default search provider to use (e.g., 'duckduckgo/search', 'brave/search')",
    )
    mission_plan: SeekAgentMissionPlanConfig
    writer: SeekAgentWriterConfig = Field(default_factory=SeekAgentWriterConfig)
    # Configuration for the persistent checkpointer
    checkpointer_path: str = Field(
        default=".checkpointer.sqlite",
        description="Path to the SQLite database for the checkpointer.",
    )
    nodes: SeekAgentNodesConfig = Field(
        default_factory=SeekAgentNodesConfig,
        description="Configuration specific to each agent node.",
    )

    # The initial prompt to send to the agent to start the mission
    initial_prompt: Optional[str] = Field(
        default=None,
        description="The initial prompt to send to the agent to start the mission. If not provided, the TUI will ask for it interactively.",
    )

    # Whether to use robots.txt
    use_robots: bool = Field(
        default=True, description="Whether to use robots.txt rules when fetching URLs"
    )


class FitnessReport(BaseModel):
    """A structured report from the FitnessAgent evaluating a source document."""
    passed: bool = Field(description="True if the document is approved, False if it is rejected.")
    reason: str = Field(description="A brief justification for the decision, explaining why the document was approved or rejected.")
