# Guide: Curating Datasets with the Data Seek Agent

## 1. Introduction

Manually collecting, cleaning, and auditing a high-quality dataset for training and evaluating LLM systems is a significant bottleneck. The **Data Seek Agent** is an automated tool designed to solve this problem. It transforms the open-ended task of "finding data" into a systematic, goal-directed, and reproducible process.

This guide will walk you through how to use the agent to curate a custom corpus tailored to your specific needs.

**What the Agent Does:**

*   **Deconstructs Goals:** Translates a high-level mission into a concrete plan.
*   **Finds and Fetches Data:** Systematically searches the web for relevant text.
*   **Cleans and Validates:** Extracts core content and runs "fitness checks" to ensure the data is useful.
*   **Creates an Audit Trail:** Automatically generates a `PEDIGREE.md` file to track the provenance of every piece of data.

The agent is configured to use the `openai/gpt-5-mini` model by default, but this can be changed in the configuration files.

## 2. Prerequisites

Before you begin, ensure you have installed the DataSeek project and its dependencies:

```bash
# From the root of the project repository
uv sync
```

## 3. The Core Concept: The Mission Plan

The entire behavior of the Data Seek Agent is controlled by a single YAML configuration file: the **Mission Plan**. Think of this file as the "brain" of the agent. You don't write code to change the agent's goals; you simply describe your desired outcome in the mission plan, and the agent adapts its strategy to achieve it.

The default mission plan is located at `settings/mission_config.yaml`.

## 4. Quick Start: A Simple Example

Let's run the agent with a simple mission to see it in action.

#### **Step 1: Define a Simple Mission**

Create a new file named `settings/simple_mission.yaml` and add the following content. This mission tells the agent to find just 10 text samples related to "Quantum Computing."

```yaml
# in settings/simple_mission.yaml
missions:
  - name: "quantum_computing_corpus"
    target_size: 10 # Find 10 total samples
    synthetic_budget: 0.1 # Allow 10% (1 sample) to be invented if needed
    goals:
      - characteristic: "Verifiability"
        topics: ["Quantum Computing Research Abstracts"]
```

#### **Step 2: Execute the Agent**

You have two options for running the agent:

**Option A: Using the Terminal User Interface (TUI) - Recommended**

Run the interactive TUI that shows real-time progress, agent conversation, and statistics:

```bash
dataseek-tui settings/simple_mission.yaml
```

The TUI provides:
- Real-time progress tracking with ETA and throughput metrics
- Live agent conversation showing prompts and responses
- Pause/resume controls (`p` key)
- Mission status and error monitoring
- Interactive controls (`q` to quit, `r` to restart)

**Option B: Using the Command-Line Interface**

Run the agent from your terminal in batch mode:

```bash
dataseek --mission settings/simple_mission.yaml
```

Both methods will execute the same underlying agent. The TUI is recommended for interactive use as it provides better visibility into the agent's progress and allows for real-time control.

#### **Step 3: Review the Output**

Once the agent completes its mission, you will find the following new files and directories:

*   **`examples/data/datasets/tier1/`**: Contains the raw, cleaned text files downloaded from the web. Each file is named with the date and topic, e.g., `2025-09-05_Quantum_Computing_Research_Abstracts.txt`.
*   **`examples/data/datasets/tier2/`**: Contains the final, curated corpus files, created by sampling from the Tier 1 data. This is the data you will use in the next steps of the pipeline.
*   **`examples/PEDIGREE.md`**: The audit trail. This file contains a detailed log of where each piece of data came from, when it was sourced, and for what purpose.

## 5. Deep Dive: The `mission_config.yaml` File

The mission plan is highly configurable. Here is a breakdown of the key parameters you can use to control the agent's behavior.

```yaml
# The root of the file is a list of missions. The agent will execute them sequentially.
missions:
  - name: "production_corpus" # A descriptive name for the mission.

    # The total number of text blocks to collect per component/characteristic.
    target_size: 150

    # The proportion of the target_size that the agent is allowed to
    # invent using an LLM if it cannot find good examples online.
    # 0.2 means 20% of the samples can be synthetic.
    synthetic_budget: 0.2

    # A list of specific goals. The agent will work to fulfill each one.
    goals:
      - # Goal 1: Find text to test the "Verifiability" characteristic.
        characteristic: "Verifiability"

        # A list of specific topics to search for within this goal.
        topics:
          - "news reports"
          - "scientific abstracts"
          - "financial statements"
          - "opinion editorials" # Good for negative examples

      - # Goal 2: Find text to test "Self-containment" (Disambiguation).
        characteristic: "Self-containment"
        topics:
          - "political analysis"
          - "historical narratives"
          - "multi-paragraph news stories"
      
      - # Goal 3: Find text to test "Atomicity" (Decomposition).
        characteristic: "Atomicity"
        topics:
          - "legal documents"
          - "policy summaries"
          - "contract clauses"
```

## 6. The Full Workflow: From Seek to Compiled Artifact

The Data Seek Agent is the first step in a three-step pipeline to create a final, optimized DSPy artifact.

**Step 1: Curate the Corpus with the Seek Agent**
Run the agent with your configured mission plan to produce the Tier 2 raw text corpus.

```bash
# Option A: Interactive TUI (recommended for monitoring progress)
dataseek-tui settings/mission_config.yaml

# Option B: Command-line batch mode
dataseek --mission settings/mission_config.yaml
```

**Step 2: Generate the "Gold Standard" Dataset**
Use the `generate-dataset` command to convert the raw text from the agent into a structured `.jsonl` training file. This step uses a powerful "teacher" model to create the ideal outputs for each example.

```bash
dataseek generate-dataset \
  --input-file examples/data/datasets/tier2/selection_raw.txt \
  --output-file data/selection_train.jsonl \
  --teacher-model gpt-4o \
  --k-window-size 2
```

**Step 3: Compile the Production Artifact**
Finally, use the `compile` command to take the gold-standard training set and produce the final, optimized `.json` artifact.

```bash
dataseek compile \
  --config configs/selection_config.yaml \
  --trainset data/selection_train.jsonl \
  --output-path compiled_prompts/selection.json
```

## 6.5. Advanced Configuration: Tool Prefetching and Validation

The Data Seek Agent includes advanced configuration options for controlling how search tools behave, including prefetching and validation features that make the research process more robust.

### Tool Configuration Options

You can configure tool behavior by adding a `tools` section to your mission configuration:

```yaml
missions:
  - name: "production_corpus"
    target_size: 150
    synthetic_budget: 0.2
    tools:
      web_search:
        pre_fetch_pages: true
        pre_fetch_limit: 5
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
      arxiv_search:
        pre_fetch_pages: true
        pre_fetch_limit: 3
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
      wikipedia_search:
        pre_fetch_pages: true
        pre_fetch_limit: 3
        validate_urls: true
        retry_on_failure: true
        max_retries: 2
    goals:
      - characteristic: "Verifiability"
        topics: ["news reports", "scientific abstracts"]
```

### Configuration Parameters

Each tool supports the following configuration parameters:

- **`pre_fetch_pages`**: Enable/disable prefetching and validation of search results
- **`pre_fetch_limit`**: Number of search results to validate (default: 3)
- **`validate_urls`**: Enable/disable URL validation (default: true)
- **`retry_on_failure`**: Enable/disable automatic retry with expanded results (default: true)
- **`max_retries`**: Maximum number of retry attempts (default: 2)

### How Prefetching Works

When prefetching is enabled:

1. **Initial Search**: The search tool returns a list of results
2. **Validation**: The system validates the accessibility of the first N results (where N = `pre_fetch_limit`)
3. **Filtering**: Inaccessible URLs are filtered out before being passed to the researcher
4. **Automatic Retry**: If all validated results are inaccessible, the system automatically retries with expanded results
5. **Smart Deduplication**: Previously validated bad URLs are excluded from retry attempts

This process makes the research workflow more robust by eliminating dead links before the researcher attempts to fetch them, while maintaining search intent by not modifying queries.

## 7. Conclusion

The Data Seek Agent is a powerful tool for creating high-quality, auditable datasets. By investing time in creating a thoughtful **Mission Plan**, you can automate the most time-consuming part of the optimization process. Remember to start with small, targeted missions and iterate as you refine your data requirements.
