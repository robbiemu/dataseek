## **Recipe: Configuring the Automated Research Assistant (MAC-AI Mission)** 🧑‍🔬

This guide details the setup for `dataseek`'s flagship demonstration: a mission to autonomously build a corpus of scientific papers classified by their research methodologies.

### **Ingredients**

  * A refactored `dataseek` repository, with "claimify"-specific code removed.
  * The `mission_config.yaml` and `prompts.yaml` files.
  * API keys for your chosen search tools (e.g., ArXiv, Wikipedia, a general web search provider).

### **Step 1: Structure Your Project for Extensibility**

The first step is to organize your configuration files to treat the MAC-AI mission as a self-contained, switchable example.

1.  **Create the Examples Directory**: In the root of the `dataseek` project, create a new directory named `examples/`.
2.  **Create the Mission Directory**: Inside `examples/`, create a directory for this specific mission: `mac_ai_research_assistant/`.
3.  **Move Configuration Files**: Place your `mission_config.yaml` and `prompts.yaml` files inside `examples/mac_ai_research_assistant/`.
4.  **Update Entry Point**: Ensure your main application script (`main.py` or similar) is updated to load mission configurations from this new path. This allows you to easily point `dataseek` to different missions in the future.

### **Step 2: Architect the Mission (`mission_config.yaml`)**

This file defines the *what* and *why* of your data collection. We will implement the two-dimensional strategy from the proposal (`Characteristic` + `Topic`).

#### **`goals` - Defining the Validation Targets**

Translate the abstract methodologies from the proposal into concrete `goals`. Each `characteristic` represents a core pattern the agent must validate.

```yaml
# In examples/mac_ai_research_assistant/mission_config.yaml

missions:
  - name: "mac_ai_corpus_v1"
    target_size: 50 # Samples per characteristic
    synthetic_budget: 0.1 # Allow 10% synthetic data for edge cases/testing
    output_paths:
      samples_path: "datasets/mac_ai_corpus/samples"
      audit_trail_path: "datasets/mac_ai_corpus/PEDIGREE.md"
    tools:
      # (Configure your arxiv_search, wikipedia_search, web_search tools here)
    goals:
      - characteristic: "Controlled Experimental Design"
        context: "The goal is to find scientific papers that describe a study with a clear experimental control. The agent must verify that the paper's methodology section details at least one treatment/intervention group and a separate control group to isolate variables. It must distinguish papers that *employ* this method from those that merely *discuss* it in a literature review."
        topics: ["Clinical Psychology", "Economics", "Public Health", "Physics", "Computer Science"]

      - characteristic: "Comparative Analysis"
        context: "The goal is to identify papers that conduct a systematic comparison between two or more distinct items, methods, or theories. The agent must validate that the paper's core contribution is the analysis of similarities and differences, supported by a structured comparative framework. The comparison should be the primary research activity, not a minor part of the discussion."
        topics: ["Literary Studies", "Algorithm Performance", "Clinical Treatments", "Engineering Materials"]

      - characteristic: "Primary Data Collection"
        context: "The goal is to distinguish papers based on original, empirical work from those that are purely theoretical or based on secondary analysis. The agent must find evidence of data being collected directly by the authors, whether through surveys, lab measurements, field observations, or other first-hand methods. The 'Methods' section is the key validation area."
        topics: ["Sociology", "Astronomy", "Field Biology", "Market Research"]

```

### **Step 3: Instruct the Agents (`prompts.yaml`)**

This is where you give each agent its specific role and intelligence for the MAC-AI mission.

#### **`research` Agent Prompt: The Method Hunter**

  * **Instruction**: Guide the agent to behave like a research assistant. It needs to look past the abstract and into the methodological core of the paper.
  * **YAML Snippet**:
    ```yaml
    # In examples/mac_ai_research_assistant/prompts.yaml

    research:
      base_prompt: >
        You are an Autonomous Research Assistant. Your mission is to find scientific papers that are prime examples of a specific research methodology ('{characteristic}') within the topic of '{topic}'.

        Your primary focus is not the paper's conclusion, but its structure and research design. You are looking for the methodological 'DNA'. A paper is a good candidate if its 'Methods' or 'Experimental Design' section clearly describes the target characteristic. Do not be fooled by papers that only mention the method in passing.
    ```

#### **`fitness` Agent Prompt: The Peer Reviewer**

  * **Instruction**: This is the most critical prompt. It must contain a clear, structured rubric that the agent will use for validation. This turns the agent from a simple classifier into a rigorous peer reviewer.
  * **YAML Snippet (Example for "Controlled Experimental Design")**:
    ```yaml
    fitness:
      base_prompt: >
        You are a Quality Inspector for the MAC-AI project, acting as a peer reviewer. Your task is to validate whether a candidate paper is a true exemplar of its target characteristic: '{characteristic}'.

        You are not evaluating the paper's scientific merit, only its methodological structure. Use the following rubric to make your decision. You MUST respond only with the required JSON object.
      strategy_block: >
        **Rubric for: Controlled Experimental Design**

        1.  **Identifies Control Group**: Does the 'Methods' section explicitly describe a control or placebo group that does not receive the primary intervention? [PASS/FAIL]
        2.  **Identifies Treatment Group**: Does the text clearly define at least one experimental group that receives the intervention? [PASS/FAIL]
        3.  **Primary Research Activity**: Is this experimental design the central focus of the paper's methodology, not just a minor sub-study or a mention in the literature review? [PASS/FAIL]

        **Decision**: A paper PASSES only if all three criteria are met. In your reasoning, you must justify your decision for each rubric point.
    ```

#### **`synthetic` Agent Prompt: The Methodological Author**

  * **Instruction**: When research fails, this agent creates a perfect, "textbook" example of the methodology. This is useful for creating gold-standard training data.
  * **YAML Snippet**:
    ```yaml
    synthetic:
      base_prompt: >
        You are a Synthetic Data Author for the MAC-AI project. The research agent has failed to find a real-world paper for '{characteristic}' in the topic of '{topic}'.

        Your task is to author a realistic-looking **abstract and methods section** of a *fictional* scientific paper that is a perfect, unambiguous example of the target characteristic. Ensure your output mimics the structure and formal tone of an academic paper.
    ```

### **Step 4: Execute the Mission**

With the configuration files built according to the MAC-AI proposal, you are ready to run the mission.

1.  **Launch `dataseek`**: Point the application to the `mac_ai_research_assistant` mission configuration.
2.  **Monitor the Process**: Observe as the agents collaborate—the `research` agent proposes candidates, and the `fitness` agent rigorously validates them against its rubric.
3.  **Inspect the Output**: The final result will be a high-quality, validated corpus of scientific papers, ready to be used for training advanced AI models.

You have now successfully configured `dataseek` to perform a sophisticated, autonomous scientific literature review—a powerful demonstration of its capabilities.