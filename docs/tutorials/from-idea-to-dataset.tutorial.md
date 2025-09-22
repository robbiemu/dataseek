# From Idea to Dataset: A Guide to Configuring Your First `dataseek` Mission

Welcome to `dataseek`! You've installed a powerful framework for building autonomous AI agents that can find, validate, and collect high-quality data from the web. But `dataseek` isn't a simple web scraper—it's something much more sophisticated.

`dataseek` is designed for missions where the data you need can't be found with a simple keyword search. It really shines when you need to find documents that possess a specific, often nuanced, quality or **characteristic** that requires deep content understanding to validate. Think of it this way: anyone can search for "articles about finance." But what if you need to find "quarterly earnings reports that contain forward-looking statements about supply chain risks"? That's where `dataseek` comes in. It's the difference between pattern matching and true content comprehension.

This tutorial will walk you through the process of designing and configuring a `dataseek` mission from scratch. While we'll be building a specific example, the steps and principles here form a generic recipe you can apply to your own unique data collection challenges.

## The Art of the Possible: What Can You Build?

Before we dive in, let's explore the kinds of sophisticated tasks `dataseek` is built for. The core pattern is always about finding a specific **characteristic** from a broader **topic**. Here are some ideas to spark your imagination:

* **Financial Document Analysis**: Find financial filings (`topic`) that contain forward-looking statements (`characteristic`) to train risk analysis models.
* **Legal Contract Review**: Search through legal agreements (`topic`) to find examples of specific clauses, like indemnification or non-compete provisions (`characteristic`).
* **Content Moderation Datasets**: Scour social media posts or news comments (`topic`) to find examples of hate speech or medical misinformation (`characteristic`) for training safety models.
* **Knowledge Base Curation**: Find technical articles (`topic`) that provide clear, step-by-step tutorials (`characteristic`) to populate a customer support bot.
* **Scientific Literature Mining**: Search academic databases (`topic`) for papers that use specific research methodologies (`characteristic`) to accelerate literature reviews.

For this tutorial, we're going to build out that last idea: **The Automated Research Assistant**. This mission, which we call MAC-AI (Methodologically-Aware Corpus for AI in Science), is the default configuration for `dataseek`. We'll walk through how we designed it so you can understand the process from start to finish.

Why did we choose this as our flagship example? Because it demonstrates `dataseek`'s unique strengths:

- **Complex Pattern Recognition**: Identifying research methodologies requires understanding document structure and distinguishing between papers that *mention* a method versus papers that *employ* it
- **Cross-Domain Intelligence**: The same methodology (like controlled experiments) appears very differently in physics versus psychology versus economics
- **Multi-Stage Validation**: It requires discovery, deep content inspection, contextual validation, and quality assessment—exercising the full `dataseek` architecture

## The Recipe for a Successful Mission

Every `dataseek` mission, regardless of its goal, follows the same three-step recipe. It's a process that moves from high-level strategy down to specific instructions for your AI agents.

### Step 1: The Proposal - Define Your "Why"

Before you write a single line of YAML, you must start with a clear plan. A mission proposal is a short document that forces you to think through your goals, justify your data selection criteria, and define what success looks like. This is the most important step!

A good proposal answers three fundamental questions:

1. **What problem are you trying to solve?** (e.g., "It's hard for scientists to find papers based on their research methods, not just their topics.")
2. **What data do you need to collect?** (e.g., "A corpus of papers, each validated as a true example of a specific research methodology.")
3. **Why is this a good use of `dataseek`?** (e.g., "Because identifying methodology requires deep content inspection and contextual understanding, not just keyword matching.")

For our MAC-AI mission, we wrote a detailed proposal that outlines these points and serves as the "North Star" for all the configuration choices we'll make in the next steps.

The proposal process also forced us to think carefully about **scope consistency**. Initially, we considered characteristics like "Randomized Controlled Trial" (very specific), "Longitudinal Study" (broad temporal category), and "Meta-Analysis or Systematic Review" (actually two different things). But this created uneven validation difficulty—some characteristics would showcase `dataseek`'s sophisticated capabilities while others would be closer to simple keyword matching.

Our final choice prioritizes methodological patterns that:
- Require the same level of contextual understanding to validate
- Appear across multiple scientific disciplines but manifest differently
- Demonstrate true pattern recognition rather than vocabulary matching

> 📖 **Read our example**: You can see the fully fleshed-out [proposal](docs/tutorials/default_research.proposal.md) for our default mission in the refined version we developed together. It demonstrates how to think through these strategic choices systematically.

### Step 2: The Mission Config - Define Your "What"

Once your proposal is clear, it's time to translate your goals into a `mission_config.yaml` file. This file is the blueprint for your data collection, defining the structure of your mission using two key concepts:

* **`characteristic`**: This is the core pattern or quality you are seeking. It's the "how" of your search—the methodological approach that requires sophisticated validation.
* **`topic`**: This is the domain you are searching within. It's the "what" of your search—the subject area that provides context but shouldn't determine the characteristic.

The `goals` section of your `mission_config.yaml` is where you bring these two concepts together. For each `characteristic`, you provide a `context`—a clear, detailed description that the `supervisor` agent will use to instruct the other agents.

Let's look at the configuration for our MAC-AI mission. This is the direct implementation of the strategy we defined in our proposal:

```yaml
# In examples/mac_ai_research_assistant/mission_config.yaml

missions:
  - name: "mac_ai_corpus_v1"
    target_size: 50 # Samples per characteristic
    synthetic_budget: 0.1 # We want mostly real-world data
    output_paths:
      samples_path: "datasets/mac_ai_corpus/samples"
      audit_trail_path: "datasets/mac_ai_corpus/PEDIGREE.md"
    tools:
      # (Your configured search tools go here)
    goals:
      - characteristic: "Controlled Experimental Design"
        context: "The goal is to find scientific papers that describe a study with a clear experimental control. The agent must verify that the paper's methodology section details at least one treatment/intervention group and a separate control group designed to isolate variables. It must distinguish papers that *employ* this method from those that merely *discuss* it in a literature review. This methodology appears across disciplines—from psychology lab studies to physics experiments to economics field trials—but always involves systematic comparison with controls."
        topics: ["Clinical Psychology", "Economics", "Public Health", "Physics", "Computer Science"]

      - characteristic: "Comparative Analysis Methodology"
        context: "The goal is to identify papers that conduct a systematic comparison between two or more distinct items, methods, theories, or approaches as their primary research activity. The agent must verify that the paper's methodology involves structured comparison with clear criteria, not just casual mentions of differences. This could manifest as algorithm benchmarking in computer science, treatment comparisons in medicine, or theoretical framework comparisons in social sciences."
        topics: ["Computer Science", "Clinical Medicine", "Literary Studies", "Engineering", "Economics"]

      - characteristic: "Primary Data Collection"
        context: "The goal is to find papers that gather original data through direct observation, measurement, surveys, or experiments, as opposed to analyzing existing datasets or conducting purely theoretical work. The agent must verify that the methodology section describes the original data gathering process—whether through laboratory measurements, field observations, surveys, interviews, or experimental procedures. This distinguishes empirical research from literature reviews, theoretical papers, or secondary data analysis."
        topics: ["Social Sciences", "Natural Sciences", "Engineering", "Public Health", "Psychology"]
```

Notice how each characteristic:
- Requires sophisticated validation (distinguishing *employing* from *mentioning*)
- Appears across multiple disciplines but manifests differently
- Has consistent complexity for demonstrating `dataseek`'s capabilities

The topic selection strategy is equally deliberate. We chose **broad topics** like "Clinical Psychology" rather than narrow ones like "Phase III diabetes trials" for two crucial reasons:

1. **Preventing Overfitting**: If we only searched for controlled experiments in diabetes research, an AI model might incorrectly learn that papers mentioning "placebo" and "glucose" are controlled experiments. By sourcing from diverse fields, we force the system to learn the *abstract pattern* of controlled experimentation.

2. **Ensuring Cross-Domain Generalizability**: This demonstrates that `dataseek` can learn methodological structures that transcend disciplinary boundaries—the hallmark of true pattern recognition.

### Step 3: The Prompts - Instruct Your Agents

With the "what" defined, the final step is to tell your AI agents *how* to behave. This is done in the `prompts.yaml` file. Each agent has a specific role, and a good prompt is the key to making them effective at their particular job.

#### The `research` Agent: The Hunter 🕵️

This agent's job is to find candidate documents. Your prompt should give it a clear search strategy and help it understand what makes a good candidate. For our MAC-AI mission, we tell it to behave like a research assistant who understands the difference between methodology and content:

```yaml
# In examples/mac_ai_research_assistant/prompts.yaml

research:
  base_prompt: >
    You are an Autonomous Research Assistant specializing in methodological discovery. Your mission is to find scientific papers that are prime examples of a specific research methodology ('{characteristic}') within the field of '{topic}'.
    
    Your primary focus is NOT the paper's conclusions or subject matter, but its research design and methodological structure. You're looking for papers where the target characteristic is central to how the research was conducted.
    
    A paper is a strong candidate if its 'Methods', 'Experimental Design', or 'Methodology' section clearly describes and implements the target characteristic. Pay special attention to papers that don't just mention the methodology in passing, but actually use it as their primary research approach.
    
    Search strategy: Focus on academic databases and look for methodological keywords in conjunction with the topic area. Prioritize papers where the abstract or introduction signals the use of the target methodology.
```

_idea: If we preferred, would could have asked it to give us focused excerpts using ellipses._

#### The `fitness` Agent: The Peer Reviewer 🧐

This is the most critical agent—it's the gatekeeper that decides if a document found by the `research` agent truly exemplifies the target characteristic. The key to a good `fitness` prompt is a **clear, structured validation rubric** that can distinguish between papers that mention versus employ the methodology:

```yaml
fitness:
  base_prompt: >
    You are a Quality Inspector for the MAC-AI project, acting as a methodological peer reviewer. Your task is to validate whether a candidate paper is a true exemplar of the target characteristic: '{characteristic}'.
    
    You must be rigorous and precise. Many papers will mention methodologies without actually employing them. Your job is to identify papers that genuinely USE the methodology as their primary research approach.
    
  strategy_block: >
    **Validation Rubric for: Controlled Experimental Design**

    1. **Identifies Control Condition**: Does the methodology section explicitly describe a control group, control condition, or baseline comparison designed to isolate variables? [PASS/FAIL]
    
    2. **Identifies Treatment/Intervention**: Does the text clearly define at least one experimental treatment, intervention, or manipulated variable that differs from the control? [PASS/FAIL]
    
    3. **Primary Research Method**: Is this experimental design the central focus of the paper's methodology, not just a minor component or preliminary study? [PASS/FAIL]
    
    4. **Methodological Implementation**: Does the paper provide sufficient detail about how the experimental control was implemented and maintained? [PASS/FAIL]

    **Decision Criteria**: A paper PASSES only if it meets ALL four criteria. If any criterion fails, the paper should be rejected regardless of topic relevance or overall quality.
    
    **Common False Positives to Reject**:
    - Papers that only mention controlled experiments in their literature review
    - Theoretical papers that discuss experimental design without implementing it
    - Observational studies that mention "control variables" (statistical controls ≠ experimental controls)
    - Review papers that summarize controlled experiments by others
```

The `fitness` agent's rubric is where the sophisticated validation happens. Notice how it's designed to catch common false positives and requires multiple forms of evidence before accepting a paper.

#### The `synthetic` Agent: The Author ✍️

This agent is your safety net. If the `research` agent can't find enough real-world examples, the `synthetic` agent steps in to create perfect, "textbook" examples of what you're looking for:

```yaml
synthetic:
  base_prompt: >
    You are a Synthetic Data Author for the MAC-AI project. Your task is to create realistic examples when real-world data is insufficient.
    
    Author a convincing **abstract and methodology section** of a *fictional* scientific paper that serves as an unambiguous, textbook example of the target characteristic ('{characteristic}') within the field of '{topic}').
    
    Your synthetic paper should:
    - Clearly demonstrate the methodology without any ambiguity
    - Use appropriate academic language and structure for the field
    - Include sufficient methodological detail for validation
    - Avoid real author names, institutions, or specific studies
    
    The goal is to create training data that exemplifies the characteristic so clearly that it helps AI models learn the pattern.
```

## Putting It All Together: From Theory to Practice

Now that we've walked through all three components, let's see how they work together in practice. When you run a `dataseek` mission:

1. **The `supervisor` agent** reads your `mission_config.yaml` and understands the overall goals
2. **The `research` agent** uses its prompts to hunt for candidate papers across your specified topics
3. **The `fitness` agent** rigorously evaluates each candidate using its validation rubric
4. **The `synthetic` agent** fills gaps with high-quality artificial examples when needed

The magic happens in the interaction between these agents. The `research` agent might find hundreds of candidates, but the `fitness` agent's rigorous validation ensures only true exemplars make it into your final dataset. This is what separates `dataseek` from simple web scraping—it's building understanding, not just collecting URLs.

## Why This Approach Works: The Science Behind the Design

Our MAC-AI mission demonstrates several key principles that make any `dataseek` mission successful:

**Methodological Abstraction**: By seeking the same methodology across different disciplines, we force the system to learn abstract patterns rather than field-specific vocabulary. A controlled experiment in psychology looks very different from one in physics, but the underlying methodological structure is the same.

**Validation Rigor**: The `fitness` agent's rubric distinguishes between mentioning and employing—a crucial distinction that requires deep content understanding. This is where `dataseek`'s multi-agent architecture really shines.

**Cross-Domain Generalization**: Our broad topic strategy prevents overfitting and demonstrates that the system is learning transferable methodological concepts, not narrow domain knowledge.

## Conclusion: You're Ready to Build!

You've now walked through the complete process of designing and configuring a `dataseek` mission. As you can see, the MAC-AI "Automated Research Assistant" mission wasn't created arbitrarily—it was the result of following this systematic three-step recipe:

1. **We started with a proposal** to define our goals and justify our approach
2. **We translated those goals** into a structured `mission_config.yaml` with consistent, cross-domain characteristics  
3. **We wrote detailed prompts** to give our AI agents the intelligence they needed to execute sophisticated validation

The key insight is that `dataseek`'s power comes from this thoughtful design process. The framework can only be as intelligent as the mission you design for it.

Now it's your turn. If you don't already have a good idea of what you want to build, or want to practice, take another look at the potential applications we discussed at the beginning—financial document analysis, legal contract review, content moderation, knowledge base curation. Pick one that interests you and try sketching out your own proposal:

- What problem would you solve?
- What characteristics would you look for? 
- What topics would provide the right diversity?
- How would your `fitness` agent distinguish between mentioning and employing?

Remember: start with the proposal. Get clear on your "why" before you worry about your "how." The most sophisticated AI agents in the world can't compensate for an unclear mission.

By following this recipe, you can harness the full power of `dataseek` to build your own autonomous data collection pipelines for any complex validation task you can imagine. The only limit is your creativity in defining what patterns matter for your use case.

Happy seeking! 🔍