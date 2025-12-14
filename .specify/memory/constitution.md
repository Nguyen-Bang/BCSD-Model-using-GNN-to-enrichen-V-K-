<!--
SYNC IMPACT REPORT
==================
Version Change: 1.0.0 → 1.1.0
Ratification Date: 2025-12-13
Last Amendment: 2025-12-13

Amendment Rationale:
Added Principle VI (Context7 Integration) to establish mandatory use of Context7 MCP
tools for code generation, setup/configuration, and library documentation tasks.
This is a MINOR version bump as it adds a new principle without breaking existing ones.

Modified Principles:
- NEW (v1.0.0): Research Documentation First
- NEW (v1.0.0): Reproducible Pipeline
- NEW (v1.0.0): Experiment Tracking
- NEW (v1.0.0): Modular Architecture
- NEW (v1.0.0): Scientific Rigor
- NEW (v1.1.0): Context7 Integration

Added Sections:
- Research Workflow (v1.0.0)
- Technical Standards (v1.0.0)

Templates Status:
✅ plan-template.md - Reviewed, aligns with constitution
✅ spec-template.md - Reviewed, aligns with constitution
✅ tasks-template.md - Reviewed, aligns with constitution
✅ checklist-template.md - Present in workspace
✅ agent-file-template.md - Present in workspace

Follow-up TODOs: None
-->

# BCSD Model Constitution

**Binary Code Similarity Detection using GNN-enriched BERT**

This constitution governs the development of a Bachelor's thesis project that implements a complete machine learning pipeline for binary code similarity detection while documenting all research processes, experiments, and learnings.

## Core Principles

### I. Research Documentation First

**All experimental work MUST be documented as it happens.**

- Every experiment, hypothesis, result (positive or negative) MUST be recorded
- Document tools used, strategies attempted, and rationale for decisions
- Maintain a research journal capturing what worked, what didn't, and why
- Failed approaches are as valuable as successful ones—document both
- Include timestamps, versions, and environmental context for reproducibility

**Rationale**: This is a Bachelor's thesis project where the research process itself is as important as the final implementation. Documentation enables knowledge transfer, reproducibility, and demonstrates scientific methodology.

### II. Reproducible Pipeline

**Each pipeline component MUST be independently executable and verifiable.**

- Angr disassembly stage MUST run standalone with clear inputs/outputs
- BERT encoder MUST function independently with documented interfaces
- GNN model MUST be testable without requiring full pipeline execution
- All components MUST specify dependencies, versions, and configuration
- Provide sample inputs and expected outputs for each stage

**Rationale**: Enables debugging individual stages, facilitates iterative development, and allows component-level validation during thesis defense.

### III. Experiment Tracking

**All experiments MUST be tracked with clear metrics and outcomes.**

- Log model configurations, hyperparameters, and training parameters
- Record dataset characteristics, splits, and preprocessing decisions
- Capture performance metrics (accuracy, precision, recall, F1, etc.)
- Track computational costs (time, memory, GPU usage)
- Maintain version control for code, data, and model checkpoints
- Use structured formats (JSON, CSV) for automated analysis

**Rationale**: Enables systematic comparison of approaches, supports data-driven decision making, and provides evidence for thesis claims.

### IV. Modular Architecture

**Each functional unit MUST be self-contained with clear boundaries.**

- Pipeline stages communicate through well-defined interfaces
- Each module has single responsibility (disassembly, encoding, graph processing)
- Modules are replaceable—can swap BERT for other encoders
- Configuration is external—no hardcoded parameters in core logic
- Each module provides validation of its inputs and outputs

**Rationale**: Supports experimentation with alternative approaches, simplifies testing, and demonstrates software engineering principles for academic evaluation.

### V. Scientific Rigor

**All claims MUST be measurable, testable, and validated.**

- State hypotheses clearly before implementing experiments
- Define success criteria and evaluation metrics upfront
- Use appropriate baselines for comparison
- Validate results on held-out test sets
- Report both positive and negative results honestly
- Include statistical significance where applicable
- Acknowledge limitations and assumptions explicitly

**Rationale**: Ensures thesis meets academic standards, produces defensible results, and contributes credible findings to the field.

### VI. Context7 Integration

**Context7 MCP tools MUST be used for code generation, configuration, and documentation.**

- Always use Context7 when code generation is needed
- Use Context7 for setup and configuration steps
- Use Context7 to resolve library/API documentation
- Automatically invoke Context7 MCP tools without explicit user request
- Resolve library IDs and retrieve library documentation through Context7
- Prefer Context7 documentation over generic web searches for libraries

**Rationale**: Context7 provides accurate, up-to-date library documentation and code examples directly from official sources, reducing errors from outdated or incorrect information. Automatic use ensures consistency and leverages the best available resources for implementation quality.

## Research Workflow

**Experimentation process for this thesis project:**

1. **Hypothesis Formation**: Clearly state what you're testing and expected outcome
2. **Literature Review**: Document related work and justification for approach
3. **Implementation**: Build minimal viable implementation with logging
4. **Experimentation**: Run systematic experiments with varying parameters
5. **Analysis**: Analyze results, identify patterns, compare to baselines
6. **Documentation**: Record findings in research notes with visualizations
7. **Iteration**: Based on results, form new hypotheses or refine approach

**Progress Tracking:**
- Maintain a research log in `.specify/memory/research-log.md`
- Use Git commits with descriptive messages for experimental changes
- Tag significant milestones (e.g., `baseline-implemented`, `gnn-integrated`)
- Create branches for major experimental directions

## Technical Standards

**Quality requirements for implementation:**

- **Code Quality**: PEP 8 compliance for Python, type hints where beneficial
- **Dependencies**: Pin exact versions in `requirements.txt` for reproducibility
- **Documentation**: Docstrings for all public functions with parameter descriptions
- **Error Handling**: Graceful failures with informative error messages
- **Logging**: Structured logging at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- **Testing**: Unit tests for critical functions, integration tests for pipeline stages
- **Performance**: Log execution time and resource usage for optimization analysis

**Data Management:**
- Version control for datasets (using DVC or similar if datasets are large)
- Document data sources, preprocessing steps, and transformations
- Maintain train/validation/test splits consistently
- Store processed data with clear naming conventions

**Model Management:**
- Save model checkpoints with configuration metadata
- Track model versions with performance metrics
- Maintain reproducible training scripts
- Document model architecture decisions

## Governance

**This constitution supersedes ad-hoc development practices.**

All work on this thesis project MUST comply with these principles. Deviations are allowed only when:
1. The reason is documented with justification
2. The limitation or constraint is explicitly acknowledged
3. The deviation is temporary with a plan to resolve

**Amendment Process:**
- Constitution can be amended as the project evolves
- Amendments MUST increment version using semantic versioning:
  - MAJOR: Backward-incompatible principle changes or removals
  - MINOR: New principle additions or significant clarifications
  - PATCH: Wording improvements, typo fixes, minor clarifications
- Each amendment MUST include rationale and impact assessment
- Amendment history is tracked in this document's commit history

**Compliance Review:**
- Before major commits, verify alignment with core principles
- During thesis writing, reference this constitution to structure methodology section
- Use templates in `.specify/templates/` that align with these principles

**Version**: 1.1.0 | **Ratified**: 2025-12-13 | **Last Amended**: 2025-12-13
