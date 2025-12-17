# Research Documentation Guide

This guide explains how to properly document implementation decisions and technical choices for automatic extraction to the thesis LaTeX document.

## Overview

The BCSD project uses a **documentation-first approach** where implementation decisions, technical rationale, and architecture choices are captured in session protocols and automatically extracted to the thesis methodology chapter.

## Documentation Workflow

### 1. Start Session
```bash
./scripts/start_session.sh
```

Creates `protocols/session_YYYY-MM-DD.md` with structured template.

### 2. Document As You Implement

The session protocol includes several critical sections for thesis-relevant documentation:

#### **Implementation Details (What, Why, How)**

This is the **most important section** for thesis documentation. When you implement a function, class, or component, document:

**What**: What was implemented
- Brief description of the component/function
- Its role in the system

**Why**: Rationale for implementation choices
- Problem being solved
- Alternatives considered
- Decision rationale (why this approach over others)
- Alignment with research goals

**How**: Technical implementation
- Key parameters and their purposes
- Integration points with existing system
- Expected behavior/output
- Performance considerations

**Contribution to System**: How this fits into overall BCSD pipeline

#### Example Documentation:

```markdown
#### Component: InfoNCELoss

**What**: Contrastive loss function for Siamese network training

**Why**: Need to learn similarity between binary code representations
- Problem being solved: Model must distinguish between similar and dissimilar code
- Alternatives considered: Triplet loss, NT-Xent, simple cosine loss
- Decision rationale: InfoNCE provides in-batch negatives efficiently, scales well with batch size, theoretically grounded in mutual information maximization

**How**: Technical implementation
- Key parameters: temperature=0.07 (controls similarity distribution sharpness)
- Integration: Used in JointLoss with MLM loss, weighted by lambda_contrastive
- Expected behavior: Lower loss when embeddings of same function are closer, higher when different functions are compared

**Contribution to System**: Enables the model to learn meaningful embeddings where functionally similar code (same function, different compilation) has high cosine similarity, critical for the similarity detection task
```

### 3. Use Documentation Helper (Optional)

For quick documentation during implementation:

```bash
./scripts/document_implementation.sh
```

This script will prompt for all required information and format it correctly in your session protocol.

### 4. Technical Decisions Section

Document broader architectural decisions:

```markdown
#### Decision 1: Using GAT over GCN for Graph Encoding

**Context**: Need to encode CFG structure into fixed-size representation

**Options Considered**:
1. Graph Convolutional Network (GCN)
   - Pros: Simpler, faster
   - Cons: Uniform neighbor weighting, less expressive
2. Graph Attention Network (GAT)
   - Pros: Learned attention weights, more expressive, handles varying node degrees better
   - Cons: Slightly more complex, higher computational cost

**Decision**: GAT with 4 attention heads

**Rationale**: 
- Performance implications: Attention mechanism learns importance of different CFG edges (e.g., loop back-edges vs. sequential flow)
- Maintainability: Well-supported in PyTorch Geometric
- Alignment with research goals: Attention weights provide interpretability for thesis discussion

**Implementation Details**:
- Parameters: 3 layers, 4 heads per layer, hidden_dim=256
- Integration: Output connects to BERT graph prefix mechanism
- Expected impact: Better graph representations for complex control flow patterns
```

### 5. End Session

```bash
./scripts/end_session.sh
```

This script will:
1. **Analyze the entire protocol** for thesis-relevant content
2. **Extract** what/why/how implementation details
3. **Extract** technical decisions and rationale
4. **Extract** architecture choices and their justification
5. **Append** everything to `thesis/methodology.tex` with proper LaTeX formatting
6. **Commit and push** all changes to git

## What Gets Extracted to LaTeX

The `end_session.sh` script performs comprehensive extraction:

### 1. Project Status
- Lists completed phases
- Shows progress through implementation

### 2. Implementation Work
- All components created/modified
- What/Why/How for each component
- Integration points

### 3. Architecture and Design Decisions
- Decision context and options
- Chosen approach with rationale
- Expected impact on system

### 4. Decision Rationale
- WHY behind each choice
- Alignment with research goals
- Performance implications

### 5. Technical Specifications
- Key parameters and their purposes
- Configurations used
- Expected behavior

### 6. Research Insights
- Novel findings
- Unexpected results
- Lessons learned

## Best Practices

### ✅ DO:
- **Document during implementation**, not after
- **Explain WHY** you chose specific parameter values (e.g., "temperature=0.07 based on SimCLR paper")
- **Compare alternatives** you considered
- **Link to system**: Explain how component fits into overall pipeline
- **Be specific**: "3 GAT layers with 4 heads" not "several GAT layers"

### ❌ DON'T:
- Don't just say "implemented X" - explain WHY and HOW
- Don't skip rationale - future you (writing thesis) will need it
- Don't use vague descriptions - be quantitative
- Don't forget to document parameter choices

## Integration with Thesis Writing

The automatically extracted documentation goes into `thesis/methodology.tex` in the "Research Methodology and Development Log" section.

This provides:
- **Transparency**: Shows iterative research process
- **Justification**: Documents why choices were made
- **Reproducibility**: Others can understand and replicate decisions
- **Thesis Material**: Pre-written methodology content

## Quick Reference

### Essential Questions to Answer

For every significant implementation:

1. **WHAT** did you implement?
2. **WHY** this approach? (What problem does it solve?)
3. **WHAT ELSE** did you consider? (Alternatives)
4. **WHY THIS** over alternatives? (Rationale)
5. **HOW** does it work? (Parameters, integration, behavior)
6. **HOW DOES IT CONTRIBUTE** to the overall system?

### File Locations

- Session protocols: `protocols/session_YYYY-MM-DD.md`
- LaTeX output: `thesis/methodology.tex`
- Helper scripts:
  - `scripts/start_session.sh` - Start new session
  - `scripts/document_implementation.sh` - Quick documentation helper
  - `scripts/end_session.sh` - End session and extract to LaTeX

## Example Session Flow

```bash
# Morning: Start session
./scripts/start_session.sh

# During implementation: Document decisions
# Edit protocols/session_2025-12-16.md
# Add to "Implementation Details" section with What/Why/How

# Optional: Use helper for quick documentation
./scripts/document_implementation.sh

# End of day: Extract to LaTeX and commit
./scripts/end_session.sh
# - Analyzes entire protocol
# - Extracts thesis-relevant content
# - Appends to methodology.tex
# - Commits and pushes to git
```

## Tips for Thesis-Quality Documentation

1. **Think like a researcher, not just a developer**
   - Don't just document what you did
   - Document why it matters for the research

2. **Quantify everything**
   - "batch_size=16" not "small batch size"
   - "temperature=0.07" not "low temperature"

3. **Cite related work when applicable**
   - "Based on SimCLR's temperature scaling approach"
   - "Following the GAT architecture from Veličković et al."

4. **Document negative results too**
   - "Tried Adam optimizer but SGD converged faster"
   - "Initially used GCN but GAT performed 15% better"

5. **Link to research questions**
   - "This addresses RQ1: Can graph structure improve similarity detection"
   - "Validates hypothesis that attention mechanism helps with complex CFGs"

---

**Remember**: Good documentation today = Easy thesis writing tomorrow!
