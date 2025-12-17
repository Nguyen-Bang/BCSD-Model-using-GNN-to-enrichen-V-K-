#!/bin/bash

##############################################################################
# Session Start Script for BCSD Research Project
#
# Creates structured Markdown session log in protocols/ directory.
# Follows research constitution principles for documentation-first development.
#
# Module: scripts/start_session.sh
# Owner: User Story 9 (US9) - Session Management System
# Tasks: T102-T103
#
# Usage:
#   ./scripts/start_session.sh
#   ./scripts/start_session.sh --date 2025-12-15
##############################################################################

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
SESSION_DATE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --date)
            SESSION_DATE="$2"
            shift 2
            ;;
        *)
            echo -e "${YELLOW}Usage: $0 [--date YYYY-MM-DD]${NC}"
            exit 1
            ;;
    esac
done

# Get date (default to today if not specified)
if [ -z "$SESSION_DATE" ]; then
    SESSION_DATE=$(date +%Y-%m-%d)
fi

# Session file path
PROTOCOLS_DIR="protocols"
SESSION_FILE="${PROTOCOLS_DIR}/session_${SESSION_DATE}.md"

# Create protocols directory if it doesn't exist
mkdir -p "${PROTOCOLS_DIR}"

# Check if session already exists
if [ -f "${SESSION_FILE}" ]; then
    echo -e "${YELLOW}⚠ Session log already exists: ${SESSION_FILE}${NC}"
    echo -e "${YELLOW}To continue existing session, just open the file.${NC}"
    echo -e "${YELLOW}To create a new session, manually rename the existing file.${NC}"
    exit 0
fi

# Create session template (T103)
cat > "${SESSION_FILE}" << 'EOF'
# Research Session: SESSION_DATE_PLACEHOLDER

**Date**: SESSION_DATE_PLACEHOLDER  
**Project**: BCSD Model using GNN to Enrich V-K  
**Phase**: Implementation Phase

---

## Session Goals

- [ ] Define objectives for today's work
- [ ] Review current implementation status
- [ ] Plan next tasks from tasks.md

---

## Current Status

### Completed Phases
- ✅ Phase 1: Setup (T001-T009)
- ✅ Phase 2: Foundational - Test Binary Validation (T010-T014)
- ✅ Phase 3: User Story 1 - Preprocessing (T015-T026)
- ✅ Phase 4: User Story 2 - Dataset Implementation (T027-T036)
- ✅ Phase 5: User Story 3 - GNN Encoder (T037-T045)
- ✅ Phase 6: User Story 4 - Custom Attention (T046-T058)
- ✅ Phase 7: User Story 5 - BERT Integration (T059-T070)
- ✅ Phase 8: User Story 6 - Demonstration Notebook (T071-T073)
- ✅ Phase 9: User Story 7 - Training Infrastructure (T074-T089)
- ✅ Phase 10: User Story 8 - Vectorization & Inference (T090-T101)
- ⚠️ Phase 13: Polish (T118-T119 complete, T120-T121 deferred, T122-T123 complete)

### Pending Phases
- ⏳ Phase 11: User Story 9 - Session Management (T102-T110)
- ⏳ Phase 12: User Story 10 - Document Review Agent (T111-T117)

### Active Context
*Document what you're currently working on*

- 

---

## Actions Taken

### Morning Session
*Document activities here as you work*

- 

### Afternoon Session
*Continue logging activities*

- 

---

## Implementation Details (What, Why, How)

### Functions/Components Created/Modified

#### Component: [Name]
**What**: [Brief description of what was implemented]

**Why**: [Rationale for this implementation choice]
- Problem being solved:
- Alternatives considered:
- Decision rationale:

**How**: [Technical implementation details]
- Key parameters and their purposes:
- Integration with existing system:
- Expected behavior/output:

**Contribution to System**: [How this contributes to overall BCSD pipeline]

---

## Technical Decisions & Architecture Choices

### Decision Log
*Record any architectural or implementation decisions with full justification*

#### Decision 1: [Title]
**Context**: [What problem or requirement led to this decision]

**Options Considered**:
1. Option A: [Description]
   - Pros: 
   - Cons:
2. Option B: [Description]
   - Pros:
   - Cons:

**Decision**: [Chosen option]

**Rationale**: [Why this option was chosen]
- Performance implications:
- Maintainability considerations:
- Alignment with research goals:

**Implementation Details**:
- Parameters/configurations used:
- Integration points:
- Expected impact on pipeline:

---

## Code Changes

### Files Modified
*List files changed with brief description of WHY each change was needed*

- `path/to/file.py`: 
  - **What changed**: 
  - **Why**: 
  - **Impact on system**:

### Tests Added/Modified
*List test files affected*

- 

---

## Issues Encountered

### Blockers
*Critical issues preventing progress*

1. 

### Warnings/Notes
*Non-blocking issues or observations*

1. 

---

## Outcomes

### Completed Tasks
*Reference task IDs from tasks.md*

- [ ] TXXX: Task description

### Validation Results
*Test results, metrics, observations*

- 

### Knowledge Gained
*New insights or learnings*

- 

---

## Next Session Planning

### Immediate Next Steps
1. 
2. 
3. 

### Questions for Review
- 

### Dependencies Needed
- 

---

## Thesis Integration Notes

### Methodology Chapter
*Key points to extract for thesis/methodology.tex*

- 

### Results/Discussion
*Observations relevant for results chapter*

- 

### Figures/Tables Generated
*Reference any visualizations created*

- 

---

## Session Metrics

- **Duration**: [To be filled at session end]
- **Tasks Completed**: 0
- **Tests Passing**: [To be verified]
- **Code Quality**: [Run quality checks]

---

## End-of-Session Summary

*Fill this section when ending the session using end_session.sh*

**Summary**: [Brief overview of session accomplishments]

**Key Achievements**:
1. 
2. 
3. 

**Blockers Remaining**:
- 

**Next Session Priority**:
- 
EOF

# Replace placeholder date
sed -i "s/SESSION_DATE_PLACEHOLDER/${SESSION_DATE}/g" "${SESSION_FILE}"

# Success message
echo -e "${GREEN}✓ Session started successfully!${NC}"
echo -e "${BLUE}Session log created: ${SESSION_FILE}${NC}"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "1. Open ${SESSION_FILE} in your editor"
echo "2. Fill in 'Session Goals' section"
echo "3. Document your work as you progress"
echo "4. Run ./scripts/end_session.sh when done"
echo ""
echo -e "${YELLOW}Remember: Document as you go, not at the end!${NC}"
