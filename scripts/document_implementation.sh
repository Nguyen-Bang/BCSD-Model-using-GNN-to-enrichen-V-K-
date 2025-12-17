#!/bin/bash

##############################################################################
# Implementation Documentation Helper
#
# Use this script to quickly document implementation decisions in the
# current session protocol with proper "what, why, how" structure.
#
# Usage:
#   ./scripts/document_implementation.sh
#
# This will prompt for:
#   - Component name
#   - What was implemented
#   - Why this approach was chosen
#   - How it works (technical details)
#   - Contribution to overall system
##############################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Find today's session file
SESSION_DATE=$(date +%Y-%m-%d)
SESSION_FILE="protocols/session_${SESSION_DATE}.md"

if [ ! -f "${SESSION_FILE}" ]; then
    echo -e "${YELLOW}No session file found for today. Please run ./scripts/start_session.sh first.${NC}"
    exit 1
fi

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Document Implementation Decision${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Prompt for information
echo -e "${GREEN}Component/Function name (e.g., GATEncoder, InfoNCELoss):${NC}"
read -p "> " COMPONENT_NAME

echo ""
echo -e "${GREEN}WHAT: Brief description of what was implemented:${NC}"
read -p "> " WHAT_DESC

echo ""
echo -e "${GREEN}WHY: Main reason/problem this solves (one line):${NC}"
read -p "> " WHY_MAIN

echo ""
echo -e "${GREEN}WHY: Alternatives considered (comma-separated, or press Enter to skip):${NC}"
read -p "> " ALTERNATIVES

echo ""
echo -e "${GREEN}WHY: Decision rationale (why this approach over others):${NC}"
read -p "> " RATIONALE

echo ""
echo -e "${GREEN}HOW: Key parameters and their purposes (comma-separated):${NC}"
read -p "> " PARAMETERS

echo ""
echo -e "${GREEN}HOW: Integration with existing system:${NC}"
read -p "> " INTEGRATION

echo ""
echo -e "${GREEN}HOW: Expected behavior/output:${NC}"
read -p "> " BEHAVIOR

echo ""
echo -e "${GREEN}CONTRIBUTION: How this contributes to overall BCSD pipeline:${NC}"
read -p "> " CONTRIBUTION

# Find the Implementation Details section and append
TEMP_FILE=$(mktemp)

# Check if Implementation Details section exists
if ! grep -q "^## Implementation Details" "${SESSION_FILE}"; then
    # Add the section if it doesn't exist
    awk '
    /^## Actions Taken/ {
        print
        print ""
        print "---"
        print ""
        print "## Implementation Details (What, Why, How)"
        print ""
        print "### Functions/Components Created/Modified"
        print ""
        next
    }
    { print }
    ' "${SESSION_FILE}" > "${TEMP_FILE}"
    mv "${TEMP_FILE}" "${SESSION_FILE}"
fi

# Append the documentation
cat >> "${SESSION_FILE}" << EOF

#### Component: ${COMPONENT_NAME}
**What**: ${WHAT_DESC}

**Why**: ${WHY_MAIN}
- Problem being solved: ${WHY_MAIN}
EOF

if [ -n "$ALTERNATIVES" ]; then
    echo "- Alternatives considered: ${ALTERNATIVES}" >> "${SESSION_FILE}"
fi

cat >> "${SESSION_FILE}" << EOF
- Decision rationale: ${RATIONALE}

**How**: Technical implementation
- Key parameters: ${PARAMETERS}
- Integration: ${INTEGRATION}
- Expected behavior: ${BEHAVIOR}

**Contribution to System**: ${CONTRIBUTION}

EOF

echo ""
echo -e "${GREEN}✓ Implementation documented in ${SESSION_FILE}${NC}"
echo -e "${YELLOW}This will be automatically extracted to LaTeX when you end the session.${NC}"
