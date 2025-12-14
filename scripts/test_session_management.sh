#!/bin/bash

##############################################################################
# Test script for session management (start_session.sh and end_session.sh)
#
# Tests:
# 1. Session creation with start_session.sh
# 2. Session closeout with end_session.sh (automated responses)
# 3. Verification of session log updates
# 4. Verification of LaTeX methodology.tex updates
##############################################################################

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

TEST_DATE="2025-12-16"
PROTOCOLS_DIR="protocols"
SESSION_FILE="${PROTOCOLS_DIR}/session_${TEST_DATE}.md"
THESIS_FILE="thesis/methodology.tex"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Testing Session Management Scripts${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Clean up any existing test files
rm -f "${SESSION_FILE}"
cp "${THESIS_FILE}" "${THESIS_FILE}.backup" 2>/dev/null || true

# Test 1: Start session
echo -e "${BLUE}[Test 1] Starting session with date ${TEST_DATE}...${NC}"
./scripts/start_session.sh --date "${TEST_DATE}"

if [ -f "${SESSION_FILE}" ]; then
    echo -e "${GREEN}✓ Session file created${NC}"
else
    echo -e "${RED}✗ Session file NOT created${NC}"
    exit 1
fi

# Verify date replacement
if grep -q "${TEST_DATE}" "${SESSION_FILE}"; then
    echo -e "${GREEN}✓ Date correctly inserted in session file${NC}"
else
    echo -e "${RED}✗ Date NOT found in session file${NC}"
    exit 1
fi

# Test 2: Modify session file (simulate work)
echo ""
echo -e "${BLUE}[Test 2] Simulating session work...${NC}"

# Add some content to the session file
sed -i 's/^- $/- Implemented session management scripts (T102-T105)/' "${SESSION_FILE}"
sed -i '0,/^- $/s/^- $/- Created thesis LaTeX structure/' "${SESSION_FILE}"

echo -e "${GREEN}✓ Session file modified${NC}"

# Test 3: End session (automated input)
echo ""
echo -e "${BLUE}[Test 3] Ending session with automated responses...${NC}"

# Create input file for automated testing
INPUT_FILE=$(mktemp)
cat > "${INPUT_FILE}" << 'INPUTS'
Implemented complete session management system (US9)
Created start_session.sh and end_session.sh scripts
Set up thesis LaTeX structure with methodology chapter

None
Validate session management with real workflow
INPUTS

# Run end_session.sh with automated input
./scripts/end_session.sh --date "${TEST_DATE}" < "${INPUT_FILE}"

# Test 4: Verify session file updates
echo ""
echo -e "${BLUE}[Test 4] Verifying session file updates...${NC}"

if grep -q "Implemented complete session management system" "${SESSION_FILE}"; then
    echo -e "${GREEN}✓ Summary updated in session file${NC}"
else
    echo -e "${RED}✗ Summary NOT found in session file${NC}"
    exit 1
fi

if grep -q "Created start_session.sh and end_session.sh scripts" "${SESSION_FILE}"; then
    echo -e "${GREEN}✓ Achievements updated in session file${NC}"
else
    echo -e "${RED}✗ Achievements NOT found in session file${NC}"
    exit 1
fi

# Test 5: Verify LaTeX file updates
echo ""
echo -e "${BLUE}[Test 5] Verifying LaTeX methodology updates...${NC}"

if grep -q "Development Session: ${TEST_DATE}" "${THESIS_FILE}"; then
    echo -e "${GREEN}✓ Session date added to methodology.tex${NC}"
else
    echo -e "${RED}✗ Session date NOT found in methodology.tex${NC}"
    exit 1
fi

if grep -q "Implemented complete session management system" "${THESIS_FILE}"; then
    echo -e "${GREEN}✓ Summary added to methodology.tex${NC}"
else
    echo -e "${RED}✗ Summary NOT found in methodology.tex${NC}"
    exit 1
fi

# Show excerpt from updated thesis file
echo ""
echo -e "${BLUE}[Excerpt from thesis/methodology.tex]${NC}"
tail -20 "${THESIS_FILE}"

# Cleanup
echo ""
echo -e "${BLUE}[Cleanup]${NC}"
rm -f "${INPUT_FILE}"
rm -f "${SESSION_FILE}"
if [ -f "${THESIS_FILE}.backup" ]; then
    mv "${THESIS_FILE}.backup" "${THESIS_FILE}"
    echo -e "${GREEN}✓ Restored original methodology.tex${NC}"
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All tests passed!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
