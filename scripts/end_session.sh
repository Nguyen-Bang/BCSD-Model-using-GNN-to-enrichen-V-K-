#!/bin/bash

##############################################################################
# Session End Script for BCSD Research Project
#
# Closes research session by:
# 1. Prompting for session summary and thesis-relevant insights
# 2. Updating session markdown with completion info
# 3. Extracting key points to thesis/methodology.tex (ALWAYS)
# 4. Displaying session statistics
# 5. Automatically committing and pushing changes to git repository
#
# Module: scripts/end_session.sh
# Owner: User Story 9 (US9) - Session Management System
# Tasks: T104-T105
#
# Usage:
#   ./scripts/end_session.sh
#   ./scripts/end_session.sh --date 2025-12-15
##############################################################################

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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
            echo -e "${RED}Unknown option: $1${NC}"
            echo -e "${YELLOW}Usage: $0 [--date YYYY-MM-DD]${NC}"
            exit 1
            ;;
    esac
done

# Get date
if [ -z "$SESSION_DATE" ]; then
    SESSION_DATE=$(date +%Y-%m-%d)
fi

# Session file path
PROTOCOLS_DIR="protocols"
SESSION_FILE="${PROTOCOLS_DIR}/session_${SESSION_DATE}.md"

# Check if session exists
if [ ! -f "${SESSION_FILE}" ]; then
    echo -e "${RED}✗ Session log not found: ${SESSION_FILE}${NC}"
    echo -e "${YELLOW}Have you started a session today with ./scripts/start_session.sh?${NC}"
    exit 1
fi

# Calculate session duration
SESSION_START=$(stat -c %Y "${SESSION_FILE}" 2>/dev/null || stat -f %B "${SESSION_FILE}")
SESSION_END=$(date +%s)
DURATION_SECONDS=$((SESSION_END - SESSION_START))
DURATION_HOURS=$((DURATION_SECONDS / 3600))
DURATION_MINUTES=$(((DURATION_SECONDS % 3600) / 60))

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Closing Session: ${SESSION_DATE}${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}Analyzing session log...${NC}"

# Extract session summary - look for recent substantial actions
SESSION_SUMMARY=$(awk '
/^## Actions Taken/,/^## / {
    if (/^### (Morning|Afternoon|Evening) Session/ || /^\*/) {
        getline
        if (!/^\*/ && !/^$/ && length($0) > 15) {
            gsub(/^[- *]+/, "")
            gsub(/^  */, "")
            if (length(summary) == 0 && length($0) > 15) {
                summary = $0
            }
        }
    }
    if (/^- [A-Z]/) {
        line = $0
        gsub(/^- /, "", line)
        if (length(summary) == 0 && length(line) > 15 && length(line) < 100) {
            summary = line
        }
    }
}
END { 
    if (length(summary) > 0) print summary
    else print "Implementation and code improvements"
}
' "${SESSION_FILE}")

# Extract achievements - look for completed items with checkmarks
ACHIEVEMENTS=()
while IFS= read -r line; do
    if [ -n "$line" ]; then
        ACHIEVEMENTS+=("$line")
    fi
done < <(awk '
/^## Actions Taken/,/^## / {
    # Look for completed items
    if (/✅|✓/) {
        line = $0
        gsub(/.*✅/, "", line)
        gsub(/.*✓/, "", line)
        gsub(/^[- *:]+/, "", line)
        gsub(/^  */, "", line)
        if (length(line) > 10 && length(line) < 120) {
            print line
        }
    }
    # Look for action items
    if (/^- (Implemented|Created|Added|Fixed|Enhanced|Removed|Updated|Completed|Built|Tested)/) {
        line = $0
        gsub(/^- /, "", line)
        if (length(line) > 15 && length(line) < 120) {
            print line
        }
    }
}
' "${SESSION_FILE}" | head -10)

# Extract blockers
BLOCKERS=()
while IFS= read -r line; do
    if [ -n "$line" ]; then
        BLOCKERS+=("$line")
    fi
done < <(awk '
/^## Actions Taken/,/^## Blockers/ {
    if (/❌|⚠️|TODO:|FIXME:|Issue:|Problem:|Failed|Error:/) {
        line = $0
        gsub(/.*❌/, "", line)
        gsub(/.*⚠️/, "", line)
        gsub(/^[- *]+/, "", line)
        gsub(/^  */, "", line)
        if (length(line) > 10 && length(line) < 120) {
            print line
        }
    }
}
' "${SESSION_FILE}" | head -5)

# Extract next priority from Pending Phases
NEXT_PRIORITY=$(awk '
/^### Pending Phases/,/^### / {
    if (/⏳/) {
        line = $0
        gsub(/.*⏳/, "", line)
        gsub(/^[- *]+/, "", line)
        if (length(line) > 10) {
            print line
            exit
        }
    }
}
END {
    if (NR == 0) print "Continue implementation and testing"
}
' "${SESSION_FILE}")

if [ -z "$NEXT_PRIORITY" ] || [ "$NEXT_PRIORITY" = "### Pending Phases" ]; then
    NEXT_PRIORITY="Continue implementation and testing"
fi

# Extract thesis-relevant insights from research notes
THESIS_INSIGHTS=()
while IFS= read -r line; do
    if [ -n "$line" ]; then
        THESIS_INSIGHTS+=("$line")
    fi
done < <(awk '
/Research|Methodology|Architecture|Design Decision|Experimental|Performance|Evaluation/ {
    if (!/^#/ && !/^\*\*Date/ && !/^\*\*Project/ && !/^\*\*Phase/) {
        line = $0
        gsub(/^[- *]+/, "", line)
        gsub(/^  */, "", line)
        if (length(line) > 25 && length(line) < 180 && line !~ /^(Morning|Afternoon|Evening)/) {
            print line
        }
    }
}
' "${SESSION_FILE}" | head -5)

echo -e "${GREEN}✓ Session data extracted${NC}"
echo ""
echo -e "${BLUE}Summary:${NC} ${SESSION_SUMMARY}"
echo -e "${BLUE}Achievements:${NC} ${#ACHIEVEMENTS[@]} items"
echo -e "${BLUE}Blockers:${NC} ${#BLOCKERS[@]} items"
echo -e "${BLUE}Next Priority:${NC} ${NEXT_PRIORITY}"
echo -e "${BLUE}Thesis Insights:${NC} ${#THESIS_INSIGHTS[@]} items"
echo ""

# Update session file with summary
echo -e "\n${BLUE}Updating session file...${NC}"

# Create temporary file with updated summary
TEMP_FILE=$(mktemp)

# Read file and update End-of-Session Summary section
awk -v summary="$SESSION_SUMMARY" \
    -v duration="${DURATION_HOURS}h ${DURATION_MINUTES}m" \
    -v next_priority="$NEXT_PRIORITY" '
/\*\*Summary\*\*: \[Brief overview/ {
    print "**Summary**: " summary
    next
}
/\*\*Duration\*\*: \[To be filled/ {
    print "- **Duration**: " duration
    next
}
/\*\*Next Session Priority\*\*:/ {
    print $0
    getline
    print "- " next_priority
    next
}
{ print }
' "${SESSION_FILE}" > "${TEMP_FILE}"

# Add achievements
if [ ${#ACHIEVEMENTS[@]} -gt 0 ]; then
    # Replace achievements section
    awk -v achievements="${ACHIEVEMENTS[*]}" '
    /\*\*Key Achievements\*\*:/ {
        print $0
        split(achievements, arr, " ")
        for (i in arr) {
            if (arr[i] != "") {
                print i ". " arr[i]
            }
        }
        # Skip next lines until blockers
        while (getline > 0) {
            if (/\*\*Blockers Remaining\*\*:/) {
                print $0
                break
            }
        }
        next
    }
    { print }
    ' "${TEMP_FILE}" > "${TEMP_FILE}.2"
    mv "${TEMP_FILE}.2" "${TEMP_FILE}"
fi

# Add blockers
if [ ${#BLOCKERS[@]} -gt 0 ]; then
    awk -v blockers="${BLOCKERS[*]}" '
    /\*\*Blockers Remaining\*\*:/ {
        print $0
        split(blockers, arr, " ")
        for (i in arr) {
            if (arr[i] != "") {
                print "- " arr[i]
            }
        }
        # Skip next lines until next priority
        while (getline > 0) {
            if (/\*\*Next Session Priority\*\*:/) {
                print $0
                break
            }
        }
        next
    }
    { print }
    ' "${TEMP_FILE}" > "${TEMP_FILE}.2"
    mv "${TEMP_FILE}.2" "${TEMP_FILE}"
fi

# Move updated file back
mv "${TEMP_FILE}" "${SESSION_FILE}"

echo -e "${GREEN}✓ Session file updated${NC}"

# Extract to LaTeX (ALWAYS - required for thesis documentation)
echo ""
echo -e "${BLUE}Extracting to LaTeX methodology chapter...${NC}"

THESIS_DIR="thesis"
METHODOLOGY_FILE="${THESIS_DIR}/methodology.tex"

# Create thesis directory if needed
mkdir -p "${THESIS_DIR}"

# Check if methodology file exists and has the development log section
if ! grep -q "Research Methodology and Development Log" "${METHODOLOGY_FILE}" 2>/dev/null; then
    cat >> "${METHODOLOGY_FILE}" << 'LATEX_EOF'

\section{Research Methodology and Development Log}

This section documents the iterative development process of the BCSD model,
providing transparency into the research methodology and implementation decisions.

LATEX_EOF
fi

# Append session summary to LaTeX
cat >> "${METHODOLOGY_FILE}" << LATEX_EOF

\subsection{Development Session: ${SESSION_DATE}}

\textbf{Summary:} ${SESSION_SUMMARY}

\textbf{Duration:} ${DURATION_HOURS}h ${DURATION_MINUTES}m

LATEX_EOF

# Add achievements if any
if [ ${#ACHIEVEMENTS[@]} -gt 0 ]; then
    echo '\textbf{Key Achievements:}' >> "${METHODOLOGY_FILE}"
    echo '\begin{itemize}' >> "${METHODOLOGY_FILE}"
    for achievement in "${ACHIEVEMENTS[@]}"; do
        echo "  \\item ${achievement}" >> "${METHODOLOGY_FILE}"
    done
    echo '\end{itemize}' >> "${METHODOLOGY_FILE}"
    echo "" >> "${METHODOLOGY_FILE}"
fi

# Add thesis-relevant insights if any
if [ ${#THESIS_INSIGHTS[@]} -gt 0 ]; then
    echo '\textbf{Research Insights:}' >> "${METHODOLOGY_FILE}"
    echo '\begin{itemize}' >> "${METHODOLOGY_FILE}"
    for insight in "${THESIS_INSIGHTS[@]}"; do
        echo "  \\item ${insight}" >> "${METHODOLOGY_FILE}"
    done
    echo '\end{itemize}' >> "${METHODOLOGY_FILE}"
fi

echo -e "${GREEN}✓ LaTeX updated: ${METHODOLOGY_FILE}${NC}"

# Display session statistics
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Session Statistics${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "Duration: ${DURATION_HOURS}h ${DURATION_MINUTES}m"
echo -e "Achievements: ${#ACHIEVEMENTS[@]}"
echo -e "Blockers: ${#BLOCKERS[@]}"
echo -e "Thesis Insights: ${#THESIS_INSIGHTS[@]}"
echo ""
echo -e "${GREEN}Session closed successfully!${NC}"
echo -e "Session log: ${SESSION_FILE}"
echo -e "LaTeX file: ${METHODOLOGY_FILE}"

# Git operations
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Git Operations${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo -e "${RED}✗ Not a git repository${NC}"
    echo -e "${YELLOW}Skipping git operations${NC}"
    exit 0
fi

# Stage all changes (respecting .gitignore)
echo -e "${BLUE}Staging all changes (respecting .gitignore)...${NC}"
git add -A

# Show what will be committed
echo ""
echo -e "${BLUE}Changes to be committed:${NC}"
git status --short

echo ""
echo -e "${BLUE}Committing and pushing changes...${NC}"

# Check if there are changes to commit
if git diff --cached --quiet; then
    echo -e "${YELLOW}No changes to commit${NC}"
    exit 0
fi

# Commit changes
COMMIT_MSG="Session ${SESSION_DATE}: ${SESSION_SUMMARY}"
git commit -m "${COMMIT_MSG}"

# Push to remote
CURRENT_BRANCH=$(git branch --show-current)

if git push origin "${CURRENT_BRANCH}"; then
    echo -e "${GREEN}✓ Changes pushed successfully to ${CURRENT_BRANCH}${NC}"
else
    echo -e "${YELLOW}⚠ Push failed - you may need to pull first or check your remote${NC}"
    echo -e "${YELLOW}Run: git pull --rebase && git push${NC}"
fi
