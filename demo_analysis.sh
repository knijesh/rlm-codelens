#!/bin/bash

################################################################################
# RLM-Codelens Self-Scan Demo
#
# Demonstrates architecture analysis by scanning this repository itself.
# No external data, API keys, or GitHub access needed.
#
# Usage:
#   ./demo_analysis.sh
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

OUTPUTS_DIR="outputs"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RLM-Codelens Demo - Self Analysis                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}This demo scans rlm-codelens itself — no API keys needed.${NC}"
echo ""

mkdir -p $OUTPUTS_DIR

################################################################################
# Phase 1: Scan Repository
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 1: Scanning Repository${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

SCAN_FILE="$OUTPUTS_DIR/self_scan.json"

echo -e "${BLUE}Running:${NC} uv run rlmc scan-repo . --name rlm-codelens --output $SCAN_FILE"
uv run rlmc scan-repo . --name "rlm-codelens" --output "$SCAN_FILE"

echo -e "${GREEN}✓ Scan complete${NC}"
echo ""

################################################################################
# Phase 2: Analyze Architecture
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 2: Analyzing Architecture${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

ARCH_FILE="$OUTPUTS_DIR/self_arch.json"

echo -e "${BLUE}Running:${NC} uv run rlmc analyze-architecture $SCAN_FILE --output $ARCH_FILE"
uv run rlmc analyze-architecture "$SCAN_FILE" --output "$ARCH_FILE"

echo -e "${GREEN}✓ Architecture analysis complete${NC}"
echo ""

################################################################################
# Phase 3: Generate Visualization
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 3: Generating Visualization${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

VIZ_FILE="$OUTPUTS_DIR/self_viz.html"

echo -e "${BLUE}Running:${NC} uv run rlmc visualize-arch $ARCH_FILE --output $VIZ_FILE --no-browser"
uv run rlmc visualize-arch "$ARCH_FILE" --output "$VIZ_FILE" --no-browser

echo -e "${GREEN}✓ Visualization generated${NC}"
echo ""

################################################################################
# Phase 4: Generate Analysis Report
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 4: Generating Analysis Report${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

REPORT_FILE="$OUTPUTS_DIR/self_report.html"

echo -e "${BLUE}Running:${NC} uv run rlmc generate-report $ARCH_FILE --output $REPORT_FILE --no-browser"
uv run rlmc generate-report "$ARCH_FILE" --output "$REPORT_FILE" --no-browser

echo -e "${GREEN}✓ Report generated${NC}"
echo ""

################################################################################
# Cleanup: Remove intermediate JSON files
################################################################################
echo -e "${YELLOW}Cleaning up intermediate files...${NC}"
rm -f "$SCAN_FILE" "$ARCH_FILE"
echo -e "${GREEN}✓ Removed intermediate files${NC}"
echo ""

################################################################################
# Summary
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Demo Complete! ✓                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  1. Visualization:  ${BLUE}$VIZ_FILE${NC}"
echo -e "  2. Report:         ${BLUE}$REPORT_FILE${NC}"
echo ""
echo -e "${GREEN}Open in browser:${NC}"
echo -e "  ${BLUE}open $VIZ_FILE${NC}"
echo -e "  ${BLUE}open $REPORT_FILE${NC}"
echo ""
