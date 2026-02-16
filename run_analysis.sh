#!/bin/bash

################################################################################
# RLM-Codelens Architecture Analysis Pipeline
#
# This script runs the complete architecture analysis workflow:
# 1. Scan repository to extract module structure
# 2. Analyze architecture (dependency graph, anti-patterns, layers)
# 3. Generate interactive visualization
#
# Usage:
#   ./run_analysis.sh <path-or-url> [output-prefix]
#
# Examples:
#   ./run_analysis.sh /path/to/local/repo
#   ./run_analysis.sh https://github.com/encode/starlette starlette
#   ./run_analysis.sh . self
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
OUTPUTS_DIR="outputs"

# Parse arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Repository path or URL required${NC}"
    echo "Usage: $0 <path-or-url> [output-prefix]"
    echo "Example: $0 /path/to/repo myrepo"
    exit 1
fi

REPO=$1
PREFIX=${2:-"analysis"}

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RLM-Codelens Architecture Analysis Pipeline           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Repository:${NC} $REPO"
echo -e "${GREEN}Output Prefix:${NC} $PREFIX"
echo -e "${GREEN}Output Directory:${NC} $OUTPUTS_DIR"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p $OUTPUTS_DIR

################################################################################
# Phase 1: Scan Repository
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 1: Scanning Repository${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

SCAN_FILE="$OUTPUTS_DIR/${PREFIX}_scan.json"

echo -e "${BLUE}Running:${NC} uv run rlmc scan-repo $REPO --output $SCAN_FILE"
uv run rlmc scan-repo "$REPO" --output "$SCAN_FILE"

if [ ! -f "$SCAN_FILE" ]; then
    echo -e "${RED}Error: Repository scan failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Scan saved to: $SCAN_FILE${NC}"
echo ""

################################################################################
# Phase 2: Analyze Architecture
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 2: Analyzing Architecture${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

ARCH_FILE="$OUTPUTS_DIR/${PREFIX}_arch.json"

echo -e "${BLUE}Running:${NC} uv run rlmc analyze-architecture $SCAN_FILE --output $ARCH_FILE"
uv run rlmc analyze-architecture "$SCAN_FILE" --output "$ARCH_FILE"

if [ ! -f "$ARCH_FILE" ]; then
    echo -e "${RED}Error: Architecture analysis failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Architecture analysis saved to: $ARCH_FILE${NC}"
echo ""

################################################################################
# Phase 3: Generate Visualization
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 3: Generating Interactive Visualization${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

VIZ_FILE="$OUTPUTS_DIR/${PREFIX}_viz.html"

echo -e "${BLUE}Running:${NC} uv run rlmc visualize-arch $ARCH_FILE --output $VIZ_FILE"
uv run rlmc visualize-arch "$ARCH_FILE" --output "$VIZ_FILE"

if [ ! -f "$VIZ_FILE" ]; then
    echo -e "${RED}Error: Visualization generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Visualization generated: $VIZ_FILE${NC}"
echo ""

################################################################################
# Summary
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Analysis Complete! ✓                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  1. Scan:           ${BLUE}$SCAN_FILE${NC}"
echo -e "  2. Architecture:   ${BLUE}$ARCH_FILE${NC}"
echo -e "  3. Visualization:  ${BLUE}$VIZ_FILE${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo -e "  • Open visualization: ${BLUE}open $VIZ_FILE${NC}"
echo -e "  • Deep RLM analysis:  ${BLUE}uv run rlmc analyze-architecture $SCAN_FILE --deep${NC}"
echo ""
