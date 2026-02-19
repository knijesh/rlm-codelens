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
#   ./run_analysis.sh <path-or-url> [output-prefix] [--deep] [--ollama] [--model MODEL]
#
# Examples:
#   ./run_analysis.sh /path/to/local/repo
#   ./run_analysis.sh https://github.com/encode/starlette starlette
#   ./run_analysis.sh . self
#   ./run_analysis.sh . self --deep                    # Deep analysis with OpenAI
#   ./run_analysis.sh . self --deep --ollama           # Deep analysis with Ollama (interactive model select)
#   ./run_analysis.sh . self --deep --ollama --model llama3.1   # Deep with specific Ollama model
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
DEEP=false
USE_OLLAMA=false
OLLAMA_URL="http://localhost:11434"
MODEL=""
POSITIONAL=()

while [ $# -gt 0 ]; do
    case $1 in
        --deep) DEEP=true; shift ;;
        --ollama) USE_OLLAMA=true; shift ;;
        --ollama-url) OLLAMA_URL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [ ${#POSITIONAL[@]} -eq 0 ]; then
    echo -e "${RED}Error: Repository path or URL required${NC}"
    echo "Usage: $0 <path-or-url> [output-prefix] [--deep] [--ollama] [--model MODEL]"
    echo "Example: $0 /path/to/repo myrepo --deep --ollama"
    exit 1
fi

# If --ollama is used, --deep is implied
if [ "$USE_OLLAMA" = true ]; then
    DEEP=true
fi

REPO="${POSITIONAL[0]}"
PREFIX="${POSITIONAL[1]:-analysis}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RLM-Codelens Architecture Analysis Pipeline           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Repository:${NC} $REPO"
echo -e "${GREEN}Output Prefix:${NC} $PREFIX"
echo -e "${GREEN}Output Directory:${NC} $OUTPUTS_DIR"
echo -e "${GREEN}Deep Analysis:${NC} $DEEP"
if [ "$USE_OLLAMA" = true ]; then
    echo -e "${GREEN}Backend:${NC} Ollama ($OLLAMA_URL)"
    if [ -n "$MODEL" ]; then
        echo -e "${GREEN}Model:${NC} $MODEL"
    else
        echo -e "${GREEN}Model:${NC} (interactive selection)"
    fi
fi
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

SCAN_CMD="uv run rlmc scan-repo $REPO --name $PREFIX --output $SCAN_FILE"
SCAN_ARGS=("$REPO" --name "$PREFIX" --output "$SCAN_FILE")
if [ "$DEEP" = true ]; then
    SCAN_CMD="$SCAN_CMD --include-source"
    SCAN_ARGS+=(--include-source)
fi

echo -e "${BLUE}Running:${NC} $SCAN_CMD"
uv run rlmc scan-repo "${SCAN_ARGS[@]}"

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

ARCH_ARGS=("$SCAN_FILE" --output "$ARCH_FILE")
if [ "$DEEP" = true ]; then
    ARCH_ARGS+=(--deep)
fi

if [ "$USE_OLLAMA" = true ]; then
    ARCH_ARGS+=(--backend openai --base-url "${OLLAMA_URL}/v1")

    if [ -z "$MODEL" ]; then
        # Interactive model selection
        echo -e "${YELLOW}Querying Ollama for available models...${NC}"
        uv run rlmc list-models --ollama-url "$OLLAMA_URL" --no-select
        echo ""
        read -rp "  Enter model name (or number) to use: " MODEL
        if [ -z "$MODEL" ]; then
            echo -e "${RED}Error: No model selected. Aborting.${NC}"
            exit 1
        fi
    fi

    ARCH_ARGS+=(--model "$MODEL")
    echo -e "${GREEN}Using Ollama model:${NC} $MODEL"
fi

echo -e "${BLUE}Running:${NC} uv run rlmc analyze-architecture ${ARCH_ARGS[*]}"
uv run rlmc analyze-architecture "${ARCH_ARGS[@]}"

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
# Phase 4: Generate Analysis Report
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 4: Generating Analysis Report${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

REPORT_FILE="$OUTPUTS_DIR/${PREFIX}_report.html"

echo -e "${BLUE}Running:${NC} uv run rlmc generate-report $ARCH_FILE --output $REPORT_FILE --no-browser"
uv run rlmc generate-report "$ARCH_FILE" --output "$REPORT_FILE" --no-browser

if [ ! -f "$REPORT_FILE" ]; then
    echo -e "${RED}Error: Report generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Report generated: $REPORT_FILE${NC}"
echo ""

################################################################################
# Copy outputs to samples/ for GitHub visibility
################################################################################
echo -e "${YELLOW}Copying outputs to samples/ ...${NC}"
mkdir -p samples
cp "$VIZ_FILE" "samples/${PREFIX}_viz.html"
cp "$REPORT_FILE" "samples/${PREFIX}_report.html"
echo -e "${GREEN}✓ Copied to samples/${NC}"
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
echo -e "${BLUE}║                    Analysis Complete! ✓                       ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  1. Visualization:  ${BLUE}$VIZ_FILE${NC}"
echo -e "  2. Report:         ${BLUE}$REPORT_FILE${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo -e "  • Open visualization: ${BLUE}open $VIZ_FILE${NC}"
echo -e "  • Open report:        ${BLUE}open $REPORT_FILE${NC}"
echo ""
