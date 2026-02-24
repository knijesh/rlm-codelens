#!/bin/bash

################################################################################
# RLM-Codelens Architecture Analysis Pipeline
#
# Single entry point for the complete analysis workflow:
# 1. Detect languages & install required tree-sitter grammars
# 2. Scan repository to extract module structure
# 3. Analyze architecture (dependency graph, anti-patterns, layers)
# 4. Generate interactive visualization
# 5. Generate HTML analysis report
#
# Usage:
#   ./run_analysis.sh <path-or-url> [output-prefix] [--deep] [--ollama] [--model MODEL]
#
# Examples:
#   ./run_analysis.sh /path/to/local/repo
#   ./run_analysis.sh https://github.com/encode/starlette starlette
#   ./run_analysis.sh . self
#   ./run_analysis.sh . self --deep                          # Deep with OpenAI
#   ./run_analysis.sh . self --ollama --model qwen3:8b       # Deep with Ollama
#   ./run_analysis.sh https://github.com/kubernetes/kubernetes kubernetes --ollama --model qwen3:8b
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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
        --ollama) USE_OLLAMA=true; DEEP=true; shift ;;
        --ollama-url) OLLAMA_URL="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done

if [ ${#POSITIONAL[@]} -eq 0 ]; then
    echo -e "${RED}Error: Repository path or URL required${NC}"
    echo ""
    echo "Usage: $0 <path-or-url> [output-prefix] [--deep] [--ollama] [--model MODEL]"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/repo"
    echo "  $0 https://github.com/encode/starlette starlette"
    echo "  $0 . self --ollama --model qwen3:8b"
    exit 1
fi

REPO="${POSITIONAL[0]}"
PREFIX="${POSITIONAL[1]:-analysis}"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         RLM-Codelens Architecture Analysis Pipeline           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Repository:${NC}     $REPO"
echo -e "${GREEN}Output Prefix:${NC}  $PREFIX"
echo -e "${GREEN}Deep Analysis:${NC}  $DEEP"
if [ "$USE_OLLAMA" = true ]; then
    echo -e "${GREEN}Backend:${NC}        Ollama ($OLLAMA_URL)"
    if [ -n "$MODEL" ]; then
        echo -e "${GREEN}Model:${NC}          $MODEL"
    else
        echo -e "${GREEN}Model:${NC}          (interactive selection)"
    fi
fi
echo ""

mkdir -p "$OUTPUTS_DIR"

################################################################################
# Phase 0: Detect Languages & Install Grammars
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 0: Detecting Languages & Installing Grammars${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Detect languages by checking file extensions in the repo
# For URLs, we detect after cloning (scan phase handles this)
# For local paths, detect now to install grammars before scanning
GRAMMAR_EXTRAS=""
REPO_PATH="$REPO"

if [[ "$REPO" =~ ^https?:// ]] || [[ "$REPO" =~ ^git@ ]]; then
    echo -e "${CYAN}Remote repository — will detect languages after cloning.${NC}"
    # For remote repos, install common grammars proactively based on URL hints
    REPO_LOWER=$(echo "$REPO" | tr '[:upper:]' '[:lower:]')
    # Always install tree-sitter base
    GRAMMAR_EXTRAS="tree-sitter"

    # Try to guess from repo URL or name
    # Default: install all common grammars for remote repos
    echo -e "${CYAN}Installing common language grammars...${NC}"
    uv pip install tree-sitter tree-sitter-go tree-sitter-java tree-sitter-javascript tree-sitter-typescript tree-sitter-rust tree-sitter-c tree-sitter-cpp 2>/dev/null || true
    echo -e "${GREEN}✓ Language grammars installed${NC}"
else
    # Local repo — detect languages from file extensions
    echo -e "${CYAN}Scanning for languages in local repository...${NC}"
    DETECTED_LANGS=""

    if find "$REPO" -name "*.go" -not -path "*/vendor/*" -not -path "*/.git/*" | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS go"
    fi
    if find "$REPO" -name "*.java" -not -path "*/target/*" -not -path "*/.git/*" | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS java"
    fi
    if find "$REPO" -name "*.ts" -o -name "*.tsx" -not -path "*/node_modules/*" -not -path "*/.git/*" 2>/dev/null | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS typescript"
    fi
    if find "$REPO" -name "*.js" -o -name "*.jsx" -not -path "*/node_modules/*" -not -path "*/.git/*" 2>/dev/null | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS javascript"
    fi
    if find "$REPO" -name "*.rs" -not -path "*/target/*" -not -path "*/.git/*" | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS rust"
    fi
    if find "$REPO" -name "*.cpp" -o -name "*.cc" -o -name "*.c" -o -name "*.h" -not -path "*/.git/*" 2>/dev/null | head -1 | grep -q .; then
        DETECTED_LANGS="$DETECTED_LANGS cpp"
    fi

    if [ -n "$DETECTED_LANGS" ]; then
        echo -e "${GREEN}Detected languages:${NC}$DETECTED_LANGS"

        # Install required grammars
        PKGS="tree-sitter"
        for lang in $DETECTED_LANGS; do
            case $lang in
                go)         PKGS="$PKGS tree-sitter-go" ;;
                java)       PKGS="$PKGS tree-sitter-java" ;;
                typescript) PKGS="$PKGS tree-sitter-typescript" ;;
                javascript) PKGS="$PKGS tree-sitter-javascript" ;;
                rust)       PKGS="$PKGS tree-sitter-rust" ;;
                cpp)        PKGS="$PKGS tree-sitter-c tree-sitter-cpp" ;;
            esac
        done

        echo -e "${CYAN}Installing grammars: ${PKGS}${NC}"
        uv pip install $PKGS 2>/dev/null || true
        echo -e "${GREEN}✓ Language grammars installed${NC}"
    else
        echo -e "${CYAN}Only Python detected — no additional grammars needed.${NC}"
    fi
fi
echo ""

################################################################################
# Phase 1: Scan Repository
################################################################################
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Phase 1: Scanning Repository${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

SCAN_FILE="$OUTPUTS_DIR/${PREFIX}_scan.json"

SCAN_ARGS=("$REPO" --name "$PREFIX" --output "$SCAN_FILE")
if [ "$DEEP" = true ]; then
    SCAN_ARGS+=(--include-source)
fi

echo -e "${BLUE}Running:${NC} uv run rlmc scan-repo ${SCAN_ARGS[*]}"
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

if [ "$USE_OLLAMA" = true ]; then
    # --ollama implies --deep and sets backend
    ARCH_ARGS+=(--ollama)

    if [ -z "$MODEL" ]; then
        # Interactive model selection
        echo -e "${YELLOW}Querying Ollama for available models...${NC}"
        uv run rlmc list-models --ollama-url "$OLLAMA_URL" --no-select
        echo ""
        read -rp "  Enter model name to use: " MODEL
        if [ -z "$MODEL" ]; then
            echo -e "${RED}Error: No model selected. Aborting.${NC}"
            exit 1
        fi
    fi

    ARCH_ARGS+=(--model "$MODEL")
    echo -e "${GREEN}Using Ollama model:${NC} $MODEL"
elif [ "$DEEP" = true ]; then
    ARCH_ARGS+=(--deep)
    if [ -n "$MODEL" ]; then
        ARCH_ARGS+=(--model "$MODEL")
        echo -e "${GREEN}Using model:${NC} $MODEL"
    fi
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

echo -e "${BLUE}Running:${NC} uv run rlmc visualize-arch $ARCH_FILE --output $VIZ_FILE --no-browser"
uv run rlmc visualize-arch "$ARCH_FILE" --output "$VIZ_FILE" --no-browser

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
# Summary
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Analysis Complete!                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Output Files:${NC}"
echo -e "  Scan data:      ${CYAN}$SCAN_FILE${NC}"
echo -e "  Architecture:   ${CYAN}$ARCH_FILE${NC}"
echo -e "  Visualization:  ${CYAN}samples/${PREFIX}_viz.html${NC}"
echo -e "  Report:         ${CYAN}samples/${PREFIX}_report.html${NC}"
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo -e "  Open report:         ${BLUE}open samples/${PREFIX}_report.html${NC}"
echo -e "  Open visualization:  ${BLUE}open samples/${PREFIX}_viz.html${NC}"
echo ""
