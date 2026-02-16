# RLM-Codelens Architecture Diagrams

## 1. High-Level System Flow (Mermaid)

```mermaid
flowchart TB
    Start([User Input: Repository Path]) --> Scanner
    
    subgraph Phase1["Phase 1: Repository Scanning"]
        Scanner[repo_scanner.py<br/>AST Parser] --> ScanData[(scan.json<br/>Module Structure)]
        Scanner --> |Extracts| Imports[Imports & Dependencies]
        Scanner --> |Extracts| Classes[Classes & Functions]
        Scanner --> |Extracts| Metadata[LOC, Docstrings, Tests]
    end
    
    ScanData --> GraphBuilder
    
    subgraph Phase2["Phase 2: Static Analysis"]
        GraphBuilder[codebase_graph.py<br/>NetworkX Graph] --> Graph[(Dependency Graph)]
        Graph --> Cycles[Cycle Detection]
        Graph --> Hubs[Hub Module Detection]
        Graph --> Coupling[Coupling Metrics]
        Graph --> Layers[Layer Classification]
        Graph --> AntiPatterns[Anti-Pattern Detection]
    end
    
    Graph --> Decision{Deep Analysis?}
    
    Decision -->|No| StaticResults[Static Analysis Results]
    Decision -->|Yes| RLMAnalysis
    
    subgraph Phase3["Phase 3: RLM Deep Analysis (Optional)"]
        RLMAnalysis[architecture_analyzer.py<br/>RLM API] --> Classify[Module Classification]
        RLMAnalysis --> Hidden[Hidden Dependencies]
        RLMAnalysis --> Patterns[Pattern Detection]
        RLMAnalysis --> Refactor[Refactoring Suggestions]
        
        Classify --> RLMResults[RLM Results]
        Hidden --> RLMResults
        Patterns --> RLMResults
        Refactor --> RLMResults
    end
    
    StaticResults --> Merge[Merge Results]
    RLMResults --> Merge
    
    Merge --> ArchData[(architecture.json<br/>Complete Analysis)]
    
    subgraph Phase4["Phase 4: Visualization & Reporting"]
        ArchData --> Visualizer[visualizer.py<br/>D3.js Generator]
        ArchData --> Reporter[report_generator.py<br/>HTML Generator]
        
        Visualizer --> VizHTML[Interactive Graph<br/>architecture_viz.html]
        Reporter --> ReportHTML[Analysis Report<br/>report.html]
    end
    
    VizHTML --> Browser([Browser Display])
    ReportHTML --> Browser
    
    style Phase1 fill:#e1f5ff
    style Phase2 fill:#fff4e1
    style Phase3 fill:#ffe1f5
    style Phase4 fill:#e1ffe1
    style Scanner fill:#4a90e2
    style GraphBuilder fill:#f5a623
    style RLMAnalysis fill:#bd10e0
    style Visualizer fill:#7ed321
```

## 2. Detailed Component Architecture (Mermaid)

```mermaid
graph TB
    subgraph CLI["CLI Layer (cli.py)"]
        CLI_Entry[rlmc Command] --> CMD_Scan[scan-repo]
        CLI_Entry --> CMD_Analyze[analyze-architecture]
        CLI_Entry --> CMD_Viz[visualize-arch]
        CLI_Entry --> CMD_Report[generate-report]
    end
    
    subgraph Commands["Command Layer (commands.py)"]
        CMD_Scan --> Cmd_ScanRepo[scan_repository]
        CMD_Analyze --> Cmd_AnalyzeArch[analyze_architecture]
        CMD_Viz --> Cmd_Visualize[visualize_architecture]
        CMD_Report --> Cmd_GenReport[generate_report]
    end
    
    subgraph Core["Core Analysis Engine"]
        Cmd_ScanRepo --> RepoScanner[RepositoryScanner]
        
        subgraph Scanner["repo_scanner.py"]
            RepoScanner --> AST[AST Parser]
            RepoScanner --> GitClone[Git Clone Handler]
            RepoScanner --> ModuleInfo[ModuleInfo Extractor]
            AST --> Imports[Import Analysis]
            AST --> ClassFunc[Class/Function Analysis]
            ModuleInfo --> RepoStruct[RepositoryStructure]
        end
        
        Cmd_AnalyzeArch --> GraphAnalyzer[CodebaseGraphAnalyzer]
        
        subgraph GraphAnalysis["codebase_graph.py"]
            GraphAnalyzer --> NetworkX[NetworkX DiGraph]
            NetworkX --> CycleDetect[Cycle Detection]
            NetworkX --> HubDetect[Hub Detection]
            NetworkX --> CouplingCalc[Coupling Metrics]
            NetworkX --> LayerDetect[Layer Detection]
            NetworkX --> AntiPatternDetect[Anti-Pattern Detection]
            
            CycleDetect --> ArchAnalysis[ArchitectureAnalysis]
            HubDetect --> ArchAnalysis
            CouplingCalc --> ArchAnalysis
            LayerDetect --> ArchAnalysis
            AntiPatternDetect --> ArchAnalysis
        end
        
        Cmd_AnalyzeArch --> |--deep flag| RLMAnalyzer[ArchitectureRLMAnalyzer]
        
        subgraph RLM["architecture_analyzer.py"]
            RLMAnalyzer --> RLMClient[RLM API Client]
            RLMClient --> OpenAI[OpenAI Backend]
            RLMClient --> Anthropic[Anthropic Backend]
            
            RLMAnalyzer --> ClassifyModules[classify_modules]
            RLMAnalyzer --> DiscoverHidden[discover_hidden_deps]
            RLMAnalyzer --> DetectPatterns[detect_patterns]
            RLMAnalyzer --> SuggestRefactor[suggest_refactoring]
            
            ClassifyModules --> CostTracker[RLMCostTracker]
            DiscoverHidden --> CostTracker
            DetectPatterns --> CostTracker
            SuggestRefactor --> CostTracker
        end
        
        Cmd_Visualize --> Visualizer[ArchitectureVisualizer]
        
        subgraph Viz["visualizer.py"]
            Visualizer --> EnrichData[Data Enrichment]
            Visualizer --> D3Template[D3.js Template]
            EnrichData --> DepTree[Dependency Trees]
            EnrichData --> CycleMembership[Cycle Membership]
            D3Template --> HTMLGen[HTML Generation]
        end
        
        Cmd_GenReport --> ReportGen[ReportGenerator]
        
        subgraph Report["report_generator.py"]
            ReportGen --> ReportTemplate[HTML Template]
            ReportGen --> Metrics[Metrics Formatting]
            ReportGen --> Charts[Chart Generation]
        end
    end
    
    subgraph Utils["Utilities"]
        CostTracker --> SecureLog[secure_logging.py]
        Config[config.py] --> |Environment| RLMAnalyzer
        Config --> |API Keys| RLMClient
    end
    
    subgraph Data["Data Models"]
        RepoStruct --> JSON1[(scan.json)]
        ArchAnalysis --> JSON2[(architecture.json)]
        HTMLGen --> HTML1[(viz.html)]
        ReportGen --> HTML2[(report.html)]
    end
    
    style CLI fill:#667eea
    style Commands fill:#764ba2
    style Core fill:#f093fb
    style Scanner fill:#4facfe
    style GraphAnalysis fill:#43e97b
    style RLM fill:#fa709a
    style Viz fill:#fee140
    style Report fill:#30cfd0
    style Utils fill:#a8edea
    style Data fill:#fed6e3
```

## 3. Data Flow Architecture (Mermaid)

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Scanner as RepositoryScanner
    participant Graph as CodebaseGraphAnalyzer
    participant RLM as ArchitectureRLMAnalyzer
    participant Viz as Visualizer
    
    User->>CLI: rlmc scan-repo /path/to/repo
    CLI->>Scanner: scan()
    
    Scanner->>Scanner: Clone repo (if URL)
    Scanner->>Scanner: Find Python files
    Scanner->>Scanner: Parse with AST
    Scanner->>Scanner: Extract imports, classes, functions
    Scanner-->>CLI: RepositoryStructure
    CLI->>CLI: Save scan.json
    
    User->>CLI: rlmc analyze-architecture scan.json --deep
    CLI->>Graph: CodebaseGraphAnalyzer(structure)
    
    Graph->>Graph: Build NetworkX DiGraph
    Graph->>Graph: Detect cycles
    Graph->>Graph: Find hub modules
    Graph->>Graph: Calculate coupling
    Graph->>Graph: Classify layers
    Graph->>Graph: Detect anti-patterns
    Graph-->>CLI: ArchitectureAnalysis (static)
    
    alt Deep Analysis Enabled
        CLI->>RLM: ArchitectureRLMAnalyzer(structure)
        RLM->>RLM: classify_modules()
        RLM->>RLM: discover_hidden_deps()
        RLM->>RLM: detect_patterns()
        RLM->>RLM: suggest_refactoring()
        RLM-->>CLI: RLM Results + Cost Summary
        CLI->>Graph: enrich_with_rlm(rlm_results)
        Graph-->>CLI: Enhanced ArchitectureAnalysis
    end
    
    CLI->>CLI: Save architecture.json
    
    User->>CLI: rlmc visualize-arch architecture.json
    CLI->>Viz: generate_architecture_visualization()
    Viz->>Viz: Load architecture.json
    Viz->>Viz: Enrich with dependency trees
    Viz->>Viz: Inject into D3.js template
    Viz-->>CLI: architecture_viz.html
    CLI->>User: Open in browser
    
    User->>CLI: rlmc generate-report architecture.json
    CLI->>Viz: generate_analysis_report()
    Viz-->>CLI: report.html
    CLI->>User: Open in browser
```

## 4. Module Dependency Graph (Mermaid)

```mermaid
graph LR
    subgraph Entry["Entry Points"]
        CLI[cli.py]
        Main[__main__.py]
    end
    
    subgraph Commands["Command Layer"]
        Cmd[commands.py]
    end
    
    subgraph Core["Core Modules"]
        Scanner[repo_scanner.py]
        Graph[codebase_graph.py]
        RLM[architecture_analyzer.py]
        Viz[visualizer.py]
        Report[report_generator.py]
    end
    
    subgraph Config["Configuration"]
        Cfg[config.py]
    end
    
    subgraph Utils["Utilities"]
        Cost[utils/cost_tracker.py]
        Log[utils/secure_logging.py]
    end
    
    subgraph External["External Dependencies"]
        AST[ast - stdlib]
        NX[networkx]
        RLMLib[rlm library]
        JSON[json - stdlib]
    end
    
    CLI --> Cmd
    Main --> CLI
    
    Cmd --> Scanner
    Cmd --> Graph
    Cmd --> RLM
    Cmd --> Viz
    Cmd --> Report
    Cmd --> Cfg
    
    Scanner --> AST
    Scanner --> JSON
    
    Graph --> Scanner
    Graph --> NX
    
    RLM --> Scanner
    RLM --> RLMLib
    RLM --> Cost
    RLM --> Cfg
    
    Viz --> JSON
    
    Report --> JSON
    
    Cost --> Log
    
    style Entry fill:#667eea
    style Commands fill:#764ba2
    style Core fill:#f093fb
    style Config fill:#4facfe
    style Utils fill:#43e97b
    style External fill:#fa709a
```

## 5. Layer Architecture (Conceptual)

```mermaid
graph TB
    subgraph Presentation["Presentation Layer"]
        CLI_UI[CLI Interface<br/>cli.py]
        HTML_UI[HTML Visualizations<br/>D3.js + Reports]
    end
    
    subgraph Application["Application Layer"]
        Commands[Command Handlers<br/>commands.py]
        Orchestration[Workflow Orchestration<br/>Phase Management]
    end
    
    subgraph Domain["Domain Layer"]
        Scanner_Domain[Repository Scanning<br/>AST Analysis]
        Graph_Domain[Graph Analysis<br/>Architecture Metrics]
        RLM_Domain[Semantic Analysis<br/>RLM Integration]
    end
    
    subgraph Infrastructure["Infrastructure Layer"]
        FileIO[File I/O<br/>JSON Serialization]
        GitOps[Git Operations<br/>Clone & Cleanup]
        APIClient[API Clients<br/>OpenAI/Anthropic]
        GraphLib[Graph Library<br/>NetworkX]
    end
    
    subgraph CrossCutting["Cross-Cutting Concerns"]
        Config[Configuration<br/>Environment Variables]
        Logging[Secure Logging<br/>Cost Tracking]
        ErrorHandling[Error Handling<br/>Budget Management]
    end
    
    CLI_UI --> Commands
    HTML_UI --> Commands
    
    Commands --> Scanner_Domain
    Commands --> Graph_Domain
    Commands --> RLM_Domain
    
    Scanner_Domain --> FileIO
    Scanner_Domain --> GitOps
    
    Graph_Domain --> GraphLib
    Graph_Domain --> FileIO
    
    RLM_Domain --> APIClient
    RLM_Domain --> FileIO
    
    Config -.-> Commands
    Config -.-> RLM_Domain
    
    Logging -.-> RLM_Domain
    Logging -.-> Commands
    
    ErrorHandling -.-> RLM_Domain
    ErrorHandling -.-> Commands
    
    style Presentation fill:#667eea,color:#fff
    style Application fill:#764ba2,color:#fff
    style Domain fill:#f093fb,color:#000
    style Infrastructure fill:#4facfe,color:#fff
    style CrossCutting fill:#43e97b,color:#000
```

## 6. Technology Stack

```mermaid
mindmap
  root((RLM-Codelens))
    Core Technologies
      Python 3.11+
      AST Parser
      NetworkX
      RLM Library
    Analysis
      Static Analysis
        AST Parsing
        Import Resolution
        Cycle Detection
        Coupling Metrics
      Dynamic Analysis
        RLM API
        OpenAI GPT-4o
        Anthropic Claude
        Semantic Classification
    Visualization
      D3.js Force Graph
      HTML/CSS/JavaScript
      Interactive UI
      Dependency Trees
    CLI
      argparse
      Command Pattern
      Phase Management
      Error Handling
    Data Formats
      JSON
      HTML
      Markdown
```

## Key Components Description

### 1. **RepositoryScanner** (`repo_scanner.py`)
- **Purpose**: Parse Python repositories using AST
- **Input**: Local path or Git URL
- **Output**: `RepositoryStructure` with modules, imports, classes, functions
- **Key Features**:
  - Git clone support for remote repos
  - Configurable exclusion patterns
  - Optional source code inclusion
  - Relative import resolution

### 2. **CodebaseGraphAnalyzer** (`codebase_graph.py`)
- **Purpose**: Build and analyze module dependency graph
- **Input**: `RepositoryStructure`
- **Output**: `ArchitectureAnalysis` with metrics
- **Key Features**:
  - NetworkX directed graph construction
  - Cycle detection (circular imports)
  - Hub module identification
  - Coupling metrics (afferent/efferent)
  - Layer classification heuristics
  - Anti-pattern detection

### 3. **ArchitectureRLMAnalyzer** (`architecture_analyzer.py`)
- **Purpose**: Deep semantic analysis using RLM
- **Input**: `RepositoryStructure` + optional graph metrics
- **Output**: Semantic clusters, hidden deps, patterns, suggestions
- **Key Features**:
  - Module classification into architectural layers
  - Dynamic import discovery
  - Pattern detection (MVC, layered, etc.)
  - Refactoring suggestions
  - Cost tracking and budget enforcement

### 4. **Visualizer** (`visualizer.py`)
- **Purpose**: Generate interactive D3.js visualizations
- **Input**: `ArchitectureAnalysis` JSON
- **Output**: Standalone HTML with embedded data
- **Key Features**:
  - Force-directed graph layout
  - Interactive node exploration
  - Dependency tree tracer
  - Layer-based coloring
  - Cycle highlighting

### 5. **CLI** (`cli.py` + `commands.py`)
- **Purpose**: Command-line interface
- **Commands**:
  - `scan-repo`: Parse repository
  - `analyze-architecture`: Run analysis
  - `visualize-arch`: Generate visualization
  - `generate-report`: Create HTML report
- **Features**:
  - Pipeline support
  - Progress reporting
  - Error handling
  - Browser auto-open

## Analysis Pipeline

```
Repository → AST Scanner → Dependency Graph → Architecture Analysis → Interactive Visualization
   (input)     (parse every    (NetworkX graph    (anti-patterns, layers,    (D3.js force-directed
                Python file)    of imports)        hub modules, cycles)       graph in HTML)
```

### Static Analysis (No API calls)
1. Parse all Python files with AST
2. Extract imports, classes, functions
3. Build NetworkX directed graph
4. Detect cycles, hubs, coupling
5. Classify layers heuristically
6. Identify anti-patterns

### Deep RLM Analysis (Optional, requires API key)
1. Classify modules semantically
2. Discover hidden/dynamic dependencies
3. Detect architectural patterns
4. Generate refactoring suggestions
5. Track costs and enforce budget

## Anti-Patterns Detected

1. **God Modules**: Large files (>500 LOC) with high fan-out (>10)
2. **Orphan Modules**: No imports or dependents
3. **Layer Violations**: Lower layers importing higher layers
4. **Circular Imports**: Detected via NetworkX cycle detection
5. **Tight Coupling**: High instability metrics

## Output Artifacts

1. **scan.json**: Repository structure with all modules
2. **architecture.json**: Complete analysis results
3. **architecture_viz.html**: Interactive D3.js graph
4. **report.html**: Detailed analysis report

---

**Generated for**: RLM-Codelens v0.2.1  
**Author**: Nijesh Kanjinghat  
**License**: MIT