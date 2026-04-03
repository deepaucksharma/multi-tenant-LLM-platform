# Flows

This page exists to show the small, repeatable flows that make up the platform.
Each diagram is intentionally condensed so a new engineer can understand sequence and ownership quickly.
Read this if you already know the big picture and now want to trace how requests, data, and decisions move.

## Flow Index

| Flow | Why it matters |
| --- | --- |
| Chat request | Main product path through routing, RAG, generation, and logging |
| Data lifecycle | How tenant documents become retrieval and training assets |
| Training lifecycle | How adapters are produced and promoted |
| Quality loop | How the platform observes itself and decides when to retrain or roll back |
| Voice path | How speech input and output wrap the same inference runtime |

## Chat Request Flow

```mermaid
sequenceDiagram
    actor User
    participant Client
    participant API as Inference API
    participant Route as Tenant router
    participant RAG as Hybrid RAG
    participant Model as Backend + model
    participant Audit as Audit log

    User->>Client: Send message
    Client->>API: /chat or /chat/stream
    API->>Route: Verify tenant + resolve model route
    alt use_rag = true
        Route->>RAG: Retrieve tenant context
        RAG-->>API: Context + citations + grounding inputs
    end
    API->>Model: Generate with base or adapter route
    Model-->>API: Tokens or final text
    API->>Audit: Save request metadata
    API-->>Client: Response + citations + stats
    Client->>API: Optional feedback
```

## Data Lifecycle

```mermaid
flowchart LR
    A[Synthetic documents] --> B[Ingest]
    B --> C[PII redaction]
    C --> D[Chunking]
    D --> E[SFT dataset build]
    D --> F[DPO dataset build]
    D --> G[RAG index build]
    E --> H[data tenant sft]
    F --> I[data tenant dpo]
    G --> J[data/chroma]
```

## Training Lifecycle

```mermaid
flowchart LR
    A[Training config] --> B[Runtime detection]
    B --> C[Base model load]
    C --> D[SFT training]
    D --> E[Optional DPO]
    E --> F[Adapter artifacts]
    F --> G[Model registry]
    G --> H[Production or staging route]
```

## Quality Loop

```mermaid
flowchart LR
    A[Inference requests] --> B[Audit DB]
    A --> C[Feedback]
    B --> D[Monitoring metrics]
    C --> D
    B --> E[Evaluation inputs]
    E --> F[Eval reports]
    D --> G[Alerts]
    D --> H[Retrain trigger]
    F --> H
    H --> I[Train new adapter]
    F --> J[Rollback or promote]
    J --> K[Registry update]
```

## Voice Path

```mermaid
flowchart LR
    A[Browser mic or upload] --> B[Voice server]
    B --> C[STT]
    C --> D[Inference API]
    D --> E[LLM response text]
    E --> F[TTS]
    F --> G[Audio response]
```

## Notes That Matter

| Detail | Why it matters |
| --- | --- |
| RAG is optional per request | The chat path can run with or without retrieval |
| Routing happens before generation | Tenant policy and model selection are not post-processing steps |
| Training and evaluation are decoupled | New adapters can be evaluated and promoted independently |
| Voice is a wrapper, not a separate intelligence stack | It reuses the same tenant-aware inference path |

## Next Pages To Read

- [architecture.md](architecture.md)
- [operations.md](operations.md)
- [subsystems.md](subsystems.md)
