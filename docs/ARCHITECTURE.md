# SkyGuard AI Architecture

```text
                         AWS TELEMETRY
                    T / P / RH only
                            |
                            v
                  +--------------------+
                  | Agent 1             |
                  | Ingestion / Replay  |
                  +----------+----------+
                             |
                             v
                  +--------------------+
                  | Agent 2             |
                  | Physical consistency|
                  | Spatial deviation   |
                  | Temporal screening  |
                  +----------+----------+
                             |
                             v
                  +--------------------+
                  | Agent 3             |
                  | LSTM Autoencoder    |
                  | 12-hour sequence    |
                  +----------+----------+
                             |
                             v
             +-------------------------------+
             | Hybrid Evidence Fusion        |
             | dropout / frozen / spike      |
             | drift / regional event shield |
             +---------------+---------------+
                             |
                             v
                  +--------------------+
                  | Agent 4             |
                  | RF safety gate      |
                  | SHAP explanation    |
                  | degradation tracker |
                  +----------+----------+
                             |
                             v
                  +--------------------+
                  | Agent 5             |
                  | safe correction     |
                  | severity / alert     |
                  +----------+----------+
                             |
                  +----------+----------+
                  |                     |
                  v                     v
             FastAPI API          Command Center
                                 Network + evidence
```

## Safety principle

The learned detector is advisory evidence, not the sole authority. Explicit fault signatures and the regional-event shield are deterministic and inspectable. This is important for a disaster-management quality-control system where incorrectly suppressing a genuine event can be worse than raising an additional review alert.
