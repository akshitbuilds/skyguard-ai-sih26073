# SkyGuard AI — SIH26073

> **Agentic Genesis · Smart India Hackathon 2026**
>
> ### Self-Aware, Self-Healing Anomaly Detection for Automatic Weather Stations

**SkyGuard AI** is an AI/ML-based intelligent anomaly detection and safe data-quality system for Automatic Weather Station (AWS) networks.

It monitors the three required AWS variables:

- 🌡️ Temperature
- 🧭 Atmospheric Pressure
- 💧 Relative Humidity

The core problem is simple:

> **An anomaly does not necessarily mean the sensor is broken.**

SkyGuard determines whether an unusual observation is caused by a **sensor fault** or represents a **genuine regional weather event**, explains the decision, estimates sensor health, and performs correction only when it is safe.

---

## 🎯 SIH26073 Problem

**Problem Statement:** SIH26073  
**Title:** AI/ML-Based Intelligent Anomaly Detection for Automatic Weather Stations (AWS)  
**Domain:** Disaster Management  
**Organization:** Ministry of Earth Sciences / India Meteorological Department

The system must identify abnormal AWS observations such as:

- Sensor spikes
- Frozen/stuck values
- Communication/dropout failures
- Calibration drift
- Temporally inconsistent observations
- Multivariate inconsistencies

while avoiding false alarms during genuine extreme weather events.

SkyGuard addresses this using a multi-agent hybrid architecture combining:

**Physical rules + temporal analysis + spatial corroboration + LSTM anomaly detection + explainable decision gating + safe correction.**

---

# 🚀 The Core Idea

## Anomaly ≠ Sensor Fault

A conventional anomaly detector may see an extreme temperature or pressure change and immediately label it as faulty.

That is dangerous.

A real heatwave, storm, or regional weather transition can legitimately produce extreme observations across multiple stations.

SkyGuard therefore asks:

> **"Is the sensor wrong, or is the weather real?"**

The system separates:

1. **Isolated sensor faults**
2. **Sensor degradation**
3. **Communication failures**
4. **Genuine regional weather events**

A genuine-event safety shield prevents automatic correction when multiple stations and temporal evidence support a real meteorological event.

---

# 🧠 Multi-Agent Architecture

```text
                    AWS OBSERVATION
              Temperature / Pressure / RH
                         │
                         ▼
              ┌─────────────────────┐
              │     AGENT 1         │
              │ Ingestion & Replay  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     AGENT 2         │
              │ Physical + Spatial  │
              │ + Temporal Screening│
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     AGENT 3         │
              │   LSTM Autoencoder  │
              │ Temporal Anomaly    │
              │      Detection      │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │   EVIDENCE FUSION   │
              │ Rules + ML + Spatial│
              │ + Temporal Evidence │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     AGENT 4         │
              │ RF Decision Gate    │
              │ SHAP Explainability │
              │ Safety Override     │
              │ Degradation Tracker │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     AGENT 5         │
              │ Safe Self-Healing   │
              │ Temporal + Spatial  │
              │     Correction      │
              └──────────┬──────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        ┌───────────┐         ┌──────────────┐
        │ FastAPI   │         │  Streamlit   │
        │ REST API  │         │ Command      │
        │           │         │ Center       │
        └───────────┘         └──────────────┘
