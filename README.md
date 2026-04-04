# Trust-Aware XAI RL Environment

## Overview

This project presents a reinforcement learning (RL) environment designed to study how AI systems should **select explanation strategies** when interacting with users. Instead of generating explanations directly, the environment focuses on evaluating *which type of explanation* is most appropriate given the context.

The environment models a real-world scenario where AI systems must explain their decisions to users with varying levels of expertise, while maintaining and improving user trust. This problem is formulated as a **sequential decision-making task**, where actions influence future states (user trust and difficulty), making it a true reinforcement learning setting rather than static classification.

---

## Motivation

Modern AI systems are often criticized as **“black boxes”**, producing outputs without clear reasoning. While Explainable AI (XAI) methods attempt to generate explanations, an equally important question remains:

> *Which type of explanation should be given to which user?*

This project addresses that gap by simulating a decision-making process where an AI agent must choose the most appropriate explanation strategy.

The environment emphasizes:

* Correctness of explanation
* Personalization based on user expertise
* Trust dynamics over time

---

## Problem Statement

Design an environment where an agent learns to:

1. Select an appropriate explanation strategy
2. Adapt explanations based on user type
3. Maximize user trust across interactions

---

## Environment Design

### State Space

Each state represents a real-world AI decision scenario and includes:

* `prediction` – AI system output (e.g., "Loan Rejected")
* `features` – key factors influencing the decision
* `user_type` – user expertise level (`beginner` / `expert`)
* `trust_score` – current trust level (range: 0–1)
* `task` – difficulty level (`easy`, `medium`, `hard`)

---

### Action Space

The agent selects one of the following explanation strategies:

* `simple` – brief, easy-to-understand explanation
* `detailed` – technical or comprehensive explanation
* `counterfactual` – “what could change the outcome” explanation
* `none` – no explanation

---

### Reward Function

The reward is designed to reflect real-world explanation quality:

#### 1. Correctness

* +1 → correct explanation strategy
* -1 → incorrect strategy

#### 2. Personalization

* +1 → matches user type
* 0 → neutral

#### 3. Trust Dynamics

* +0.1 → trust increases for correct explanation
* -0.1 → trust decreases for incorrect explanation

---

## Tasks (Difficulty Levels)

The environment includes three tasks with increasing complexity:

### 🟢 Easy Task — Correctness

* Objective: Choose the correct explanation strategy
* Reward depends only on correctness
* Single-step episode

---

### 🟡 Medium Task — Personalization

* Objective: Choose explanation that is both correct and suitable for the user
* Reward includes correctness + personalization
* Multi-step episode (short horizon)

---

### 🔴 Hard Task — Trust Optimization

* Objective: Maximize user trust over multiple interactions
* Multi-step episode
* Reward includes correctness, personalization, and trust changes

---

## Grading and Evaluation

Each task includes a programmatic grader that outputs a score between **0.0 and 1.0**.

### Evaluation Results (Baseline)

* Easy: **1.0**
* Medium: **1.0**
* Hard: **~0.8**

These scores indicate that:

* The environment is deterministic for simpler tasks
* The hard task introduces meaningful complexity

---

## Baseline Agent

A baseline agent is implemented using a Hugging Face Transformer model (Flan-T5). The agent observes the environment state and selects an explanation strategy.

The system is **model-agnostic**, meaning any LLM (e.g., LLaMA, Mistral) can be plugged in.

---
## Agent Evaluation

We evaluate two types of agents in the environment:

### 1. Rule-Based Agent
- Uses deterministic policies based on user type
- Serves as an upper bound benchmark

### 2. LLM Agent (Flan-T5)
- Selects actions based on natural language understanding
- Demonstrates real-world AI behavior under uncertainty

### Key Results

| Agent | Easy | Medium | Hard |
|------|------|--------|------|
| Rule-Based | 1.0 | 1.0 | ~0.8 |
| LLM | lower | moderate | strong |

We observe that:
- LLM struggles in simple/static settings
- Performs significantly better in multi-step environments
- Improves trust over time through interaction

-----

## Key Contributions

* A novel RL environment for **explanation strategy selection**
* Integration of **user modeling** and **trust dynamics**
* Multi-level task design with increasing complexity
* Deterministic grading for reproducible evaluation
* Alignment with real-world Human-AI interaction scenarios

---

## Why This Matters

This environment goes beyond traditional XAI approaches by focusing not just on *what explanations are*, but on:

> **How explanations should be chosen in human-centric AI systems**

It provides a foundation for training and evaluating AI agents that are:

* More transparent
* More adaptive
* More trustworthy

---

## Future Work

* Integration with real-world datasets
* More nuanced user modeling
* Dynamic trust decay and recovery mechanisms
* Multi-agent interaction scenarios

---

## Conclusion

This project introduces a structured way to evaluate explanation strategies in AI systems. By combining reinforcement learning with explainability and trust modeling, it addresses a critical gap in modern AI deployment. This work highlights that explanation selection should be treated as a **sequential, adaptive decision problem**, not a one-step prediction task.

---
