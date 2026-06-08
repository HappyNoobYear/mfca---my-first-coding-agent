# mfca — My First Coding Agent

An autonomous, multi-turn AI coding assistant built for the Advanced Media Technologies (AMT) Master's module. The agent iteratively reasons, interacts with directory systems, modifies source code, and executes files inside a secure, network-isolated Docker sandbox container.
## Table of Contents
- [Features](#features)
- [Architecture & Isolation](#architecture--isolation)
- [Technologies](#technologies)
- [Installation & Prerequisites](#installation--prerequisites)
- [Execution / Usage](#execution--usage)
- [Change Log](#change-log)
- [Contributors & License](#contributors--license)

## Features

**Multi-Turn Reconciling Loop:**

Retains conversational history context allowing users to issue follow-up execution or debugging requests natively.

**Autonomous Tool-Calling Ecosystem:**

    ReadDirectoryTool: Safely scans workspaces to locate target source paths.

    ReadCodeTool: Extracts file contents for structural context mapping.

    WriteCodeTool: Author-writes or iteratively updates custom functional modules.

    ExecuteCodeTool: Dispatches executions to the sandbox worker.

## Architecture & Isolation

To safely process LLM-generated code blocks without putting the host machine at risk, this project enforces an isolated environment splitting logic and runtime operations:

    agent_controller: Houses the chat memory management loop and handles Ollama routing context.

    sandbox_worker: A lightweight python container mounted over a shared target workspace volume (./sandbox_workspace). Code executes entirely restricted down inside this stack under a strict 5-second timeout constraint using a non-root agentuser context.

## Technologies

This framework leverages modern containerized development ecosystems:

    Language: Python 3.11

    Local Inference: Ollama Engine

    Model Target: Gemma 4 (e2b)

    Virtualization: Docker & Docker Compose

## Installation & Prerequisites

Ensure you have Docker Desktop running and Ollama installed on your host system before proceeding.

    Pull the target model locally:
    Bash

    ollama pull gemma4:e2b

    Clone this repository and step into the root workspace:
    Bash

    git clone https://github.com/DavidRestle/mfca---my-first-coding-agent.git
    cd mfca---my-first-coding-agent

    Verify host resolution mapping:
    Ensure OLLAMA_HOST inside your docker-compose.yml points appropriately to your localized docker interface (typically http://host.docker.internal:11434).

## Execution / Usage

To interact with the agent natively inside an active terminal input session, spin up the containerized network and attach a pseudo-TTY shell:
Bash

Build the target agent configurations and spin up worker services

    docker compose run --rm agent_controller

Attach interactively to the agent controller's live console input stream

docker compose run --rm agent_controller

Multi-Turn Prompt Routines Example

    Turn 1: "Show me the files inside the sandbox workspace directory."

    Turn 2: "Write a python script named Print.py that prints a custom logging message, and save it using your tools."

    Turn 3: "Now locate that script and use your execute tool to verify its output works."

### Environment Teardown

To safely stop and remove the multi-container network setup when your session is finished:
Bash

docker compose down

## Change Log

    v1.2 (Current)

        Implemented multi-turn message loops enabling full conversation session memory tracking.

    v1.0

        Introduced the ExecuteCodeTool tied natively with standalone container isolation architectures.

    v0.3

        Configured WriteCodeTool and initial code-modification blocks.

    v0.2

        Added baseline analysis utilities (ReadDirectoryTool and ReadCodeTool).

    v0.1

        Initial prototype architecture blueprint setup.

## Contributors

    Author: David Restle

    Academic Context: Master's Student — Module: Advanced Media Technologies (AMT), Summer Semester 2026

## License

This project is open-source and licensed under the MIT License.