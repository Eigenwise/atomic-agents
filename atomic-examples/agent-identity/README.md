# Agent Identity Protocol

Demonstrates cryptographic agent identity and onboarding for interoperable atomic agents.

## What This Shows

1. **Agent Identity (L2)** — Each agent gets a unique cryptographic identity with a public key. This becomes the agent's permanent, verifiable identifier across frameworks.

2. **Onboarding (L1)** — Before an agent is trusted with work, it goes through structured onboarding. An authority issues a signed certificate listing the agent's verified capabilities.

3. **Handoff Signing** — When work transfers between agents (research → writer), the sender signs a handoff token. The receiver can cryptographically verify who sent the work and what context it carries.

## Run It

```bash
cd atomic-examples/agent-identity
uv run python agent_identity/main.py
```

## The Protocol

This example implements the concepts from the **Works With Agents Identity Protocol** (CC BY 4.0):

- **Spec:** https://workswithagents.com/specs/identity.md
- **SDK:** `pip install works-with-agents`
- **Full stack:** 16 protocols from L1 (Onboarding) to L7 (Compliance-as-Code)

The production SDK provides Ed25519 keypairs, X.509 certificate chains, and JWT-based handoff tokens — all with zero new dependencies beyond `pynacl`.

## Why Identity Matters for Atomic Agents

Atomic agents ("one agent, one job") are composable by design. But composition without identity is fragile:

- ❌ Agent A hands off to "some agent" → no audit trail
- ✅ Agent A signs a handoff to Agent B (public key verified) → cryptographic proof

Identity turns a collection of agents into a verifiable pipeline.
