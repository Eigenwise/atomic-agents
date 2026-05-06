"""
Agent Identity Protocol — demonstrate cryptographic agent identity
for interoperable atomic agents using Ed25519 keypairs.

This is a standalone demonstration of the Identity Protocol concept.
It uses only the Python standard library (no external deps beyond atomic-agents).

The Works With Agents Identity Protocol (CC BY 4.0) standardizes this pattern:
    https://workswithagents.com/specs/identity.md

Full SDK: pip install works-with-agents
"""

import base64
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Ed25519 is available in Python 3.12+ stdlib via cryptography primitives.
# For demonstration we use the hashlib-based approach, but the protocol
# specifies Ed25519 for production use (pip install pynacl or use the
# works-with-agents SDK which bundles it).


@dataclass
class AgentIdentity:
    """Cryptographic identity for an atomic agent.

    Implements the Works With Agents Identity Protocol (L2).
    Each agent gets an Ed25519 keypair. The public key is the agent's
    permanent identifier. The private key signs capability manifests
    and handoff tokens.
    """

    agent_id: str
    public_key: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0.0"

    def fingerprint(self) -> str:
        """Stable, human-readable fingerprint for agent identification."""
        return hashlib.sha256(
            f"{self.agent_id}:{self.public_key}".encode()
        ).hexdigest()[:16]

    def sign_claim(self, claim: dict) -> str:
        """Sign a claim (capability manifest, handoff token, etc.).

        In production this uses Ed25519. Here we demonstrate the concept
        with a hash-based signature that shows the protocol structure.
        """
        payload = json.dumps(claim, sort_keys=True).encode()
        # Production: ed25519.SigningKey.sign(payload)
        return base64.b64encode(
            hashlib.sha256(self.public_key.encode() + payload).digest()
        ).decode()

    @classmethod
    def create(cls, agent_id: str) -> "AgentIdentity":
        """Create a new agent identity with a generated keypair.

        In production this generates an Ed25519 keypair:
            from nacl.signing import SigningKey
            sk = SigningKey.generate()
            pk = sk.verify_key
        """
        # Demo keypair — in production, use Ed25519
        demo_key = base64.b64encode(os.urandom(32)).decode()
        return cls(agent_id=agent_id, public_key=demo_key)


@dataclass
class OnboardingCertificate:
    """Proof that an agent completed onboarding (L1 Onboarding Protocol).

    Issued by an onboarding authority after the agent demonstrates
    basic functionality: tool execution, schema compliance, error handling.
    """

    agent_id: str
    authority: str
    capabilities: list[str] = field(default_factory=list)
    issued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    signature: str = ""

    def is_valid(self, authority_identity: "AgentIdentity") -> bool:
        """Verify the onboarding certificate against the authority's identity."""
        payload = {
            "agent_id": self.agent_id,
            "authority": self.authority,
            "capabilities": self.capabilities,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
        }
        expected = authority_identity.sign_claim(payload)
        return expected == self.signature


# ──────────────────────────────────────────────────────────────
# Demo: Create atomic agents with cryptographic identity
# ──────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("Agent Identity Protocol — Demonstration")
    print("=" * 60)

    # 1. Create identities for two agents
    research_agent = AgentIdentity.create("research-agent-01")
    writer_agent = AgentIdentity.create("writer-agent-01")
    authority = AgentIdentity.create("onboarding-authority")

    print(f"\n🔑 Research Agent: {research_agent.agent_id}")
    print(f"   Fingerprint: {research_agent.fingerprint()}")
    print(f"   Public Key:  {research_agent.public_key[:32]}...")

    print(f"\n🔑 Writer Agent:   {writer_agent.agent_id}")
    print(f"   Fingerprint: {writer_agent.fingerprint()}")
    print(f"   Public Key:  {writer_agent.public_key[:32]}...")

    # 2. Onboard agents — prove they can do their job
    cert = OnboardingCertificate(
        agent_id=research_agent.agent_id,
        authority=authority.agent_id,
        capabilities=["web_search", "document_analysis", "citation_extraction"],
    )
    cert.signature = authority.sign_claim({
        "agent_id": cert.agent_id,
        "authority": cert.authority,
        "capabilities": cert.capabilities,
        "issued_at": cert.issued_at,
        "expires_at": cert.expires_at,
    })

    print(f"\n📜 Onboarding Certificate")
    print(f"   Issued to:  {cert.agent_id}")
    print(f"   Capabilities: {', '.join(cert.capabilities)}")
    print(f"   Verified:   {'✅' if cert.is_valid(authority) else '❌'}")

    # 3. Sign a handoff token — proof of task transfer
    handoff = {
        "from": research_agent.agent_id,
        "to": writer_agent.agent_id,
        "task": "write_report",
        "context_hash": hashlib.sha256(b"research findings...").hexdigest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    handoff["signature"] = research_agent.sign_claim(handoff)

    print(f"\n🤝 Handoff Token (signed)")
    print(f"   {handoff['from']} → {handoff['to']}")
    print(f"   Task: {handoff['task']}")
    print(f"   Context: {handoff['context_hash'][:16]}...")
    print(f"   Signature: {handoff['signature'][:32]}...")

    # 4. Verify the handoff
    verify_payload = {k: v for k, v in handoff.items() if k != "signature"}
    expected_sig = research_agent.sign_claim(verify_payload)
    print(f"   Verified:  {'✅' if expected_sig == handoff['signature'] else '❌'}")

    print(f"\n{'=' * 60}")
    print("This demonstrates the Identity + Onboarding protocol stack.")
    print("For production Ed25519 signing, see:")
    print("  https://workswithagents.com/specs/identity.md")
    print("  pip install works-with-agents")
    print("=" * 60)


if __name__ == "__main__":
    main()
