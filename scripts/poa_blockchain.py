"""Simple permissioned Proof-of-Authority blockchain for anomaly provenance.

This module provides a small proof-of-concept permissioned blockchain with:

- Three authorized validator nodes
- Proof-of-Authority block creation
- SHA-256 hash chaining
- HMAC-SHA256 block authentication
- Replicated blockchain copies across validators
- Blockchain integrity verification
- Rejection of unauthorized validators
"""

import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import os


# Authorized validator nodes
AUTHORIZED_VALIDATORS = [
    "validator_1",
    "validator_2",
    "validator_3",
]


class PoABlockchain:
    """Small permissioned Proof-of-Authority blockchain.

    Blocks can only be created by authorized validators.
    Each validator maintains a replicated copy of the blockchain.
    """

    def __init__(
        self,
        ledger_dir: str = "poa_ledgers",
        secret_key: str = None,
    ):
        """Initialize the permissioned blockchain.

        Args:
            ledger_dir: Directory containing validator ledger files
            secret_key: Secret key used for HMAC authentication
        """

        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

        self.secret_key = secret_key or os.getenv(
            "BLOCKCHAIN_SECRET_KEY",
            "default-secret-key",
        )

        self.validators = AUTHORIZED_VALIDATORS.copy()

        # Used for simple round-robin validator selection
        self.validator_index = 0

    def _ledger_path(self, validator: str) -> Path:
        """Return the ledger file belonging to a validator."""

        return self.ledger_dir / f"{validator}_ledger.jsonl"

    def is_authorized_validator(self, validator: str) -> bool:
        """Check whether a validator belongs to the permissioned network."""

        return validator in self.validators

    def get_current_validator(self) -> str:
        """Return the validator currently authorized to create the next block."""

        return self.validators[self.validator_index]

    def _advance_validator(self) -> None:
        """Move authority to the next validator."""

        self.validator_index = (
            self.validator_index + 1
        ) % len(self.validators)

    def _read_chain(self, validator: str) -> List[Dict[str, Any]]:
        """Read the blockchain stored by one validator."""

        ledger = self._ledger_path(validator)

        if not ledger.exists():
            return []

        blocks = []

        with open(ledger, "r") as f:
            for line in f:
                if line.strip():
                    blocks.append(json.loads(line))

        return blocks

    def _get_previous_hash(self) -> str:
        """Return the hash of the latest block."""

        chain = self._read_chain(self.validators[0])

        if not chain:
            return None

        return chain[-1]["hash"]

    def _compute_hash(self, block: Dict[str, Any]) -> str:
        """Compute the SHA-256 hash of a block."""

        block_copy = block.copy()
        block_copy.pop("hash", None)
        block_copy.pop("signature", None)

        block_string = json.dumps(
            block_copy,
            sort_keys=True,
        )

        return hashlib.sha256(
            block_string.encode()
        ).hexdigest()

    def _compute_signature(self, block: Dict[str, Any]) -> str:
        """Compute an HMAC-SHA256 authentication tag for a block."""

        block_copy = block.copy()
        block_copy.pop("signature", None)

        block_string = json.dumps(
            block_copy,
            sort_keys=True,
        )

        return hmac.new(
            self.secret_key.encode(),
            block_string.encode(),
            hashlib.sha256,
        ).hexdigest()

    def create_block(
        self,
        validator: str,
        anomaly_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create and replicate a new block.

        Only the validator selected by the round-robin PoA mechanism
        can create the next block.

        Args:
            validator: Validator proposing the block
            anomaly_data: Anomaly information to store

        Returns:
            Newly created block

        Raises:
            PermissionError: If validator is unauthorized or not selected
        """

        # Validator must belong to the permissioned network
        if not self.is_authorized_validator(validator):
            raise PermissionError(
                f"Validator '{validator}' is not authorized."
            )

        # Proof-of-Authority validator selection
        expected_validator = self.get_current_validator()

        if validator != expected_validator:
            raise PermissionError(
                f"Validator '{validator}' cannot create this block. "
                f"Expected validator: '{expected_validator}'."
            )

        previous_hash = self._get_previous_hash()

        # Create block
        block = {
            "index": len(self._read_chain(self.validators[0])),
            "timestamp": datetime.now().timestamp(),
            "validator": validator,
            "data": anomaly_data,
            "previous_hash": previous_hash,
        }

        # Add SHA-256 block hash
        block["hash"] = self._compute_hash(block)

        # Add HMAC-SHA256 authentication tag
        block["signature"] = self._compute_signature(block)

        # Replicate the block across all validator nodes
        self._replicate_block(block)

        # Give authority to the next validator
        self._advance_validator()

        return block

    def _replicate_block(self, block: Dict[str, Any]) -> None:
        """Append a validated block to every validator ledger."""

        for validator in self.validators:

            ledger = self._ledger_path(validator)

            with open(ledger, "a") as f:
                f.write(json.dumps(block) + "\n")

    def verify_chain(self, validator: str) -> bool:
        """Verify one validator's complete blockchain.

        Verification checks:

        1. Validator identity
        2. Previous-block hash linkage
        3. SHA-256 block hash
        4. HMAC-SHA256 authentication tag
        """

        if not self.is_authorized_validator(validator):
            raise PermissionError(
                f"Validator '{validator}' is not authorized."
            )

        chain = self._read_chain(validator)

        previous_hash = None

        for block in chain:

            # Check block creator is authorized
            if not self.is_authorized_validator(
                block.get("validator")
            ):
                return False

            # Check previous hash
            if block.get("previous_hash") != previous_hash:
                return False

            # Check block hash
            stored_hash = block.get("hash")

            expected_hash = self._compute_hash(block)

            if stored_hash != expected_hash:
                return False

            # Check HMAC authentication tag
            stored_signature = block.get("signature")

            expected_signature = self._compute_signature(block)

            if stored_signature != expected_signature:
                return False

            previous_hash = stored_hash

        return True

    def verify_all_validators(self) -> bool:
        """Verify all validator chains and confirm replication consistency."""

        chains = []

        for validator in self.validators:

            if not self.verify_chain(validator):
                return False

            chains.append(
                self._read_chain(validator)
            )

        # All validators must contain exactly the same blockchain
        first_chain = chains[0]

        for chain in chains[1:]:

            if chain != first_chain:
                return False

        return True

    def get_chain(self, validator: str) -> List[Dict[str, Any]]:
        """Return the blockchain stored by an authorized validator."""

        if not self.is_authorized_validator(validator):
            raise PermissionError(
                f"Validator '{validator}' is not authorized."
            )

        return self._read_chain(validator)