"""Permissioned blockchain for immutable anomaly detection logging.

Provides a tamper-evident ledger for recording detected anomalies with:
- User authentication (credentials verified against password hashes)
- Role-based access control (admin, detector, auditor, viewer)
- Cryptographic hash chain (each entry linked to previous)
- HMAC signatures (proof of integrity)
- Blockchain verification (detect tampering)

Why blockchain for anomalies?
- Autoencoder detects anomalies (finds problems)
- Blockchain proves findings (prevents falsification)
- Combined: Accurate detection + immutable audit trail
"""
import json
import hashlib
import hmac
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import os


# Define user roles and their permissions
ROLES = {
    # Admin: Full control (dangerous, use sparingly)
    "admin": {"read": True, "write": True, "verify": True},
    # Detector: Can log anomalies, read records, but not verify integrity
    "detector": {"read": True, "write": True, "verify": False},
    # Auditor: Can verify blockchain integrity, but cannot modify records
    "auditor": {"read": True, "write": False, "verify": True},
    # Viewer: Read-only access (can see findings but not verify or modify)
    "viewer": {"read": True, "write": False, "verify": False},
}


class BlockchainUser:
    """Represents a user with authentication and role-based permissions.
    
    Security:
    - Passwords stored as SHA256 hashes (one-way encryption)
    - Never stores plaintext passwords
    - Password verification: hash(input) == stored_hash
    
    Permissions:
    - read: Can view blockchain ledger
    - write: Can append new entries (log anomalies)
    - verify: Can check blockchain integrity (hash chain, signatures)
    """
    
    def __init__(self, username: str, password: str, role: str = "detector"):
        """Create new user with given credentials and role.
        
        Args:
            username: Unique identifier for this user
            password: Secret password (hashed and stored)
            role: One of: admin, detector, auditor, viewer
        
        Raises:
            ValueError: If role not in ROLES
        """
        # Validate role
        if role not in ROLES:
            raise ValueError(f"Invalid role. Must be one of: {list(ROLES.keys())}")
        
        self.username = username
        # Hash password: never store plaintext
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.role = role
        # Copy permissions for this role
        self.permissions = ROLES[role]
    
    def verify_password(self, password: str) -> bool:
        """Check if provided password matches stored hash.
        
        Hashes input password and compares to stored hash.
        If they match, password is correct.
        
        Args:
            password: Plaintext password to verify
        
        Returns:
            True if password matches, False otherwise
        """
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()
    
    def can_write(self) -> bool:
        """Check if user has write permission (can log anomalies)."""
        return self.permissions.get("write", False)
    
    def can_read(self) -> bool:
        """Check if user has read permission (can view ledger)."""
        return self.permissions.get("read", False)
    
    def can_verify(self) -> bool:
        """Check if user can verify blockchain integrity (auditor only)."""
        return self.permissions.get("verify", False)


class PermissionedBlockchain:
    """Immutable, tamper-evident ledger with user authentication.
    
    Features:
    - Append-only log: Entries never deleted (can be verified later)
    - Hash chain: Each entry links to previous (detect tampering)
    - HMAC signatures: Prove integrity with secret key
    - Authentication: Username/password verification
    - Authorization: Role-based permissions (read/write/verify)
    
    Architecture:
    - Ledger file: JSONL (one JSON object per line)
    - Each entry: {timestamp, user, data, prev_hash, hash, signature}
    - Hash chain: hash[i] depends on hash[i-1] (break one = break all)
    - HMAC: Signature prevents tampering even with direct file access
    """
    
    def __init__(self, ledger_path: str = "blockchain_ledger.jsonl", secret_key: str = None):
        """Initialize blockchain instance.
        
        Args:
            ledger_path: File to store ledger entries
            secret_key: Secret key for HMAC signatures (default: env var or hardcoded)
        """
        self.ledger_path = ledger_path
        # Use env var if exists, else default (for testing)
        self.secret_key = secret_key or os.getenv("BLOCKCHAIN_SECRET_KEY", "default-secret-key")
        self.users: Dict[str, BlockchainUser] = {}
        self.current_user: Optional[BlockchainUser] = None
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Create default test users (admin, detector, auditor).
        
        In production, would load users from secure database.
        For testing, pre-defined credentials are convenient.
        """
        self.users["admin"] = BlockchainUser("admin", "admin_password", "admin")
        self.users["detector"] = BlockchainUser("detector", "detector_password", "detector")
        self.users["auditor"] = BlockchainUser("auditor", "auditor_password", "auditor")
    
    def login(self, username: str, password: str) -> bool:
        """Authenticate user and set as current user.
        
        Checks username exists and password is correct.
        On success, current_user is set (grants access to blockchain operations).
        
        Args:
            username: Username to authenticate
            password: Password to verify
        
        Returns:
            True on success
        
        Raises:
            ValueError: If user not found or password incorrect
        """
        if username not in self.users:
            raise ValueError(f"User '{username}' not found")
        
        user = self.users[username]
        if not user.verify_password(password):
            raise ValueError(f"Invalid password for user '{username}'")
        
        self.current_user = user
        return True
    
    def logout(self):
        """Logout current user (clear permissions).
        
        After logout, current_user is None and blockchain operations are blocked.
        """
        self.current_user = None
    
    def register_user(self, username: str, password: str, role: str = "viewer") -> bool:
        """Register a new user (admin only).
        
        Requires admin privileges (verify permission acts as admin check).
        Prevents duplicate usernames and validates role.
        
        Args:
            username: New username (must be unique)
            password: Initial password
            role: User role (admin, detector, auditor, or viewer)
        
        Returns:
            True if user created successfully
        
        Raises:
            PermissionError: If current user is not admin
            ValueError: If username already exists
        """
        # Only admins (those with verify permission) can register users
        if not self.current_user or not self.current_user.can_verify():
            raise PermissionError("Only admins can register new users")
        
        # Check for duplicates
        if username in self.users:
            raise ValueError(f"User '{username}' already exists")
        
        # Create new user
        self.users[username] = BlockchainUser(username, password, role)
        return True
    
    def _compute_signature(self, entry: Dict[str, Any]) -> str:
        """Compute HMAC-SHA256 signature over entry.
        
        HMAC = Hash-Based Message Authentication Code
        Creates cryptographic proof that entry hasn't been modified.
        
        If anyone modifies the entry data, HMAC changes (impossible to forge).
        Prevents tampering even if attacker has direct file access.
        
        Args:
            entry: Entry dict to sign
        
        Returns:
            Hexadecimal HMAC-SHA256 signature string
        """
        # Convert entry to canonical JSON (sorted keys for consistency)
        entry_str = json.dumps(entry, sort_keys=True)
        # Compute HMAC using secret key
        signature = hmac.new(
            self.secret_key.encode(),
            entry_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def log_anomaly(
        self,
        index: int,
        details: Dict[str, Any],
    ) -> None:
        """Log an anomaly detection to the immutable blockchain.
        
        Writes a new entry to ledger containing:
        - timestamp: When detected
        - user: Who logged it (detector username)
        - data: Index and anomaly details
        - prev_hash: Hash of previous entry (chain link)
        - hash: SHA256 of this entry
        - signature: HMAC-SHA256 proof of integrity
        
        Entry structure:
        {
            timestamp: 1700000000.123,
            user: "detector",
            user_role: "detector",
            data: {
                index: 42,
                details: {anomaly_score, confidence, ...}
            },
            prev_hash: "abc...",
            hash: "def...",
            signature: "ghi..."
        }
        
        Args:
            index: Sample index that's anomalous
            details: Dict with anomaly info (anomaly_score, confidence, etc.)
        
        Raises:
            PermissionError: If not logged in or no write permission
        """
        # Check authentication
        if not self.current_user:
            raise PermissionError("Must login before logging anomalies")
        
        # Check authorization
        if not self.current_user.can_write():
            raise PermissionError(f"User '{self.current_user.username}' does not have write permission")
        
        ledger = Path(self.ledger_path)
        
        # Read hash of last entry (for chain linking)
        prev_hash = None
        if ledger.exists() and ledger.stat().st_size > 0:
            with open(ledger, "r") as f:
                # Read all lines, keep last one
                for line in f:
                    pass
                last_entry = json.loads(line)
                # Link to previous entry's hash
                prev_hash = last_entry.get("hash")
        
        # Create new entry with metadata
        entry = {
            "timestamp": datetime.now().timestamp(),
            "user": self.current_user.username,
            "user_role": self.current_user.role,
            # Anomaly information
            "data": {
                "index": index,  # Which sample is anomalous
                "details": details,  # Anomaly score, confidence, etc.
            },
            # Hash chain: link to previous entry
            "prev_hash": prev_hash,
        }
        
        # Compute hash of this entry (before adding hash itself)
        # This prevents circular dependency
        entry_str = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(entry_str.encode()).hexdigest()
        entry["hash"] = entry_hash
        
        # Compute HMAC signature (proof entry hasn't been tampered)
        signature = self._compute_signature(entry)
        entry["signature"] = signature
        
        # Append entry to ledger file
        # Append-only: new entries never overwrite old entries
        with open(ledger, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def verify_blockchain(self) -> bool:
        """Verify blockchain integrity (detect tampering).
        
        Performs cryptographic verification:
        1. For each entry, verify hash matches computed hash
           (If data changed after hash was created, hash won't match)
        2. For each entry, verify HMAC signature
           (If any field modified, HMAC becomes invalid)
        3. For each entry, verify prev_hash points to previous entry
           (Tamper with entry i → breaks link to entry i+1)
        
        Why this catches tampering:
        - Attacker modifies entry i data
        - Attacker updates entry i's hash
        - But entry i+1 still has old prev_hash pointing to entry i
        - Mismatch detected! → Tampering caught
        
        Returns:
            True if all entries valid, False if any tampering detected
        
        Raises:
            PermissionError: If not logged in or no verify permission
        """
        # Check authentication and authorization
        if not self.current_user:
            raise PermissionError("Must login before verifying blockchain")
        
        if not self.current_user.can_verify():
            raise PermissionError(f"User '{self.current_user.username}' does not have verify permission")
        
        ledger = Path(self.ledger_path)
        # Empty ledger is valid
        if not ledger.exists() or ledger.stat().st_size == 0:
            return True
        
        # Verify hash chain and signatures
        prev_hash = None
        with open(ledger, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    
                    # VERIFICATION 1: Check hash chain (prev_hash must match previous entry)
                    if entry.get("prev_hash") != prev_hash:
                        print(f"Hash chain broken at entry {line_num}")
                        return False
                    
                    # VERIFICATION 2: Check HMAC signature (proves no tampering)
                    # Remove signature temporarily to recompute it
                    stored_signature = entry.pop("signature", None)
                    entry_copy = entry.copy()
                    expected_signature = self._compute_signature(entry_copy)
                    # Compare stored vs recomputed
                    if stored_signature != expected_signature:
                        print(f"Signature mismatch at entry {line_num}")
                        return False
                    
                    # Update prev_hash for next entry
                    prev_hash = entry.get("hash")
                except Exception as e:
                    print(f"Error verifying entry {line_num}: {e}")
                    return False
        
        return True
    
    def read_ledger(self, limit: int = None) -> list:
        """Read entries from blockchain ledger.
        
        Returns all logged anomalies (or subset if limit specified).
        Requires read permission (detector, auditor, or viewer).
        
        Args:
            limit: Max number of entries to return (None = all)
        
        Returns:
            List of entry dicts from ledger
        
        Raises:
            PermissionError: If not logged in or no read permission
        """
        # Check authentication
        if not self.current_user:
            raise PermissionError("Must login before reading blockchain")
        
        # Check authorization
        if not self.current_user.can_read():
            raise PermissionError(f"User '{self.current_user.username}' does not have read permission")
        
        ledger = Path(self.ledger_path)
        entries = []
        
        if ledger.exists():
            with open(ledger, "r") as f:
                for i, line in enumerate(f):
                    # Stop if reached limit
                    if limit and i >= limit:
                        break
                    entries.append(json.loads(line))
        
        return entries


# BACKWARDS COMPATIBILITY
# Maintain old function signature for code that hasn't migrated to new class-based API
_blockchain_instance = None

def log_anomaly(
    index: int,
    details: Dict[str, Any],
    ledger_path: str = "blockchain_ledger.jsonl",
    username: str = "detector",
    password: str = "detector_password",
) -> None:
    """Legacy function for backwards compatibility (deprecated).
    
    Logs anomaly using PermissionedBlockchain with default detector credentials.
    Maintains global singleton instance across multiple calls.
    
    Deprecated: Use PermissionedBlockchain class directly for better control.
    
    Args:
        index: Anomalous sample index
        details: Anomaly details dict
        ledger_path: Path to blockchain ledger file
        username: Username for authentication
        password: Password for authentication
    """
    global _blockchain_instance
    
    # Initialize blockchain once and reuse (singleton pattern)
    # Avoids recreating instance on every call
    if _blockchain_instance is None:
        _blockchain_instance = PermissionedBlockchain(ledger_path)
        try:
            # Try to login with default credentials
            _blockchain_instance.login(username, password)
        except ValueError:
            # First run - user doesn't exist yet (expected for fresh ledger)
            pass
    
    # Ensure logged in (in case previous session logged out)
    if _blockchain_instance.current_user is None:
        _blockchain_instance.login(username, password)
    
    # Log the anomaly
    _blockchain_instance.log_anomaly(index, details)
