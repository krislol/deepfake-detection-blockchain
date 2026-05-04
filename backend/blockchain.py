"""
Blockchain Service for hash registration and verification.

Simulation mode stores a real block structure (genesis block, transactions grouped
into blocks of BLOCK_SIZE) in data/blockchain_simulation.json.

Polygon mode connects to the real network via Web3 (requires .env setup).
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

SIMULATION_DB = Path(__file__).parent.parent / "data" / "blockchain_simulation.json"
BLOCK_SIZE = 5  # transactions per mined block


class BlockchainService:

    def __init__(self):
        self.mode = self._detect_mode()
        logger.info(f"Blockchain service initialised in {self.mode.upper()} mode")

        if self.mode == "polygon":
            self._init_polygon()
        else:
            self._init_simulation()

    def _detect_mode(self) -> str:
        has_contract = bool(os.getenv("CONTRACT_ADDRESS"))
        has_key      = bool(os.getenv("PRIVATE_KEY"))
        has_rpc      = bool(os.getenv("POLYGON_RPC_URL"))
        return "polygon" if (has_contract and has_key and has_rpc) else "simulation"

    # -----------------------------------------------------------------------
    # SIMULATION MODE — file I/O
    # -----------------------------------------------------------------------

    def _init_simulation(self):
        SIMULATION_DB.parent.mkdir(parents=True, exist_ok=True)
        if not SIMULATION_DB.exists():
            genesis = self._make_genesis()
            self._save_chain({"chain": [genesis], "pending_transactions": []})
        else:
            # Ensure genesis block exists (handles legacy empty-object files)
            data = self._load_chain()
            if not data.get("chain"):
                data["chain"] = [self._make_genesis()]
                data.setdefault("pending_transactions", [])
                self._save_chain(data)
        logger.info(
            "SIMULATION mode — chain stored in data/blockchain_simulation.json. "
            "Set CONTRACT_ADDRESS, PRIVATE_KEY, POLYGON_RPC_URL to use Polygon."
        )

    def _load_chain(self) -> Dict:
        return json.loads(SIMULATION_DB.read_text(encoding="utf-8"))

    def _save_chain(self, data: Dict) -> None:
        SIMULATION_DB.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # -----------------------------------------------------------------------
    # BLOCK CONSTRUCTION
    # -----------------------------------------------------------------------

    def _make_genesis(self) -> Dict:
        ts = datetime.utcnow().isoformat()
        return {
            "block_number": 0,
            "timestamp": ts,
            "block_hash": "0x" + hashlib.sha256(b"deepfake-detection-genesis-block").hexdigest(),
            "previous_block_hash": "0x" + "0" * 64,
            "transaction_count": 0,
            "transactions": []
        }

    def _mine_block(self, data: Dict) -> Dict:
        """Move all pending transactions into a new block and append it to the chain."""
        pending    = data["pending_transactions"]
        chain      = data["chain"]
        prev_hash  = chain[-1]["block_hash"]
        block_num  = len(chain)
        ts         = datetime.utcnow().isoformat()
        tx_hashes  = "".join(tx["tx_hash"] for tx in pending)
        raw        = f"{block_num}{prev_hash}{tx_hashes}{ts}".encode()
        block_hash = "0x" + hashlib.sha256(raw).hexdigest()

        for tx in pending:
            tx["block_number"] = block_num
            tx["block_hash"]   = block_hash

        block = {
            "block_number":       block_num,
            "timestamp":          ts,
            "block_hash":         block_hash,
            "previous_block_hash": prev_hash,
            "transaction_count":  len(pending),
            "transactions":       pending
        }
        chain.append(block)
        data["pending_transactions"] = []
        logger.info(f"[SIMULATION] Mined Block #{block_num} — {len(pending)} tx(s) | {block_hash[:20]}…")
        return block

    # -----------------------------------------------------------------------
    # SIMULATION IMPLEMENTATIONS
    # -----------------------------------------------------------------------

    def _find_transaction(self, data: Dict, file_hash: str) -> Optional[Dict]:
        for block in data["chain"]:
            for tx in block["transactions"]:
                if tx["file_hash"] == file_hash:
                    return tx
        for tx in data["pending_transactions"]:
            if tx["file_hash"] == file_hash:
                return tx
        return None

    def _register_simulation(self, file_hash: str, filename: str) -> Dict:
        data = self._load_chain()

        existing = self._find_transaction(data, file_hash)
        if existing:
            return {
                "success":                  True,
                "transaction_hash":         existing["tx_hash"],
                "block_number":             existing.get("block_number"),
                "block_hash":               existing.get("block_hash"),
                "pending_count":            len(data["pending_transactions"]),
                "transactions_until_block": BLOCK_SIZE - len(data["pending_transactions"]),
                "message":                  "Hash already registered",
                "network":                  "simulation"
            }

        ts      = datetime.utcnow().isoformat()
        tx_hash = "0x" + hashlib.sha256(f"{file_hash}{filename}{ts}".encode()).hexdigest()

        tx = {
            "tx_hash":      tx_hash,
            "file_hash":    file_hash,
            "filename":     filename,
            "registered_at": ts,
            "block_number": None,
            "block_hash":   None
        }
        data["pending_transactions"].append(tx)

        mined_block = None
        if len(data["pending_transactions"]) >= BLOCK_SIZE:
            mined_block = self._mine_block(data)

        self._save_chain(data)

        pending_count = len(data["pending_transactions"])

        if mined_block:
            block_number = mined_block["block_number"]
            block_hash   = mined_block["block_hash"]
            msg = f"Registered and confirmed in Block #{block_number} (simulation)"
        else:
            block_number = None
            block_hash   = None
            remaining    = BLOCK_SIZE - pending_count
            msg = (
                f"Transaction pending — {remaining} more transaction(s) "
                f"needed to mine the next block (simulation)"
            )

        logger.info(f"[SIMULATION] Registered {file_hash[:16]}… → {tx_hash[:20]}…")

        return {
            "success":                  True,
            "transaction_hash":         tx_hash,
            "block_number":             block_number,
            "block_hash":               block_hash,
            "pending_count":            pending_count,
            "transactions_until_block": BLOCK_SIZE - pending_count,
            "message":                  msg,
            "network":                  "simulation"
        }

    def _verify_simulation(self, file_hash: str) -> Dict:
        data = self._load_chain()
        tx   = self._find_transaction(data, file_hash)

        if tx:
            return {
                "is_registered": True,
                "registered_at": tx.get("registered_at"),
                "filename":      tx.get("filename"),
                "block_number":  tx.get("block_number"),
                "block_hash":    tx.get("block_hash"),
                "is_confirmed":  tx.get("block_number") is not None,
                "network":       "simulation"
            }

        return {
            "is_registered": False,
            "registered_at": None,
            "filename":      None,
            "block_number":  None,
            "block_hash":    None,
            "is_confirmed":  False,
            "network":       "simulation"
        }

    # -----------------------------------------------------------------------
    # PUBLIC INTERFACE
    # -----------------------------------------------------------------------

    def register_hash(self, file_hash: str, filename: str) -> Dict:
        if self.mode == "polygon":
            return self._register_polygon(file_hash, filename)
        return self._register_simulation(file_hash, filename)

    def verify_hash(self, file_hash: str) -> Dict:
        if self.mode == "polygon":
            return self._verify_polygon(file_hash)
        return self._verify_simulation(file_hash)

    def get_chain(self) -> Dict:
        """Return the full chain with stats. Simulation mode only."""
        if self.mode != "simulation":
            return {"error": "get_chain is only available in simulation mode"}

        data          = self._load_chain()
        confirmed_tx  = sum(b["transaction_count"] for b in data["chain"])
        pending_count = len(data["pending_transactions"])

        return {
            "mode":                 "simulation",
            "network":              "Polygon (Simulation)",
            "chain":                data["chain"],
            "pending_transactions": data["pending_transactions"],
            "stats": {
                "total_blocks":                    len(data["chain"]),
                "total_confirmed_transactions":    confirmed_tx,
                "pending_transactions":            pending_count,
                "transactions_per_block":          BLOCK_SIZE,
                "transactions_until_next_block":   BLOCK_SIZE - pending_count
            }
        }

    # -----------------------------------------------------------------------
    # POLYGON MODE
    # -----------------------------------------------------------------------

    def _init_polygon(self):
        try:
            from web3 import Web3

            rpc_url       = os.getenv("POLYGON_RPC_URL")
            self.w3       = Web3(Web3.HTTPProvider(rpc_url))
            self.account  = self.w3.eth.account.from_key(os.getenv("PRIVATE_KEY"))
            contract_addr = os.getenv("CONTRACT_ADDRESS")

            abi = [
                {
                    "inputs": [
                        {"name": "fileHash", "type": "string"},
                        {"name": "filename", "type": "string"}
                    ],
                    "name": "registerFile",
                    "outputs": [],
                    "stateMutability": "nonpayable",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "fileHash", "type": "string"}],
                    "name": "verifyFile",
                    "outputs": [{"name": "", "type": "bool"}],
                    "stateMutability": "view",
                    "type": "function"
                },
                {
                    "inputs": [{"name": "fileHash", "type": "string"}],
                    "name": "getFileDetails",
                    "outputs": [
                        {"name": "fileHash",  "type": "string"},
                        {"name": "timestamp", "type": "uint256"},
                        {"name": "uploader",  "type": "address"},
                        {"name": "filename",  "type": "string"},
                        {"name": "verified",  "type": "bool"}
                    ],
                    "stateMutability": "view",
                    "type": "function"
                }
            ]

            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(contract_addr),
                abi=abi
            )
            logger.info(f"Connected to Polygon | Account: {self.account.address}")

        except Exception as e:
            logger.error(f"Polygon connection failed: {e}. Falling back to simulation mode.")
            self.mode = "simulation"
            self._init_simulation()

    def _register_polygon(self, file_hash: str, filename: str) -> Dict:
        try:
            tx = self.contract.functions.registerFile(
                file_hash, filename
            ).build_transaction({
                "from":     self.account.address,
                "nonce":    self.w3.eth.get_transaction_count(self.account.address),
                "gas":      300000,
                "gasPrice": self.w3.eth.gas_price
            })
            signed   = self.account.sign_transaction(tx)
            tx_hash  = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt  = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            logger.info(f"[POLYGON] Registered on-chain: {tx_hash.hex()[:20]}…")
            return {
                "success":          True,
                "transaction_hash": tx_hash.hex(),
                "block_number":     receipt.blockNumber,
                "block_hash":       receipt.blockHash.hex(),
                "pending_count":    0,
                "transactions_until_block": 0,
                "message":          "Hash registered on Polygon blockchain",
                "network":          "polygon"
            }
        except Exception as e:
            logger.error(f"Polygon registration error: {e}")
            return {
                "success":          False,
                "transaction_hash": None,
                "block_number":     None,
                "block_hash":       None,
                "pending_count":    None,
                "transactions_until_block": None,
                "message":          f"Blockchain error: {str(e)}",
                "network":          "polygon"
            }

    def _verify_polygon(self, file_hash: str) -> Dict:
        try:
            is_registered = self.contract.functions.verifyFile(file_hash).call()
            if is_registered:
                details   = self.contract.functions.getFileDetails(file_hash).call()
                timestamp = datetime.utcfromtimestamp(details[1]).isoformat()
                return {
                    "is_registered": True,
                    "registered_at": timestamp,
                    "filename":      details[3],
                    "block_number":  None,
                    "block_hash":    None,
                    "is_confirmed":  True,
                    "network":       "polygon"
                }
            return {
                "is_registered": False,
                "registered_at": None,
                "filename":      None,
                "block_number":  None,
                "block_hash":    None,
                "is_confirmed":  False,
                "network":       "polygon"
            }
        except Exception as e:
            logger.error(f"Polygon verification error: {e}")
            return {
                "is_registered": False,
                "registered_at": None,
                "filename":      None,
                "block_number":  None,
                "block_hash":    None,
                "is_confirmed":  False,
                "network":       "polygon"
            }
