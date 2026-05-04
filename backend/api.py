"""
FastAPI Backend for Deepfake Detection System.
Handles video uploads, deepfake detection, and blockchain verification.
"""

import os
import sys
import time
import hashlib
import tempfile
import logging
from pathlib import Path
from typing import Optional

# ── Path & env setup (must come before local imports) ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ───────────────────────────────────────────────────────────────────────────

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.models import (
    DetectionResponse,
    RegisterResponse,
    VerifyResponse,
    HealthResponse,
    ErrorResponse
)
from backend.blockchain import BlockchainService
from main import DeepfakeDetectionPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# APP INITIALISATION
# ============================================================================

app = FastAPI(
    title="Deepfake Detection API",
    description="Detect deepfakes using AI and verify file authenticity via blockchain.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}

_pipeline: Optional[DeepfakeDetectionPipeline] = None
_blockchain: Optional[BlockchainService] = None


def get_pipeline() -> DeepfakeDetectionPipeline:
    global _pipeline
    if _pipeline is None:
        model_path = os.getenv("MODEL_PATH", None)

        # Resolve relative path against project root
        if model_path:
            resolved = PROJECT_ROOT / model_path
            if resolved.exists():
                model_path = str(resolved)
                logger.info(f"Loading trained model from: {model_path}")
            else:
                logger.warning(f"Model file not found at {resolved} — using base weights")
                model_path = None
        else:
            logger.warning("MODEL_PATH not set in .env — using base ImageNet weights")

        _pipeline = DeepfakeDetectionPipeline(model_path=model_path)
    return _pipeline


def get_blockchain_service() -> BlockchainService:
    global _blockchain
    if _blockchain is None:
        _blockchain = BlockchainService()
    return _blockchain


def compute_sha256(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def validate_video_file(file: UploadFile) -> None:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {', '.join(ALLOWED_EXTENSIONS)}"
        )


# ============================================================================
# ROUTES
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Deepfake Detection API",
        "version": "1.0.0",
        "model_path": os.getenv("MODEL_PATH", "Not set"),
        "endpoints": {
            "GET  /health":        "Health check",
            "POST /detect":        "Detect deepfake in uploaded video",
            "POST /register":      "Register file hash on blockchain",
            "GET  /verify/{hash}": "Verify file authenticity via blockchain",
            "GET  /blockchain":    "View full simulated blockchain",
            "GET  /docs":          "Interactive API documentation",
        }
    }


@app.get("/blockchain")
async def get_blockchain():
    """
    Return the full simulated blockchain — all blocks, transactions, and pending queue.

    Each block contains up to 5 transactions. When 5 transactions accumulate, they
    are grouped into a new block with a computed block hash chained to the previous block.
    """
    blockchain = get_blockchain_service()
    return blockchain.get_chain()


@app.get("/health", response_model=HealthResponse)
async def health():
    model_path = os.getenv("MODEL_PATH", None)
    resolved = PROJECT_ROOT / model_path if model_path else None
    model_loaded = resolved.exists() if resolved else False

    return HealthResponse(
        status="healthy",
        message=f"API running | Model loaded: {model_loaded} | Path: {model_path}"
    )


@app.post("/detect", response_model=DetectionResponse)
async def detect_deepfake(
    file: UploadFile = File(..., description="Video file to analyse"),
    aggregation: str = "mean"
):
    """Run the full detection pipeline on an uploaded video and return the verdict."""
    start_time = time.time()
    validate_video_file(file)

    if aggregation not in {"mean", "max", "median", "weighted"}:
        raise HTTPException(status_code=400, detail="aggregation must be one of: mean, max, median, weighted")

    tmp_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_hash = compute_sha256(tmp_path)
        pipeline = get_pipeline()
        result = pipeline.process_video(tmp_path, aggregation_method=aggregation)

        if not result["success"]:
            raise HTTPException(status_code=422, detail=result.get("error", "Detection failed"))

        pred = result["video_prediction"]
        processing_time = round(time.time() - start_time, 2)

        logger.info(f"Detection: {pred['classification']} (p={pred['fake_probability']:.3f}) in {processing_time}s")

        return DetectionResponse(
            success=True,
            filename=file.filename,
            file_hash=file_hash,
            classification=pred["classification"],
            fake_probability=round(pred["fake_probability"], 4),
            real_probability=round(pred["real_probability"], 4),
            confidence=round(pred["confidence"], 4),
            frames_analysed=result["num_frames_analyzed"],
            faces_detected=result["num_faces_detected"],
            aggregation_method=aggregation,
            processing_time_seconds=processing_time
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/register", response_model=RegisterResponse)
async def register_file(
    file: UploadFile = File(..., description="Video file to register on blockchain"),
):
    """Hash the uploaded file and register it on the blockchain as a tamper-proof fingerprint."""
    validate_video_file(file)
    tmp_path = None
    try:
        suffix = Path(file.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        file_hash = compute_sha256(tmp_path)
        blockchain = get_blockchain_service()
        tx_result = blockchain.register_hash(file_hash, file.filename)

        return RegisterResponse(
            success=tx_result["success"],
            filename=file.filename,
            file_hash=file_hash,
            transaction_hash=tx_result.get("transaction_hash"),
            block_number=tx_result.get("block_number"),
            block_hash=tx_result.get("block_hash"),
            pending_count=tx_result.get("pending_count"),
            transactions_until_block=tx_result.get("transactions_until_block"),
            message=tx_result.get("message", "Hash registered successfully"),
            blockchain_network=tx_result.get("network", "simulation")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/verify/{file_hash}", response_model=VerifyResponse)
async def verify_file(file_hash: str):
    """Check whether a SHA-256 hash exists on the blockchain."""
    if len(file_hash) != 64:
        raise HTTPException(status_code=400, detail="Invalid hash. SHA-256 hashes must be exactly 64 characters.")

    try:
        blockchain = get_blockchain_service()
        result = blockchain.verify_hash(file_hash)

        registered = result["is_registered"]
        confirmed  = result.get("is_confirmed", False)

        if registered and confirmed:
            msg = f"Registered and confirmed in Block #{result.get('block_number')}."
        elif registered:
            msg = "Transaction is pending — not yet included in a block."
        else:
            msg = "Hash not found. File may have been modified or was never registered."

        return VerifyResponse(
            file_hash=file_hash,
            is_registered=registered,
            registered_at=result.get("registered_at"),
            original_filename=result.get("filename"),
            block_number=result.get("block_number"),
            block_hash=result.get("block_hash"),
            is_confirmed=confirmed,
            blockchain_network=result.get("network", "simulation"),
            message=msg
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api:app", host="0.0.0.0", port=8000, reload=True)
