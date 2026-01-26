"""Configuration du logging pour l'agent."""
import logging
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "agent.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("agent_btp_v2")
