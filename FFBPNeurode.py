"""FFBPNeurode class inherited from FFNeurode and BPNeurode."""

from FFNeurode import FFNeurode
from BPNeurode import BPNeurode


class FFBPNeurode(FFNeurode, BPNeurode):
    """FFBPNeurode class inherited from FFNeurode and BPNeurode."""

    def __init__(self):
        """Initialize FFBPNeurode."""
        super().__init__()
