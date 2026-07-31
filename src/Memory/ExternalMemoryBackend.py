import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


class ExternalMemoryBackend:
    """Persists conversation history to JSON files for cross-session recall."""

    def __init__(self, storage_dir: str = "./conversation_memory"):
        """
        Initialize external memory backend.

        Args:
            storage_dir: Directory to store conversation JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """
        Save conversation to a JSON file.

        Args:
            session_id: Unique session identifier
            messages: List of message dicts to save
        """
        file_path = self.storage_dir / f"{session_id}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "messages": messages,
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save session {session_id}: {e}")

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load conversation from a JSON file.

        Args:
            session_id: Unique session identifier

        Returns:
            List of message dicts, or empty list if not found
        """
        file_path = self.storage_dir / f"{session_id}.json"

        if not file_path.exists():
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("messages", [])
        except Exception as e:
            print(f"Warning: Failed to load session {session_id}: {e}")
            return []

    def list_sessions(self) -> List[str]:
        """
        List all saved session IDs.

        Returns:
            List of session identifiers
        """
        return sorted([f.stem for f in self.storage_dir.glob("*.json")])

    def delete(self, session_id: str) -> bool:
        """
        Delete a saved session.

        Args:
            session_id: Session to delete

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
                return True
            except Exception as e:
                print(f"Warning: Failed to delete session {session_id}: {e}")
                return False
        return False
