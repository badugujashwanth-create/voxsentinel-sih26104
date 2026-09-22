"""Ephemeral speaker profile references for local SIH operation."""

from __future__ import annotations

from dataclasses import dataclass
from secrets import token_urlsafe


@dataclass(frozen=True)
class SpeakerProfile:
    """Stored biometric reference metadata without raw source audio."""
    profile_id: str
    expected_speaker_id: str
    embedding: list[float]
    model_id: str
    model_revision: str
    embedding_dimensions: int
    threshold: float
    provenance: str


class SpeakerProfileRegistry:
    """Bounded process-local profile registry."""

    def __init__(self, maximum_profiles: int = 32) -> None:
        """Creates a registry with a fixed profile bound."""
        if maximum_profiles < 1:
            raise ValueError("maximum_profiles must be positive")
        self._maximum_profiles = maximum_profiles
        self._profiles: dict[str, SpeakerProfile] = {}

    def add(self, expected_speaker_id: str, embedding_payload: dict, provenance: str) -> SpeakerProfile:
        """Stores only the normalized embedding returned by the ML service."""
        if len(self._profiles) >= self._maximum_profiles:
            raise ValueError("speaker profile capacity is full")
        embedding = embedding_payload.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError("embedding is required")
        profile_id = token_urlsafe(12)
        profile = SpeakerProfile(profile_id=profile_id, expected_speaker_id=expected_speaker_id, embedding=[float(value) for value in embedding], model_id=str(embedding_payload["model_id"]), model_revision=str(embedding_payload["model_revision"]), embedding_dimensions=int(embedding_payload["embedding_dimensions"]), threshold=0.55, provenance=provenance)
        self._profiles[profile_id] = profile
        return profile

    def get(self, profile_id: str) -> SpeakerProfile | None:
        """Returns one profile without exposing source audio."""
        return self._profiles.get(profile_id)

    def require(self, profile_id: str) -> SpeakerProfile:
        """Returns a profile or rejects an unknown profile identifier."""
        profile = self.get(profile_id)
        if profile is None:
            raise KeyError(f"unknown speaker profile: {profile_id}")
        return profile

    def clear(self) -> None:
        """Disposes all in-memory profile references."""
        self._profiles.clear()
