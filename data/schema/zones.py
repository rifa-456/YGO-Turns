from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MonsterZoneColumns:
    name: list[str]
    atk: list[str]
    defense: list[str]
    position: list[str]
    face_down: list[str]
    counters: list[str]
    equips: list[list[str]]
    equip_sizes: list[str]


@dataclass(frozen=True)
class SpellZoneColumns:
    name: list[str]
    face_down: list[str]
    counters: list[str]


@dataclass(frozen=True)
class FieldSpellZoneColumns:
    name: str
    counters: str
