from __future__ import annotations

from enum import Enum


class CardCategory(str, Enum):
    MONSTER = "Monster"
    SPELL = "Spell"
    TRAP = "Trap"
    SKILL = "Skill"


class MonsterCategory(str, Enum):
    NORMAL = "Normal"
    EFFECT = "Effect"
    FUSION = "Fusion"
    RITUAL = "Ritual"
    TOON = "Toon"
    SPIRIT = "Spirit"
    UNION = "Union"
    GEMINI = "Gemini"
    SYNCHRO = "Synchro"
    XYZ = "Xyz"
    LINK = "Link"


class Attribute(str, Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    EARTH = "EARTH"
    WATER = "WATER"
    FIRE = "FIRE"
    WIND = "WIND"
    DIVINE = "DIVINE"


class MonsterRace(str, Enum):
    DRAGON = "Dragon"
    SPELLCASTER = "Spellcaster"
    WARRIOR = "Warrior"
    MACHINE = "Machine"
    ZOMBIE = "Zombie"
    FIEND = "Fiend"
    FAIRY = "Fairy"
    AQUA = "Aqua"
    PYRO = "Pyro"
    ROCK = "Rock"
    THUNDER = "Thunder"
    DINOSAUR = "Dinosaur"
    BEAST = "Beast"
    BEAST_WARRIOR = "Beast-Warrior"
    WINGED_BEAST = "Winged Beast"
    INSECT = "Insect"
    PLANT = "Plant"
    REPTILE = "Reptile"
    GALAXY = "Galaxy"
    ILLUSION = "Illusion"
    CYBERSE = "Cyberse"
    PSYCHIC = "Psychic"
    WYRM = "Wyrm"
    FISH = "Fish"
    HIGH_DRAGON = "High Dragon"
    CELESTIAL_WARRIOR = "Celestial Warrior"
    MAGICAL_KNIGHT = "Magical Knight"
    CYBORG = "Cyborg"
    OMEGA_PSYCHIC = "Omega Psychic"
    DIVINE_BEAST = "Divine-Beast"
    PROXY = "Proxy"
    CREATOR_GOD = "Creator God"
    SEA_SERPENT = "Sea Serpent"


class SpellType(str, Enum):
    NORMAL = "Normal"
    CONTINUOUS = "Continuous"
    EQUIP = "Equip"
    QUICK_PLAY = "Quick-Play"
    FIELD = "Field"
    RITUAL = "Ritual"


class TrapType(str, Enum):
    NORMAL = "Normal"
    CONTINUOUS = "Continuous"
    COUNTER = "Counter"


class FineCategory(str, Enum):
    NORMAL_MONSTER = "Normal Monster"
    EFFECT_MONSTER = "Effect Monster"
    FUSION_MONSTER = "Fusion Monster"
    RITUAL_MONSTER = "Ritual Monster"
    SYNCHRO_MONSTER = "Synchro Monster"
    XYZ_MONSTER = "Xyz Monster"
    LINK_MONSTER = "Link Monster"
    TOON_MONSTER = "Toon Monster"
    SPIRIT_MONSTER = "Spirit Monster"
    UNION_MONSTER = "Union Monster"
    GEMINI_MONSTER = "Gemini Monster"
    SPELL_NORMAL = "Spell-Normal"
    SPELL_CONTINUOUS = "Spell-Continuous"
    SPELL_EQUIP = "Spell-Equip"
    SPELL_QUICK_PLAY = "Spell-Quick-Play"
    SPELL_FIELD = "Spell-Field"
    SPELL_RITUAL = "Spell-Ritual"
    TRAP_NORMAL = "Trap-Normal"
    TRAP_CONTINUOUS = "Trap-Continuous"
    TRAP_COUNTER = "Trap-Counter"
