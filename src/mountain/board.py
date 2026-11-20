"""Accès à la carte Alif Ensemble E7 AppKit.

Découverte des ports série et débits de la carte. Volontairement petit : la
seule chose non triviale ici est de **ne pas confondre les deux UART** que le
port PRG USB expose.

- **console applicative**, 115 200 bauds [POSÉ, REQ-113] — c'est là que
  sortent les temps d'inférence (REQ-023) ;
- **UART du Secure Enclave**, 55 000 bauds [SOURCÉ] — celle que SETOOLS
  utilise pour programmer la MRAM.

S'y tromper ne produit pas une erreur franche mais du binaire illisible, ou un
flash qui échoue parce qu'on parlait sur le lien pendant la programmation.
C'est le piège n° 6 de ``CLAUDE.md``.

Les valeurs viennent de ``configs/board.yaml``.
"""

from __future__ import annotations

import fnmatch
import glob
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from . import paths

try:  # pyserial est une dépendance optionnelle : pip install -e ".[board]"
    from serial.tools import list_ports as _list_ports
except ImportError:  # pragma: no cover
    _list_ports = None


@lru_cache(maxsize=None)
def config() -> dict[str, Any]:
    """Contenu de ``configs/board.yaml``."""
    import yaml

    with open(paths.BOARD_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def console_baud() -> int:
    """Débit de la console applicative. [POSÉ] — à confirmer sur la carte."""
    return int(config()["serial"]["console_baud"])


def se_baud() -> int:
    """Débit de l'UART du Secure Enclave. Pas la console applicative."""
    return int(config()["serial"]["se_baud"])


def part_number() -> str:
    """Référence attendue du composant, p. ex. ``AE722F80F55D5``."""
    return str(config()["part_number"])


@dataclass(frozen=True)
class Port:
    """Un port série candidat."""

    device: str
    description: str = ""
    #: ``True`` si le nom correspond au périphérique Secure Enclave déclaré.
    is_se: bool = False

    def __str__(self) -> str:
        role = " [Secure Enclave]" if self.is_se else ""
        desc = f" — {self.description}" if self.description else ""
        return f"{self.device}{role}{desc}"


def _platform() -> str:
    return "darwin" if sys.platform == "darwin" else "linux"


def _is_excluded(device: str) -> bool:
    """Vrai si le port correspond à un motif d'exclusion de la plateforme.

    macOS expose en permanence des pseudo-ports (Bluetooth, console de debug)
    qui ne seront jamais la carte, et un nœud ``/dev/tty.*`` par port dont
    l'ouverture bloque en attente de DCD. Les écarter est ce qui permet à
    ``pick_default()`` de trancher au lieu de déclarer le choix ambigu.
    """
    patterns = config()["serial"].get("exclude_globs", {}).get(_platform(), [])
    return any(fnmatch.fnmatch(device, pattern) for pattern in patterns)


def list_candidates() -> list[Port]:
    """Ports série pouvant être la carte, dans l'ordre où les proposer.

    Deux sources, fusionnées : l'énumération USB de pyserial, qui donne une
    description lisible, et les motifs de ``configs/board.yaml``, qui rattrapent
    les nœuds créés par une règle udev (``/dev/alif-se``) que pyserial
    n'énumère pas toujours. Les deux passent par le filtre d'exclusion.
    """
    se_device = config()["serial"]["se_device"]
    found: dict[str, Port] = {}

    if _list_ports is not None:
        for p in _list_ports.comports():
            if _is_excluded(p.device):
                continue
            found[p.device] = Port(
                device=p.device,
                description=(p.description or "").strip(),
                is_se=(p.device == se_device),
            )

    for pattern in config()["serial"]["port_globs"].get(_platform(), []):
        for device in sorted(glob.glob(pattern)):
            if _is_excluded(device):
                continue
            found.setdefault(device, Port(device=device, is_se=(device == se_device)))

    # Le Secure Enclave en tête : c'est le seul qu'on sait identifier avec
    # certitude, donc le seul qu'on peut proposer d'écarter sciemment.
    return sorted(found.values(), key=lambda p: (not p.is_se, p.device))


def pick_default() -> Port | None:
    """Port à utiliser faute de ``--port``, ou ``None`` si le choix est ambigu.

    Ne devine jamais entre plusieurs ports non identifiés : deux interfaces CDC
    du même composite USB sont indiscernables par leur nom, et ouvrir la
    mauvaise ne donne aucune erreur claire. Mieux vaut demander.
    """
    candidates = [p for p in list_candidates() if not p.is_se]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        remaining = list_candidates()
        return remaining[0] if len(remaining) == 1 else None
    return None
