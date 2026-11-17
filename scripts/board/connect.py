"""Ouvre une console série sur la carte Alif Ensemble E7 AppKit.

Carte branchée sur son **port PRG USB**. Ce port expose deux UART qu'il ne
faut pas confondre (piège n° 6 de ``CLAUDE.md``) :

- la **console applicative**, 115 200 bauds [POSÉ, REQ-113], où sortent les
  temps d'inférence (REQ-023) — c'est le défaut de ce script ;
- l'**UART du Secure Enclave**, 55 000 bauds [SOURCÉ], que SETOOLS utilise
  pour programmer la MRAM. Accessible avec ``--se``, et à éviter pendant un
  flash.

Usage :
    python scripts/board/connect.py --list          # lister les ports
    python scripts/board/connect.py                 # console applicative
    python scripts/board/connect.py --log           # idem, en journalisant
    python scripts/board/connect.py --se            # UART Secure Enclave
    python scripts/board/connect.py --setools       # SETOOLS voit-il un E7 ?

Ce qu'on tape est envoyé **ligne par ligne** (entrée pour valider).
Ctrl-C pour quitter.

Prérequis : ``pip install -e ".[board]"``. C'est volontairement léger — ni
torch ni AIDGE ne sont nécessaires pour lire un port série.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

try:
    from mountain import board, paths
except ModuleNotFoundError as exc:  # pragma: no cover
    if exc.name != "mountain":
        raise
    # Ce script est souvent le premier lancé sur une machine neuve, branchée à
    # la carte. Un ModuleNotFoundError nu n'indique pas quoi faire.
    sys.exit(
        "Le paquet `mountain` n'est pas installé.\n"
        "\n"
        "  cd <racine du dépôt>\n"
        '  pip install -e ".[board]"      # minimum pour la console série\n'
        "\n"
        "Ou `make install` pour la chaîne complète (entraînement, export AIDGE)."
    )


def cmd_list() -> int:
    """Liste les ports candidats et sort."""
    ports = board.list_candidates()
    if not ports:
        print("Aucun port série trouvé.")
        print("Vérifier que la carte est branchée sur son port PRG USB et sous tension.")
        return 1

    print(f"{len(ports)} port(s) candidat(s) :")
    for p in ports:
        print(f"  {p}")

    print()
    print(f"Console applicative : {board.console_baud()} bauds  [POSÉ, à confirmer]")
    print(f"Secure Enclave      : {board.se_baud()} bauds  (SETOOLS, pas la console)")

    if len(ports) > 1:
        print()
        print("Plusieurs ports : le script ne devine pas lequel est la console.")
        print("Les deux interfaces CDC du même composite USB sont indiscernables")
        print("par leur nom — essayer l'un puis l'autre avec --port.")
    return 0


def cmd_setools() -> int:
    """Vérifie que SETOOLS voit bien un E7, via sa base de configuration."""
    setools = os.environ.get("SETOOLS")
    if not setools:
        print("SETOOLS n'est pas défini.", file=sys.stderr)
        print("  export SETOOLS=/chemin/vers/app-release-exec-linux", file=sys.stderr)
        return 1

    db = Path(setools) / "utils" / "global-cfg.db"
    if not db.exists():
        print(f"Base de configuration absente : {db}", file=sys.stderr)
        print("SETOOLS n'a pas encore été configuré depuis la carte :", file=sys.stderr)
        print(f"  (cd $SETOOLS && ./tools-config -a -b {board.se_baud()} "
              f"-c {board.config()['serial']['se_device']})", file=sys.stderr)
        return 1

    content = db.read_text(errors="replace")
    expected = board.part_number()
    print(content.strip())
    print()
    if expected in content:
        print(f"OK — SETOOLS voit bien un {expected}.")
        return 0
    print(f"ATTENTION : {expected} absent de la configuration.", file=sys.stderr)
    print("La carte connectée n'est peut-être pas l'E7 attendu.", file=sys.stderr)
    return 1


def resolve_port(requested: str | None) -> str | None:
    """Port à ouvrir : celui demandé, ou le seul candidat non ambigu."""
    if requested:
        return requested

    default = board.pick_default()
    if default is not None:
        print(f"Port : {default}")
        return default.device

    ports = board.list_candidates()
    if not ports:
        print("Aucun port série trouvé. Carte branchée sur le port PRG USB ?",
              file=sys.stderr)
    else:
        print("Plusieurs ports candidats — préciser lequel avec --port :", file=sys.stderr)
        for p in ports:
            print(f"  --port {p.device}", file=sys.stderr)
    return None


def resolve_baud(args, cfg) -> int | None:
    """Débit à utiliser, en refusant les valeurs qui ne mènent nulle part."""
    if args.se:
        return board.se_baud()
    if args.baud is None:
        return board.console_baud()

    allowed = cfg["serial"]["allowed_bauds"]
    if args.baud not in allowed and not args.force:
        print(f"Débit inhabituel : {args.baud}. Attendus : {allowed}.", file=sys.stderr)
        print("Un mauvais débit ne produit pas d'erreur, juste du binaire illisible.",
              file=sys.stderr)
        print("Utiliser --force pour l'imposer malgré tout.", file=sys.stderr)
        return None
    return args.baud


def console(port: str, baud: int, log_path: Path | None, read_only: bool) -> int:
    """Boucle de console : série -> stdout (et journal), stdin -> série."""
    try:
        import serial
    except ImportError:
        print("pyserial est absent. Installer avec :", file=sys.stderr)
        print('  pip install -e ".[board]"', file=sys.stderr)
        return 1

    try:
        link = serial.Serial(port, baud, timeout=0.2)
    except serial.SerialException as exc:
        print(f"Impossible d'ouvrir {port} à {baud} bauds : {exc}", file=sys.stderr)
        print("Port déjà occupé par un autre terminal, ou droits insuffisants ?",
              file=sys.stderr)
        return 1

    log = log_path.open("a", encoding="utf-8") if log_path else None
    if log:
        log.write(f"\n--- {datetime.now().isoformat(timespec='seconds')} "
                  f"{port} @ {baud} ---\n")
        log.flush()
        print(f"Journal : {log_path}")

    print(f"Connecté à {port} @ {baud} bauds. Ctrl-C pour quitter.")
    if read_only:
        print("(lecture seule)")
    print("-" * 60)

    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                chunk = link.read(4096)
            except Exception:
                break
            if not chunk:
                continue
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            if log:
                log.write(text)
                log.flush()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    try:
        if read_only:
            while thread.is_alive():
                time.sleep(0.2)
        else:
            # Saisie ligne par ligne : suffisant pour une console de traces, et
            # évite de basculer le terminal en mode brut.
            for line in sys.stdin:
                link.write(line.encode())
                link.flush()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        thread.join(timeout=1.0)
        link.close()
        if log:
            log.close()

    print("\nDéconnecté.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Console série sur la carte Alif Ensemble E7 AppKit.",
        epilog="Les 55 000 bauds sont l'UART du Secure Enclave, pas la console "
               "applicative. Voir CLAUDE.md, piège n° 6.",
    )
    parser.add_argument("--list", action="store_true",
                        help="lister les ports candidats et sortir")
    parser.add_argument("--setools", action="store_true",
                        help="vérifier que SETOOLS voit un %s" % board.part_number())
    parser.add_argument("--port", help="port série à ouvrir (sinon auto si non ambigu)")
    parser.add_argument("--baud", type=int,
                        help="débit (défaut : console applicative)")
    parser.add_argument("--se", action="store_true",
                        help="UART du Secure Enclave — à éviter pendant un flash")
    parser.add_argument("--log", action="store_true",
                        help="journaliser la session dans artifacts/")
    parser.add_argument("--read-only", action="store_true",
                        help="ne rien envoyer à la carte")
    parser.add_argument("--force", action="store_true",
                        help="autoriser un débit hors de allowed_bauds")
    args = parser.parse_args()

    if args.list:
        return cmd_list()
    if args.setools:
        return cmd_setools()

    cfg = board.config()

    baud = resolve_baud(args, cfg)
    if baud is None:
        return 2
    if args.se:
        # Sur stderr, pour que l'avertissement reste lisible même quand la
        # sortie de la console est redirigée vers un fichier.
        print(f"ATTENTION : UART du Secure Enclave ({baud} bauds). Ce n'est pas la\n"
              "console applicative, et parler dessus pendant un flash peut le faire\n"
              "échouer.", file=sys.stderr, flush=True)

    port = resolve_port(args.port)
    if port is None:
        return 2

    log_path = None
    if args.log:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = paths.artifact(f"board-{stamp}.log")

    return console(port, baud, log_path, args.read_only)


if __name__ == "__main__":
    raise SystemExit(main())
