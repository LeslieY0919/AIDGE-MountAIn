"""Ouvre une console série sur la carte Alif Ensemble E7 AppKit.

Carte branchée sur son **port PRG USB**. Ce port expose deux UART qu'il ne
faut pas confondre (piège n° 6 de ``CLAUDE.md``) :

- la **console applicative**, où sortent les temps d'inférence (REQ-023) —
  c'est le défaut de ce script. Son débit est dans ``configs/board.yaml`` ;
  ``--scan`` sert à le (re)mesurer, parce qu'un mauvais débit ne lève aucune
  erreur et ne sort que du binaire ;
- l'**UART du Secure Enclave**, 55 000 bauds [SOURCÉ], que SETOOLS utilise
  pour programmer la MRAM. Accessible avec ``--se``, et à éviter pendant un
  flash.

Usage :
    python scripts/board/connect.py --list          # lister les ports
    python scripts/board/connect.py                 # console applicative
    python scripts/board/connect.py --log           # idem, en journalisant
    python scripts/board/connect.py --se            # UART Secure Enclave
    python scripts/board/connect.py --scan          # quel débit décode ?
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


def cmd_scan(port: str, seconds: float) -> int:
    """Essaie plusieurs débits et dit lequel décode proprement.

    Un mauvais débit ne lève aucune erreur : on reçoit du binaire. La seule
    façon de trancher est d'écouter et de compter les octets lisibles.

    Attention à la tolérance de l'UART : un écart de quelques pourcents décode
    encore, parce que chaque caractère se resynchronise sur son bit de start.
    Plusieurs débits voisins peuvent donc « marcher ». Le vrai débit est au
    **centre** de la fenêtre qui décode, pas à son bord — c'est ce que la
    dernière ligne du rapport estime.
    """
    try:
        import serial
    except ImportError:
        print('pyserial est absent : pip install -e ".[board]"', file=sys.stderr)
        return 1

    bauds = board.config()["serial"].get("scan_bauds") or [board.console_baud()]
    print(f"Balayage de {port}, {seconds:g} s par débit.")
    print()
    print(f"  {'débit':>7}  {'octets':>6}  {'lisible':>7}  échantillon")
    print(f"  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*40}")

    results: list[tuple[int, float, int]] = []
    for baud in bauds:
        try:
            with serial.Serial(port, baud, timeout=0.3) as link:
                link.reset_input_buffer()
                buf = b""
                deadline = time.time() + seconds
                while time.time() < deadline:
                    buf += link.read(4096)
        except Exception as exc:
            print(f"  {baud:>7}  erreur : {exc}")
            continue

        if not buf:
            print(f"  {baud:>7}  {0:>6}  {'—':>7}  (rien reçu)")
            results.append((baud, 0.0, 0))
            continue

        # ESC et backspace comptent comme lisibles : une console qui s'anime en
        # émet légitimement.
        readable = sum(1 for c in buf if 32 <= c < 127 or c in (8, 9, 10, 13, 27))
        ratio = readable / len(buf)
        sample = "".join(chr(c) if 32 <= c < 127 else "." for c in buf[:40])
        print(f"  {baud:>7}  {len(buf):>6}  {ratio*100:>6.1f}%  |{sample}|")
        results.append((baud, ratio, len(buf)))

    good = [b for b, ratio, n in results if n > 0 and ratio >= 0.95]
    print()
    if not good:
        print("Aucun débit ne décode proprement.")
        print("La carte émet-elle quelque chose ? Essayer --scan --port <autre port>.")
        return 1

    print(f"Décodent proprement : {', '.join(str(b) for b in sorted(good))}")
    if len(good) > 1:
        centre = (min(good) + max(good)) / 2
        print(f"Fenêtre {min(good)}–{max(good)} : le vrai débit est proche de "
              f"{centre:.0f}, pas d'un bord.")
        print("Resserrer avec --scan autour de cette valeur pour trancher.")
    print()
    print(f"Pour fixer le choix : console_baud dans {paths.BOARD_CONFIG.name}.")
    return 0


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
    parser.add_argument("--scan", action="store_true",
                        help="essayer plusieurs débits et dire lequel décode")
    parser.add_argument("--scan-seconds", type=float, default=3.0,
                        help="durée d'écoute par débit pour --scan (défaut : 3)")
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
    if args.scan:
        port = resolve_port(args.port)
        if port is None:
            return 2
        return cmd_scan(port, args.scan_seconds)

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
