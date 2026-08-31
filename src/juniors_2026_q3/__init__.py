# src/juniors_2026_q3/__init__.py

import argparse


def main():
    parser = argparse.ArgumentParser(prog="juniors-2026-q3")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("train", help="Entrena el modelo y genera artefacto + model card")
    subparsers.add_parser("serve", help="Levanta la API de inferencia")  # pendiente de implementar

    args = parser.parse_args()

    if args.command == "train":
        from juniors_2026_q3.models.train import main as train_main
        train_main()
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("juniors_2026_q3.api.main:app", host="127.0.0.1", port=8000)