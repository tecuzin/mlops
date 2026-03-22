#!/usr/bin/env python3
"""
Vérifie les résultats des entraînements et reprend les runs interrompus.

Usage:
  API_URL=http://localhost:8000 python scripts/check_and_resume_runs.py [--reset-stale] [--reset-failed]

Sans options: affiche le statut des runs.
--reset-stale: remet en 'pending' les runs bloqués (training/evaluating/security_scanning > 5 min sans update)
--reset-failed: remet en 'pending' les runs en 'failed' pour retry
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone, timedelta

API_URL = os.getenv("API_URL", "http://localhost:8000")
STALE_MINUTES = 5  # runs in training/eval/security sans update depuis N min = considérés orphelins


def main() -> None:
    parser = argparse.ArgumentParser(description="Vérifier et reprendre les runs interrompus")
    parser.add_argument("--reset-stale", action="store_true", help="Remettre en pending les runs orphelins")
    parser.add_argument("--reset-failed", action="store_true", help="Remettre en pending les runs failed")
    parser.add_argument("--dry-run", action="store_true", help="Afficher sans modifier")
    args = parser.parse_args()

    try:
        req = urllib.request.Request(f"{API_URL}/runs")
        with urllib.request.urlopen(req, timeout=30) as resp:
            runs = json.load(resp)
    except urllib.error.URLError as e:
        print(f"Erreur: impossible de joindre l'API ({API_URL}). Démarrez la stack Docker.", file=sys.stderr)
        sys.exit(1)
    now = datetime.now(timezone.utc)
    stale_threshold = now - timedelta(minutes=STALE_MINUTES)
    in_progress_statuses = ("training", "evaluating", "security_scanning")

    print("=== Statut des runs ===\n")
    for r in sorted(runs, key=lambda x: -x["id"]):
        rid = r["id"]
        name = r["model_name"]
        task = r["task_type"]
        status = r["status"]
        updated = r.get("updated_at")
        results = r.get("results") or []

        # Parse updated_at pour vérifier si stale
        updated_dt = None
        if updated:
            try:
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                pass

        is_stale = (
            status in in_progress_statuses
            and updated_dt is not None
            and updated_dt < stale_threshold
        )

        # Résumé des résultats
        res_str = ""
        if results:
            res_str = " | ".join(f"{m['metric_name']}={m['metric_value']:.3f}" for m in results[:5])
            if len(results) > 5:
                res_str += f" (+{len(results)-5})"

        status_emoji = {
            "pending": "⏳",
            "training": "🏋️",
            "evaluating": "📊",
            "security_scanning": "🔒",
            "completed": "✅",
            "failed": "❌",
        }.get(status, "?")

        flags = []
        if is_stale:
            flags.append("(orphelin)")
        if r.get("error_message"):
            flags.append("error")

        flag_str = " " + " ".join(flags) if flags else ""

        print(f"  {rid:3} {status_emoji} {name:22} {task:15} {status:18} {res_str}{flag_str}")

    # Actions
    to_reset = []
    if args.reset_stale:
        for r in runs:
            if r["status"] not in in_progress_statuses:
                continue
            updated = r.get("updated_at")
            if not updated:
                continue
            try:
                updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if updated_dt.tzinfo is None:
                    updated_dt = updated_dt.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue
            if updated_dt < stale_threshold:
                to_reset.append((r["id"], r["model_name"], r["status"], "orphelin"))

    if args.reset_failed:
        for r in runs:
            if r["status"] == "failed":
                to_reset.append((r["id"], r["model_name"], "failed", "failed"))

    if to_reset:
        print("\n=== Runs à remettre en pending ===")
        for rid, name, old_status, reason in to_reset:
            print(f"  {rid}: {name} ({old_status}) — {reason}")
        if not args.dry_run:
            print("\nRemise en pending...")
            for rid, name, _, _ in to_reset:
                try:
                    req = urllib.request.Request(
                        f"{API_URL}/runs/{rid}/status?status=pending",
                        method="PATCH",
                    )
                    urllib.request.urlopen(req, timeout=10)
                    print(f"  ✓ Run {rid} → pending")
                except Exception as e:
                    print(f"  ✗ Run {rid}: {e}", file=sys.stderr)
        else:
            print("\n(Dry-run: aucune modification)")
    elif args.reset_stale or args.reset_failed:
        print("\nAucun run à réinitialiser.")

    if not args.reset_stale and not args.reset_failed:
        print("\nOptions:")
        print("  --reset-stale   : reprendre les runs bloqués (training/evaluating/security > 5 min)")
        print("  --reset-failed  : retenter les runs en échec")


if __name__ == "__main__":
    main()
