"""
Utility: pindahkan draft ground truth yang sudah disetujui ke evaluation_dataset.json.

Hanya memproses entry dengan "approved": true di ground_truth_drafts.json.
Entry dengan "approved": false dilewati — tidak ada perubahan ke dataset.

Usage:
    source venv/bin/activate
    python -m app.apply_ground_truth
"""
from __future__ import annotations

import json
from pathlib import Path

DATASET_PATH: Path = Path("app/data/evaluation_dataset.json")
DRAFTS_PATH: Path  = Path("app/data/ground_truth_drafts.json")


def main() -> None:
    """Salin ground truth yang sudah diapprove ke evaluation_dataset.json."""
    if not DRAFTS_PATH.exists():
        print(f"File tidak ditemukan: {DRAFTS_PATH}")
        print("Jalankan dulu: python -m app.create_ground_truth")
        return

    dataset: list[dict] = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    drafts:  list[dict] = json.loads(DRAFTS_PATH.read_text(encoding="utf-8"))

    # Kumpulkan semua entry yang sudah approved dan punya draft non-kosong
    approved: dict[str, str] = {
        d["question_id"]: d["draft_ground_truth"]
        for d in drafts
        if d.get("approved") is True
        and d.get("draft_ground_truth")
        and not d["draft_ground_truth"].startswith("ERROR:")
    }

    if not approved:
        print("Belum ada entry dengan 'approved': true di ground_truth_drafts.json.")
        print("Buka file tersebut, koreksi draft, lalu set 'approved': true.")
        return

    updated = 0
    skipped_existing = 0
    for item in dataset:
        qid = item["question_id"]
        if qid not in approved:
            continue
        if item.get("ground_truth") is not None:
            # Sudah ada ground truth sebelumnya — tidak ditimpa
            skipped_existing += 1
            continue
        item["ground_truth"] = approved[qid]
        updated += 1

    DATASET_PATH.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_null     = sum(1 for item in dataset if item.get("ground_truth") is None)
    n_filled   = sum(1 for item in dataset if item.get("ground_truth") is not None)
    print(f"Selesai:")
    print(f"  Diupdate  : {updated} entry")
    print(f"  Dilewati (sudah ada): {skipped_existing} entry")
    print(f"  Filled ground_truth : {n_filled}/50")
    print(f"  Masih null          : {n_null}/50")


if __name__ == "__main__":
    main()
