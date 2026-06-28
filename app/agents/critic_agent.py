from __future__ import annotations
"""Critic Agent — LLM-based validation untuk H2 + H4.

Menggunakan Llama-3.3-70B-Versatile (Groq) dengan structured JSON output.
Output schema FROZEN (lihat system prompt §16, Keputusan #1).

Mapping:
- H2 (Fabricated Financial Metric): metrik finansial yang tidak ada di evidence
- H4 (Incorrect Inference): kesimpulan tidak konsisten secara logis dengan evidence

Catatan: Critic juga memeriksa H1 + H3 sebagai cross-check, namun deteksi
otoritatif untuk H1 + H3 ada di Domain Guardrails (deterministik).

Prompt versions:
  CRITIC_PROMPT_V1 — dipakai eksperimen 50Q (data final, JANGAN dihapus).
  CRITIC_PROMPT_V2 — nonaktif: {fundamental_context} + aturan H2 terlalu luas →
                     FP masif (H2=49/50 di eksperimen V3).
  CRITIC_PROMPT_V3 — nonaktif: CoT 5-langkah + "overall_verdict: pending" →
                     masih FP tinggi (H2=39/44 ev>0 di eksperimen V5) meskipun
                     B1 clarification sudah ditambahkan.
  CRITIC_PROMPT_V4 — aktif (D1): basis V1 + dua tambahan minimal:
                     (a) H2: flag=true hanya jika klaim nilai angka, bukan sekadar
                         menyebut nama metrik atau mengakui ketidaktersediaan.
                     (b) H4: jika evidence kosong/tidak relevan, H4=false.
"""
import json
import re
from typing import Any, Literal

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from app.config import GROQ_CRITIC_MODEL
from app.services.llm_service import build_critic_llm

CRITIC_TEMPERATURE: float = 0.0
PROMPT_VERSION: str = "critic_v4"


class CategoryFlag(BaseModel):
    """Satu kategori halusinasi: flag boolean + rationale string (Bahasa Indonesia)."""
    flag: bool = Field(description="True jika kategori halusinasi terdeteksi")
    rationale: str = Field(description="Penjelasan singkat dalam Bahasa Indonesia")


class CriticVerdict(BaseModel):
    """Output struktur frozen Critic Agent (system prompt §16).

    Catatan: `model` dan `temperature` di-fill dari config server-side, bukan
    dari LLM, untuk audit trail yang reliable.
    """
    H1_unsupported_numeric: CategoryFlag
    H2_fabricated_metric: CategoryFlag
    H3_stale_timestamp: CategoryFlag
    H4_incorrect_inference: CategoryFlag
    overall_verdict: Literal["pass", "fail"]
    model: str = GROQ_CRITIC_MODEL
    temperature: float = CRITIC_TEMPERATURE


# ── Prompt V1 — JANGAN dihapus; dipakai eksperimen 50Q (data sudah final) ─────
CRITIC_PROMPT_V1 = """Anda adalah Critic Agent yang memvalidasi jawaban tentang pasar saham IDX30.

Tugas: periksa apakah JAWABAN konsisten dengan EVIDENCE, dan flag 4 kategori halusinasi.

KATEGORI HALUSINASI:
- H1 (Unsupported Numeric Claim): angka di jawaban tidak ada di evidence.
- H2 (Fabricated Financial Metric): metrik finansial (PER, PBV, ROE, EPS, dll) yang disebut di jawaban tetapi tidak muncul di evidence.
- H3 (Stale Timestamp Misrepresentation): klaim numerik disajikan tanpa konteks waktu yang sesuai.
- H4 (Incorrect Inference): kesimpulan tidak konsisten secara logis dengan evidence (contoh: evidence menunjukkan harga turun, jawaban bilang naik).

EVIDENCE:
{evidence}

PERTANYAAN:
{question}

JAWABAN UNTUK DIVALIDASI:
{answer}

INSTRUKSI OUTPUT:
Balas HANYA dengan JSON valid (tanpa preamble, tanpa markdown fence). Schema:
{{
  "H1_unsupported_numeric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H2_fabricated_metric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H3_stale_timestamp": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H4_incorrect_inference": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "overall_verdict": "pass"
}}

Aturan:
- rationale wajib non-empty walaupun flag = false (untuk audit trail).
- overall_verdict akan dihitung server-side, kamu bisa selalu menulis "pass".
"""

# ── Fundamental Metrics Knowledge — domain IDX30 untuk Critic ────────────────
# Dipisah sebagai konstanta agar mudah diperbarui tanpa menyentuh template prompt.
# Tidak mengandung karakter { atau } (aman untuk str.format()).
FUNDAMENTAL_METRICS_V1 = (
    "Metrik fundamental saham yang VALID di domain IDX30:\n"
    "  PER (Price-to-Earnings Ratio), PBV (Price-to-Book Value),\n"
    "  ROE (Return on Equity), EPS (Earnings per Share),\n"
    "  DER (Debt-to-Equity Ratio), NPM (Net Profit Margin),\n"
    "  Revenue/Pendapatan, Laba Bersih/Net Income,\n"
    "  Market Cap/Kapitalisasi Pasar, Dividend Yield, EBITDA,\n"
    "  Current Ratio, Operating Margin, Gross Margin.\n"
    "H2 flag=true HANYA jika salah satu metrik di atas disebutkan di JAWABAN "
    "tetapi TIDAK muncul di EVIDENCE manapun."
)

# ── Prompt V2 — NONAKTIF ──────────────────────────────────────────────────────
# Root cause kegagalan V2 (ditemukan dari eksperimen V3):
#   - {fundamental_context} menyebabkan aturan H2 "disebutkan di JAWABAN tapi
#     tidak di EVIDENCE" terlalu luas: frasa "ROE tidak tersedia" pun di-flag H2
#     karena "ROE" muncul di jawaban tapi tidak ada di evidence.
#   - Akibat: H2=49/50 dan H4=49/50 (false positive masif).
# JANGAN dihapus — dipertahankan sebagai catatan regresi.
# Perubahan vs V1:
#   1. {fundamental_context}  — pengetahuan domain metrik IDX30 (mitigasi H2)
#   2. CoT steps              — 4 langkah evaluasi sebelum output JSON (mitigasi H4)
#   3. H4 concrete example    — contoh flag=true dan flag=false untuk H4
#   4. Hapus bias "pass"      — schema contoh netral, overall_verdict="pending"
#   5. Evidence reference     — instruksi menyebut [evidence-N] di rationale
#   6. Bahasa formal          — explisit Bahasa Indonesia formal di rationale
CRITIC_PROMPT_V2 = """Anda adalah Critic Agent yang memvalidasi jawaban tentang pasar saham IDX30.
Tugas: periksa apakah JAWABAN konsisten dengan EVIDENCE, dan flag 4 kategori halusinasi.

PENGETAHUAN DOMAIN:
{fundamental_context}

KATEGORI HALUSINASI:
- H1 (Unsupported Numeric Claim): angka di JAWABAN (harga, volume, persentase, rasio)
  tidak dapat ditemukan di EVIDENCE manapun.
- H2 (Fabricated Financial Metric): metrik finansial disebutkan di JAWABAN tetapi
  tidak muncul di EVIDENCE. Lihat daftar metrik valid di PENGETAHUAN DOMAIN.
- H3 (Stale Timestamp Misrepresentation): JAWABAN mengandung klaim numerik tetapi
  tidak mencantumkan tanggal/periode data sama sekali.
- H4 (Incorrect Inference): kesimpulan di JAWABAN BERTENTANGAN dengan fakta di EVIDENCE.
  Contoh flag=true: EVIDENCE menyatakan "BBCA turun 3,2% kemarin", JAWABAN menyatakan
  "BBCA menunjukkan momentum bullish yang kuat" — kontradiksi arah.
  Contoh flag=false: EVIDENCE mendukung kesimpulan (arah dan magnitude konsisten).
  Catatan: inferensi yang tidak bisa dikonfirmasi dari evidence (bukan bertentangan)
  → flag=false, sebutkan dalam rationale sebagai keterbatasan evidence.

EVIDENCE:
{evidence}

PERTANYAAN:
{question}

JAWABAN UNTUK DIVALIDASI:
{answer}

LANGKAH EVALUASI (lakukan secara berurutan sebelum menulis JSON):
1. Catat semua angka di JAWABAN. Cek setiap angka — ada di [evidence-N] mana?
2. Catat semua metrik finansial di JAWABAN. Cek — ada di EVIDENCE atau tidak?
3. Periksa apakah JAWABAN mencantumkan tanggal/periode untuk klaim numerik.
4. Periksa apakah kesimpulan JAWABAN BERTENTANGAN (bukan sekadar tidak dikonfirmasi)
   dengan fakta di EVIDENCE.

INSTRUKSI OUTPUT:
Setelah evaluasi selesai, balas HANYA dengan JSON valid (tanpa preamble, tanpa markdown fence):
{{
  "H1_unsupported_numeric": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan angka bermasalah atau konfirmasi semua terlacak di [evidence-N]>"}},
  "H2_fabricated_metric": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan metrik yang tidak ada di evidence, atau konfirmasi semua ada>"}},
  "H3_stale_timestamp": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal>"}},
  "H4_incorrect_inference": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan kontradiksi spesifik jika ada>"}},
  "overall_verdict": "pending"
}}

Aturan:
- flag berdasarkan fakta: true hanya jika ada bukti halusinasi, false jika tidak ada.
- rationale WAJIB non-empty walaupun flag=false — jelaskan mengapa tidak ada masalah.
- Referensikan [evidence-N] spesifik dalam rationale bila relevan.
- overall_verdict selalu tulis "pending" — akan dihitung server-side.
- Bahasa rationale: Bahasa Indonesia formal akademik.
"""


# ── Prompt V3 — AKTIF (B1 fix) ────────────────────────────────────────────────
# Fix vs V2:
#   - Hapus {fundamental_context}: tidak perlu karena aturan H2 sudah diperjelas.
#   - Klarifikasi H2 (kunci): flag=true HANYA jika jawaban MENGKLAIM NILAI SPESIFIK
#     metrik yang tidak ada di evidence (contoh: "PER = 15x"). Jika jawaban
#     mengakui ketidaktersediaan metrik ("ROE tidak tersedia"), H2=false.
#   - Klarifikasi H4: jika evidence=0 (tidak ada), H4=false — tidak ada dasar
#     untuk menyatakan kesimpulan "bertentangan" dengan evidence yang tidak ada.
#   - CoT steps, H4 concrete example, "pending" verdict, evidence reference
#     dipertahankan dari V2.
CRITIC_PROMPT_V3 = """Anda adalah Critic Agent yang memvalidasi jawaban tentang pasar saham IDX30.
Tugas: periksa apakah JAWABAN konsisten dengan EVIDENCE, dan flag 4 kategori halusinasi.

KATEGORI HALUSINASI:
- H1 (Unsupported Numeric Claim): angka spesifik di JAWABAN (harga, volume, persentase,
  rasio) tidak dapat ditemukan di EVIDENCE manapun.
- H2 (Fabricated Financial Metric): JAWABAN MENGKLAIM NILAI SPESIFIK suatu metrik
  finansial (contoh: "PER = 15x", "ROE sebesar 18%") yang tidak muncul di EVIDENCE.
  PENTING — H2=false jika:
    (a) Jawaban mengakui metrik tidak tersedia ("ROE tidak tersedia", "EPS tidak ditemukan").
    (b) Metrik disebutkan hanya dalam konteks pertanyaan ulang atau klarifikasi.
  H2=true HANYA jika jawaban menyebut nilai angka untuk metrik tersebut tanpa dukungan evidence.
- H3 (Stale Timestamp Misrepresentation): JAWABAN mengandung klaim numerik tetapi
  tidak mencantumkan tanggal/periode data sama sekali.
- H4 (Incorrect Inference): kesimpulan di JAWABAN BERTENTANGAN secara langsung
  dengan fakta di EVIDENCE.
  Contoh flag=true: EVIDENCE menyatakan "BBCA turun 3,2%", JAWABAN menyatakan
  "BBCA naik signifikan" — kontradiksi arah yang jelas.
  Contoh flag=false: EVIDENCE mendukung kesimpulan, atau evidence kosong/tidak ada.
  PENTING — jika EVIDENCE kosong ("tidak ada evidence") atau EVIDENCE tidak membahas
  topik yang ditanyakan, H4=false karena tidak ada fakta yang bisa dikontradiksi.

EVIDENCE:
{evidence}

PERTANYAAN:
{question}

JAWABAN UNTUK DIVALIDASI:
{answer}

LANGKAH EVALUASI (lakukan secara berurutan sebelum menulis JSON):
1. Apakah EVIDENCE kosong atau tidak relevan? Jika ya → H1=false, H4=false (tidak ada basis perbandingan).
2. Catat semua angka di JAWABAN. Cek setiap angka — ada di [evidence-N] mana?
3. Catat metrik finansial di JAWABAN yang disertai nilai angka. Cek — ada di EVIDENCE?
   (Metrik yang disebut tanpa nilai, atau yang diakui tidak tersedia, TIDAK di-flag H2.)
4. Periksa apakah JAWABAN mencantumkan tanggal/periode untuk klaim numerik.
5. Periksa apakah ada pernyataan di JAWABAN yang secara langsung BERTENTANGAN dengan
   fakta eksplisit di EVIDENCE (bukan sekadar tidak dikonfirmasi).

INSTRUKSI OUTPUT:
Setelah evaluasi selesai, balas HANYA dengan JSON valid (tanpa preamble, tanpa markdown fence):
{{
  "H1_unsupported_numeric": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan angka bermasalah atau konfirmasi semua terlacak di [evidence-N]>"}},
  "H2_fabricated_metric": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan nilai metrik yang tidak ada di evidence, atau konfirmasi semua ada/diakui tidak tersedia>"}},
  "H3_stale_timestamp": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal>"}},
  "H4_incorrect_inference": {{"flag": <true|false>, "rationale": "<Bahasa Indonesia formal — sebutkan kontradiksi spesifik atau konfirmasi tidak ada kontradiksi>"}},
  "overall_verdict": "pending"
}}

Aturan:
- flag berdasarkan bukti: true hanya jika ada halusinasi yang terdokumentasi.
- rationale WAJIB non-empty walaupun flag=false — jelaskan mengapa tidak ada masalah.
- Referensikan [evidence-N] spesifik dalam rationale bila relevan.
- overall_verdict selalu tulis "pending" — akan dihitung server-side.
- Bahasa rationale: Bahasa Indonesia formal akademik.
"""

# ── Prompt V4 — AKTIF (D1) ────────────────────────────────────────────────────
# Basis: CRITIC_PROMPT_V1 (terbukti: H2=1/44, H4=4/44 di eksperimen 50Q V1).
# Tambahan minimal vs V1:
#   (a) Klarifikasi H2: flag=true HANYA jika jawaban menyebut NILAI ANGKA suatu
#       metrik finansial yang tidak ada di evidence. Sekadar menyebut nama metrik
#       atau mengakui ketidaktersediaan ("ROE tidak tersedia") → H2=false.
#   (b) Klarifikasi H4: jika evidence kosong atau tidak membahas topik pertanyaan,
#       H4=false — tidak ada fakta eksplisit yang bisa dikontradiksi.
# Tidak ada: CoT steps, {fundamental_context}, H4 examples, "pending" hint,
# evidence reference instruction — semua terbukti meningkatkan FP.
CRITIC_PROMPT_V4 = """Anda adalah Critic Agent yang memvalidasi jawaban tentang pasar saham IDX30.

Tugas: periksa apakah JAWABAN konsisten dengan EVIDENCE, dan flag 4 kategori halusinasi.

KATEGORI HALUSINASI:
- H1 (Unsupported Numeric Claim): angka di jawaban tidak ada di evidence.
- H2 (Fabricated Financial Metric): JAWABAN menyebut NILAI ANGKA suatu metrik finansial
  (contoh: "PER = 15x", "ROE sebesar 18%", "EPS Rp 500") yang tidak muncul di evidence.
  H2=false jika jawaban hanya menyebut nama metrik tanpa nilai, atau mengakui metrik
  tidak tersedia ("ROE tidak tersedia", "data EPS tidak ditemukan").
- H3 (Stale Timestamp Misrepresentation): klaim numerik disajikan tanpa konteks waktu
  yang sesuai.
- H4 (Incorrect Inference): kesimpulan tidak konsisten secara logis dengan evidence.
  H4=false jika evidence kosong atau tidak membahas topik pertanyaan — tidak ada fakta
  yang bisa dikontradiksi.

EVIDENCE:
{evidence}

PERTANYAAN:
{question}

JAWABAN UNTUK DIVALIDASI:
{answer}

INSTRUKSI OUTPUT:
Balas HANYA dengan JSON valid (tanpa preamble, tanpa markdown fence). Schema:
{{
  "H1_unsupported_numeric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H2_fabricated_metric": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H3_stale_timestamp": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "H4_incorrect_inference": {{"flag": <bool>, "rationale": "<id-string non-empty>"}},
  "overall_verdict": "pass"
}}

Aturan:
- rationale wajib non-empty walaupun flag = false (untuk audit trail).
- overall_verdict akan dihitung server-side, kamu bisa selalu menulis "pass".
"""


class CriticAgent:
    """LLM-based critic (Llama-3.3-70B) untuk H1/H2/H3/H4 validation.

    Args:
        llm: Optional ChatGroq instance. Jika None, dibuat via build_critic_llm().
             Parameter ini ada untuk dependency injection (testing dengan mock).
    """

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm or build_critic_llm(temperature=CRITIC_TEMPERATURE)
        self._prompt = PromptTemplate.from_template(CRITIC_PROMPT_V4)
        self.prompt_version: str = PROMPT_VERSION

    def validate(
        self,
        question: str,
        answer: str,
        evidence: list[dict],
    ) -> CriticVerdict:
        """Run critic LLM dan return parsed structured verdict.

        Args:
            question: pertanyaan asli pengguna.
            answer: jawaban yang akan divalidasi.
            evidence: list of dict dengan minimal key 'content'.

        Returns:
            CriticVerdict. Jika LLM error / JSON invalid, return fail-safe
            verdict (semua flag=true, overall_verdict="fail") supaya mode runner
            tidak crash.
        """
        evidence_blob = "\n\n".join(
            f"[evidence-{i}] {e.get('content', '')}" for i, e in enumerate(evidence)
        ) or "(tidak ada evidence)"

        prompt = self._prompt.format(
            evidence=evidence_blob,
            question=question,
            answer=answer,
        )

        try:
            raw = self._llm.invoke(prompt).content
            return self._parse(raw)
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            return self._failsafe_verdict(reason=f"parse_error: {type(exc).__name__}")
        except Exception as exc:
            return self._failsafe_verdict(reason=f"llm_error: {type(exc).__name__}")

    def _parse(self, raw: str) -> CriticVerdict:
        """Parse JSON output dari critic LLM. Strip markdown fence kalau ada."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        data = json.loads(cleaned)

        # Override overall_verdict dari flags (jangan trust nilai dari LLM)
        flags = [
            data.get("H1_unsupported_numeric", {}).get("flag", False),
            data.get("H2_fabricated_metric", {}).get("flag", False),
            data.get("H3_stale_timestamp", {}).get("flag", False),
            data.get("H4_incorrect_inference", {}).get("flag", False),
        ]
        data["overall_verdict"] = "fail" if any(flags) else "pass"

        return CriticVerdict(**data)

    def _failsafe_verdict(self, reason: str) -> CriticVerdict:
        """Fail-safe: kalau critic tidak bisa dijalankan, treat sebagai fail."""
        return CriticVerdict(
            H1_unsupported_numeric=CategoryFlag(flag=True, rationale=reason),
            H2_fabricated_metric=CategoryFlag(flag=True, rationale=reason),
            H3_stale_timestamp=CategoryFlag(flag=True, rationale=reason),
            H4_incorrect_inference=CategoryFlag(flag=True, rationale=reason),
            overall_verdict="fail",
        )
