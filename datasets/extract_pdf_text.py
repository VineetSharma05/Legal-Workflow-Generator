"""
Temporary utility: dump the text of every PDF in a domain folder to a matching
.txt file, so the provisions can be read/structured into the domain's
complete_*.json without reopening each PDF by hand.

Uses pdfplumber (already a project dependency) rather than a new OCR stack —
the gazette PDFs here carry a real text layer. Scanned pages come out empty,
so the summary flags any file whose text layer is thin enough to need OCR.

    python datasets/extract_pdf_text.py                  # datasets/dpdp -> datasets/extracted_text/dpdp
    python datasets/extract_pdf_text.py --src datasets/tax --out datasets/extracted_text/tax
    python datasets/extract_pdf_text.py --force          # re-extract files already done
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pdfplumber

# Page markers are kept by default: provisions in these gazette PDFs are easiest
# to locate by page while transcribing, and the marker is trivial to strip later.
PAGE_MARKER = "\n\n===== [page {n}] =====\n\n"

# Below this many characters per page a PDF is almost certainly scanned images
# with no usable text layer, and needs OCR instead of pdfplumber.
MIN_CHARS_PER_PAGE = 100

# How much of each extraction to echo, so opaquely-named sources
# (250880.pdf, in212en_1.pdf) can be identified without opening them.
PREVIEW_CHARS = 220


def slugify(stem: str) -> str:
    """Turn a gazette filename into a readable, predictable output name."""
    s = stem.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_") or "untitled"


def tidy(text: str) -> str:
    """Strip trailing whitespace per line and collapse runs of blank lines."""
    lines = [ln.rstrip() for ln in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def extract(pdf_path: Path, page_markers: bool) -> tuple[str, int, list[int]]:
    """Return (text, page_count, 1-indexed page numbers that came out empty)."""
    parts: list[str] = []
    empty_pages: list[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                empty_pages.append(i)
            if page_markers:
                parts.append(PAGE_MARKER.format(n=i))
            parts.append(page_text)
        page_count = len(pdf.pages)

    return tidy("".join(parts)), page_count, empty_pages


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="datasets/dpdp", help="folder holding the source PDFs")
    ap.add_argument("--out", default=None, help="output folder (default: datasets/extracted_text/<src name>)")
    ap.add_argument("--force", action="store_true", help="overwrite .txt files that already exist")
    ap.add_argument("--no-page-markers", action="store_true", help="omit the '===== [page N] =====' separators")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = repo_root / src
    out = Path(args.out) if args.out else repo_root / "datasets" / "extracted_text" / src.name
    if not out.is_absolute():
        out = repo_root / out

    if not src.is_dir():
        print(f"error: source folder not found: {src}", file=sys.stderr)
        return 1

    pdfs = sorted(p for p in src.iterdir() if p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"error: no PDFs in {src}", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(pdfs)} PDF(s): {src}  ->  {out}\n")

    manifest = []
    needs_ocr = []

    for pdf_path in pdfs:
        dest = out / f"{slugify(pdf_path.stem)}.txt"

        if dest.exists() and not args.force:
            print(f"  skip   {dest.name}  (exists; use --force to redo)")
            continue

        try:
            text, pages, empty_pages = extract(pdf_path, page_markers=not args.no_page_markers)
        except Exception as e:
            print(f"  FAIL   {pdf_path.name}: {type(e).__name__}: {e}", file=sys.stderr)
            manifest.append({"source": pdf_path.name, "error": f"{type(e).__name__}: {e}"})
            continue

        dest.write_text(text, encoding="utf-8")

        chars = len(text)
        per_page = chars / pages if pages else 0
        if per_page < MIN_CHARS_PER_PAGE:
            needs_ocr.append(dest.name)

        manifest.append({
            "source": pdf_path.name,
            "output": dest.name,
            "pages": pages,
            "chars": chars,
            "empty_pages": empty_pages,
        })

        flag = "  [thin text layer - may need OCR]" if per_page < MIN_CHARS_PER_PAGE else ""
        print(f"  ok     {dest.name}  ({pages} pages, {chars:,} chars{flag})")
        if empty_pages:
            print(f"         empty pages: {empty_pages}")

        # First words of the document, to identify sources by content not filename.
        preview = " ".join(re.sub(r"=====.*?=====", " ", text).split())[:PREVIEW_CHARS]
        print(f'         "{preview}..."')

    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    done = [m for m in manifest if "output" in m]
    print(f"\nwrote {len(done)} file(s) + _manifest.json to {out}")
    if needs_ocr:
        print(f"check for OCR: {', '.join(needs_ocr)}")
    failed = [m["source"] for m in manifest if "error" in m]
    if failed:
        print(f"failed: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
