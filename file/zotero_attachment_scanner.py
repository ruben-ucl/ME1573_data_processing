import re
from pathlib import Path

try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader

storage_dir = Path(r"C:\Users\rlamb\Zotero\storage")
SIZE_THRESHOLD = 10 * 1e6  # 10 MB
DOI_RE = re.compile(r'\b10\.\d{4,}/\S+')
NONDESCRIPTIVE_RE = re.compile(r'^[A-Za-z0-9_\-]+$')  # no spaces = no title words


def is_nondescriptive(stem):
    return bool(NONDESCRIPTIVE_RE.match(stem))


def extract_doi(pdf_path, max_pages=3):
    try:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            m = DOI_RE.search(text)
            if m:
                return m.group(0).rstrip('.,)')
    except Exception:
        pass
    return None


pdfs = [(p, p.stat().st_size) for p in storage_dir.rglob("*.pdf")]
pdfs.sort(key=lambda x: x[1], reverse=True)

print(f"Found {len(pdfs)} PDFs in {storage_dir}\n")
print(f"{'Size (MB)':>10}  {'DOI':<45}  {'Filename'}")
print("-" * 120)
for path, size in pdfs:
    stem = path.stem
    if size > SIZE_THRESHOLD and is_nondescriptive(stem):
        doi = extract_doi(path) or "DOI not found"
    else:
        doi = ""
    print(f"{size / 1e6:>10.2f}  {doi:<45}  {path.name}")
