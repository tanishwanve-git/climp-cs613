"""Markdown -> styled HTML -> PDF with headless Chrome (no pandoc/poppler needed).

  python tools/md_to_pdf.py report/report.md report/report.pdf
Relative image paths resolve from the markdown file's folder (the HTML is written next to it).
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10pt; line-height: 1.45; color: #1f1f1e; max-width: 920px; margin: 0 auto; }
h1 { font-size: 19pt; margin: 0 0 6pt; } h2 { font-size: 14pt; border-bottom: 1.5px solid #e6e5e1; padding-bottom: 3pt; margin-top: 16pt; }
h3 { font-size: 11.5pt; margin-top: 11pt; } table { border-collapse: collapse; font-size: 8.6pt; margin: 6pt 0; page-break-inside: avoid; }
th { background: #f1f0ec; } th, td { border: 1px solid #e6e5e1; padding: 3pt 5pt; vertical-align: top; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; page-break-inside: avoid; } em { color: #52514e; }
code { font-size: 8.6pt; background: #f4f3ef; padding: 0 2px; } pre { font-size: 8.2pt; background: #f7f7f5; padding: 6pt 8pt; white-space: pre-wrap; }
pre code { background: none; } hr { border: none; border-top: 1px solid #e6e5e1; margin: 12pt 0; } li { margin: 2pt 0; }
"""


LIST_ITEM = re.compile(r"^\s*([-*+]|\d+\.)\s")


def list_breaks(text):
    """Python-Markdown (unlike GitHub) needs a blank line before a list that follows a paragraph line."""
    out, fenced = [], False
    for line in text.splitlines():
        fenced ^= line.lstrip().startswith("```")
        prev = out[-1] if out else ""
        if (not fenced and LIST_ITEM.match(line) and prev.strip() and not LIST_ITEM.match(prev)
                and not prev.startswith((" ", "\t", "|"))):
            out.append("")
        out.append(line)
    return "\n".join(out)


def main(src, dst):
    src, dst = Path(src).resolve(), Path(dst).resolve()
    body = markdown.markdown(list_breaks(src.read_text()), extensions=["tables", "fenced_code", "sane_lists"])
    html = src.with_suffix(".html")
    html.write_text(f'<!doctype html><html><head><meta charset="utf-8"><title>{src.stem}</title>'
                    f"<style>{CSS}</style></head><body>{body}</body></html>")
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--allow-file-access-from-files", "--no-pdf-header-footer",
                    f"--print-to-pdf={dst}", f"file://{html}"], check=True, capture_output=True, timeout=180)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
