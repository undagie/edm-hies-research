"""Inject Zotero citation/bibliography fields into rev1 DOCX → ``*_zotero.docx``.

Reuses submitted CSL-JSON for linked refs; 6 revision adds are embedded-only
placeholders (re-link in Zotero). Leaves clean rev1.docx untouched.

Usage: python scripts/inject_zotero_fields.py
"""

from __future__ import annotations

import json
import random
import re
import string
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SUBMIT = PAPER / "edm_hies_paper_inass_submit.docx"
REV1 = PAPER / "edm_hies_paper_inass_rev1.docx"
MD = PAPER / "edm_hies_paper_inass.md"
OUT = PAPER / "edm_hies_paper_inass_rev1_zotero.docx"

USER = 12502089
CITE = re.compile(r"\[\d+(?:[-,\s]+\d+)*\]")


def norm_title(t: str) -> str:
    t = re.sub(r"<[^>]+>", "", t)
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def rand_id(n=8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def rand_key(n=8) -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def expand(group: str) -> list[int]:
    nums = []
    for part in re.split(r"\s*,\s*", group.strip("[]")):
        if "-" in part:
            a, b = part.split("-")
            nums += list(range(int(a), int(b) + 1))
        else:
            nums.append(int(part))
    return nums


NEW_ITEMS = {
    17: {"type": "article-journal", "title": "A stacking-based RF-CatBoost model for student performance prediction",
         "container-title": "ICCK Transactions on Educational Data Mining", "volume": "1", "issue": "1",
         "page": "16-24", "DOI": "10.62762/TEDM.2025.397583",
         "author": [{"family": "Zhao", "given": "Jing"}], "issued": {"date-parts": [["2025"]]}},
    19: {"type": "article-journal", "title": "Individual comparisons by ranking methods",
         "container-title": "Biometrics Bulletin", "volume": "1", "issue": "6", "page": "80-83",
         "DOI": "10.2307/3001968", "author": [{"family": "Wilcoxon", "given": "Frank"}],
         "issued": {"date-parts": [["1945"]]}},
    20: {"type": "article-journal", "title": "A simple sequentially rejective multiple test procedure",
         "container-title": "Scandinavian Journal of Statistics", "volume": "6", "issue": "2", "page": "65-70",
         "author": [{"family": "Holm", "given": "Sture"}], "issued": {"date-parts": [["1979"]]}},
    26: {"type": "article-journal", "title": "Complementary CatBoost based on residual error for student performance prediction",
         "container-title": "Pattern Recognition", "volume": "161", "page": "111265",
         "DOI": "10.1016/j.patcog.2024.111265",
         "author": [{"family": "Fan", "given": "Zongwen"}, {"family": "Gou", "given": "Jin"},
                    {"family": "Weng", "given": "Shaoyuan"}], "issued": {"date-parts": [["2025"]]}},
    27: {"type": "article-journal", "title": "Towards accessible AI for addressing students' academic dropout: An auto machine learning and explainable artificial intelligence approach",
         "container-title": "Universal Access in the Information Society", "volume": "25", "issue": "1",
         "DOI": "10.1007/s10209-025-01278-4",
         "author": [{"family": "Bunay-Guisnan", "given": "Pamela"}, {"family": "Lara", "given": "Juan A."},
                    {"family": "Cano", "given": "Alberto"}, {"family": "Cerezo", "given": "Rebeca"},
                    {"family": "Romero", "given": "Cristobal"}], "issued": {"date-parts": [["2026"]]}},
    28: {"type": "article-journal", "title": "Explainable artificial intelligence in LSTM transformer models for student performance analysis",
         "container-title": "Discover Computing", "volume": "28", "issue": "1",
         "DOI": "10.1007/s10791-025-09814-9",
         "author": [{"family": "Ofori", "given": "Elijah"}, {"family": "Dake", "given": "Delali Kwasi"}],
         "issued": {"date-parts": [["2025"]]}},
}


def json_objects_after(s: str, marker: str):
    for mm in re.finditer(re.escape(marker), s):
        i = s.find("{", mm.end())
        if i < 0:
            continue
        depth = 0
        for j, ch in enumerate(s[i:], i):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    yield s[i:j + 1]
                    break


def extract_library_items() -> dict:
    doc = zipfile.ZipFile(SUBMIT).read("word/document.xml").decode("utf-8", "ignore")
    instr = "".join(re.findall(r"<w:instrText[^>]*>(.*?)</w:instrText>", doc, re.DOTALL))
    for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")]:
        instr = instr.replace(a, b)
    items = {}
    for blob in json_objects_after(instr, "ADDIN ZOTERO_ITEM CSL_CITATION"):
        try:
            obj = json.loads(blob)
        except Exception:
            continue
        for ci in obj.get("citationItems", []):
            t = ci.get("itemData", {}).get("title", "")
            if t:
                items[norm_title(t)] = ci
    return items


def build_num_to_item() -> dict:
    lib = extract_library_items()
    md = MD.read_text(encoding="utf-8")
    reflist = md[md.rfind("\n## References"):]
    num2item = {}
    for num, entry in re.findall(r"^\[(\d+)\]\s+(.*)$", reflist, re.MULTILINE):
        num = int(num)
        if num in NEW_ITEMS:
            item = dict(NEW_ITEMS[num])
            item["id"] = 900000 + num
            num2item[num] = {"id": item["id"],
                             "uris": [f"http://zotero.org/users/{USER}/items/NEW{num:05d}Z"],
                             "itemData": {**item, "id": item["id"]}}
            continue
        tm = re.search(r'"([^"]{6,})"', entry) or re.search(r"\*([^*]{6,})\*", entry)
        key = norm_title(tm.group(1)) if tm else ""
        hit = lib.get(key)
        if not hit and key:
            for k, v in lib.items():
                if key in k or k in key:
                    hit = v
                    break
        if hit is None:
            raise SystemExit(f"No library match and no constructed item for ref [{num}]: {entry[:60]}")
        num2item[num] = hit
    missing = [n for n in range(1, 50) if n not in num2item]
    if missing:
        raise SystemExit(f"Unmapped references: {missing}")
    return num2item


def xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def citation_field(nums: list[int], shown: str, num2item: dict, rpr: str) -> str:
    items = [num2item[n] for n in nums]
    payload = {
        "citationID": rand_id(),
        "properties": {"formattedCitation": shown, "plainCitation": shown, "noteIndex": 0},
        "citationItems": [{"id": it["id"], "uris": it["uris"], "itemData": it["itemData"]} for it in items],
        "schema": "https://github.com/citation-style-language/schema/raw/master/csl-citation.json",
    }
    instr = " ADDIN ZOTERO_ITEM CSL_CITATION " + xml_escape(json.dumps(payload, ensure_ascii=False)) + " "
    R = f"<w:r>{rpr}"
    return (
        f"{R}<w:fldChar w:fldCharType=\"begin\"/></w:r>"
        f"{R}<w:instrText xml:space=\"preserve\">{instr}</w:instrText></w:r>"
        f"{R}<w:fldChar w:fldCharType=\"separate\"/></w:r>"
        f"{R}<w:t xml:space=\"preserve\">{xml_escape(shown)}</w:t></w:r>"
        f"{R}<w:fldChar w:fldCharType=\"end\"/></w:r>"
    )


def wrap_body_citations(body: str, num2item: dict) -> tuple[str, int]:
    """Wrap citation tokens in text runs with ZOTERO_ITEM fields."""
    run_re = re.compile(r"<w:r\b[^>]*>(<w:rPr>.*?</w:rPr>)?<w:t\b[^>]*>([^<]*)</w:t></w:r>", re.DOTALL)
    count = 0

    def repl(m: re.Match) -> str:
        nonlocal count
        rpr = m.group(1) or ""
        text = m.group(2)
        if not CITE.search(text):
            return m.group(0)
        out = []
        pos = 0
        for c in CITE.finditer(text):
            before = text[pos:c.start()]
            if before:
                out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{before}</w:t></w:r>')
            shown = c.group(0)
            out.append(citation_field(expand(shown), shown, num2item, rpr))
            count += 1
            pos = c.end()
        tail = text[pos:]
        if tail:
            out.append(f'<w:r>{rpr}<w:t xml:space="preserve">{tail}</w:t></w:r>')
        return "".join(out)

    return run_re.sub(repl, body), count


def build_bibliography_field(ref_block: str) -> str:
    """Wrap reference-list paragraphs in one ZOTERO_BIBL field."""
    prefs = ' ADDIN ZOTERO_BIBL {"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY '
    begin = (
        '<w:r><w:fldChar w:fldCharType="begin"/></w:r>'
        f'<w:r><w:instrText xml:space="preserve">{prefs}</w:instrText></w:r>'
        '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
    )
    end = '<w:r><w:fldChar w:fldCharType="end"/></w:r>'
    first_p = re.search(r"<w:p\b[^>]*>(<w:pPr>.*?</w:pPr>)?", ref_block, re.DOTALL)
    ins_at = first_p.end()
    ref_block = ref_block[:ins_at] + begin + ref_block[ins_at:]
    last = ref_block.rfind("</w:p>")
    ref_block = ref_block[:last] + end + ref_block[last:]
    return ref_block


def main() -> None:
    random.seed(20260717)
    num2item = build_num_to_item()

    with zipfile.ZipFile(REV1, "r") as z:
        parts = {n: z.read(n) for n in z.namelist()}
    doc = parts["word/document.xml"].decode("utf-8")

    ref_h = doc.find(">References<")
    p_start = doc.rfind("<w:p", 0, ref_h)
    body, refregion = doc[:p_start], doc[p_start:]
    tail_m = re.search(r"<w:sectPr\b", refregion)
    ref_paras, tail = (refregion[:tail_m.start()], refregion[tail_m.start():]) if tail_m else (refregion, "")

    body2, n_cites = wrap_body_citations(body, num2item)
    heading_end = ref_paras.find("</w:p>") + len("</w:p>")
    heading, entries = ref_paras[:heading_end], ref_paras[heading_end:]
    entries2 = build_bibliography_field(entries)

    new_doc = body2 + heading + entries2 + tail
    parts["word/document.xml"] = new_doc.encode("utf-8")

    # Carry Zotero prefs (style = zotero.org/styles/ijies) from submit doc.
    parts["docProps/custom.xml"] = zipfile.ZipFile(SUBMIT).read("docProps/custom.xml")

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for n, d in parts.items():
            zout.writestr(n, d)

    print(f"Wrote {OUT.name}")
    print(f"In-text citation fields injected: {n_cites}")
    print(f"References mapped: {len(num2item)} (42 linked + 6 embedded + Wohlin book)")


if __name__ == "__main__":
    main()
