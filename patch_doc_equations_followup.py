#!/usr/bin/env python3
"""
Follow-up patch: convert additional plain-text identifiers that were missed
in the first pass due to string mismatches (args included, case differences).

Targets:
  P506: kernel.h(qubit), kernel.x(qubit), kernel.cz(control, target)
  P515: CudaQ.set_target("nvidia")
  P503: Cirq.Simulator

Imports helpers from patch_doc_equations (safe because main() is guarded).
"""

import os
import sys
import zipfile
import tempfile

# ---------------------------------------------------------------------------
# Import helpers from the existing module
# ---------------------------------------------------------------------------
sys.path.insert(0, "/Users/pablomateos/TFG")
from patch_doc_equations import (
    convert_to_equations_in_para,
    w, m,
)

from lxml import etree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCX_PATH = "/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx"
DOCUMENT_XML = "word/document.xml"

# ---------------------------------------------------------------------------
# Additions: paragraph index -> list of identifiers to convert
# ---------------------------------------------------------------------------
ADDITIONS = {
    506: ["kernel.cz(control, target)", "kernel.h(qubit)", "kernel.x(qubit)"],
    515: ['CudaQ.set_target("nvidia")'],
    503: ["Cirq.Simulator"],
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load document XML
    with zipfile.ZipFile(DOCX_PATH) as z:
        xml_bytes = z.read(DOCUMENT_XML)

    tree = etree.fromstring(xml_bytes)
    paragraphs = tree.findall(f".//{w('p')}")
    print(f"Total paragraphs: {len(paragraphs)}")

    total_converted = 0

    for idx, identifiers in sorted(ADDITIONS.items()):
        # Sort longest-first so longer identifiers are matched before substrings
        for identifier in sorted(identifiers, key=len, reverse=True):
            if idx >= len(paragraphs):
                print(f"P{idx}: index out of range for '{identifier}'")
                continue

            result = convert_to_equations_in_para(paragraphs[idx], identifier)

            if result == -1:
                print(f"P{idx}: already equation, skipping '{identifier}'")
            elif result >= 1:
                print(f"P{idx}: converted '{identifier}'")
                total_converted += 1
            else:
                # NOT FOUND at expected index — try ±3 fallback
                print(f"P{idx}: NOT FOUND '{identifier}' at P{idx} — trying ±3 fallback...")
                found_fallback = False
                for fi in range(max(0, idx - 3), min(len(paragraphs), idx + 4)):
                    if fi == idx:
                        continue
                    fb_result = convert_to_equations_in_para(paragraphs[fi], identifier)
                    if fb_result == -1:
                        print(f"  -> P{fi}: already equation (found near P{idx})")
                        found_fallback = True
                        break
                    elif fb_result >= 1:
                        print(f"  -> P{fi}: converted '{identifier}' (found near expected P{idx})")
                        total_converted += 1
                        found_fallback = True
                        break
                if not found_fallback:
                    print(f"  -> NOT FOUND in ±3 range around P{idx}")

    # Repack the docx in place
    new_xml = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".docx", dir=os.path.dirname(DOCX_PATH))
    os.close(tmp_fd)

    try:
        with zipfile.ZipFile(DOCX_PATH, "r") as zin, \
             zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == DOCUMENT_XML:
                    zout.writestr(item, new_xml)
                else:
                    zout.writestr(item, zin.read(item.filename))

        os.replace(tmp_path, DOCX_PATH)
        print(f"\nDocx repacked successfully.")
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"\nSummary: {total_converted} conversion(s) made.")


if __name__ == "__main__":
    main()
