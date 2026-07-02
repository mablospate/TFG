#!/usr/bin/env python3
"""
Patch a Word document's XML to:
  1. Apply DeepL text fixes.
  2. Convert plain-text technical identifiers into inline equations (m:oMath).
  3. Fix a malformed "controlled" equation in P217.

Uses only the standard library plus lxml.
"""

import copy
import os
import shutil
import zipfile

from lxml import etree

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DOCX_PATH = "/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx"
BACKUP_PATH = "/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx.backup"
DOCUMENT_XML = "word/document.xml"

# ---------------------------------------------------------------------------
# Namespaces / helpers
# ---------------------------------------------------------------------------
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def w(t):
    return f"{{{W}}}{t}"


def m(t):
    return f"{{{M}}}{t}"


# ---------------------------------------------------------------------------
# Equations to inject, keyed by paragraph index.
# ---------------------------------------------------------------------------
EQUATIONS_BY_PARA = {
    197: ["AdderCircuit", "QuantumCircuit", "exponentiate_modulo"],
    211: ["SamplerV2", "EstimatorV2"],
    212: ["qiskit-aer", "PassManager", "optimization_level=1"],
    484: ["ZGate().control(n-1)", "PassManager"],
    485: ["PassManager", "optimization_level=1", "SamplerV2", "qiskit-aer"],
    486: ["benjamin-assel/qiskit-shor"],
    487: ["build_time", "startup_time", "simulation_time"],
    488: ["test_adder.py", "test_shor.py", "test_grover.py"],
    489: ["test_shor.py", "test_grover.py"],
    491: ["SamplerV1", "SamplerV2", "result.data", "result.quasi_dists"],
    492: ["startup_time", "PassManager"],
    495: ["cirq.LineQubit.range(n)"],
    496: ["cirq.Z.controlled(num_controls=n-1)", "ZGate().control(n-1)"],
    497: ["cirq.ArithmeticGate", "apply()", "ModularExp"],
    498: ["cirq.qft(*exponent_qubits, inverse=True)", "AdderCircuit"],
    499: ["ArithmeticGate"],
    500: ["test_grover.py", "test_shor.py", "test_modular_exp.py", "apply()", "test_adder.py"],
    502: ["ArithmeticGate"],
    503: ["cirq.Simulator", "startup_time", "build_time", "simulation_time"],
    505: ["qpp-cpu"],
    506: ["cudaq.make_kernel()", "kernel.qalloc(n)", "kernel.h()", "kernel.x()", "kernel.cz()"],
    507: ["cudaq.sample()", "bitstring[::-1]"],
    509: ["AdderCircuit", "ArithmeticGate"],
    510: ["build_mod_exp_permutation", "controlled_swap_permutation"],
    511: ["cr1", "qft.py"],
    512: ["test_grover.py", "test_shor.py", "test_permutation.py", "build_mod_exp_permutation", "controlled_swap_permutation"],
    515: ['cudaq.set_target("nvidia")', "qpp-cpu"],
}

# DeepL plain-text fixes: para index -> (old, new)
DEEPL_FIXES = {
    238: ("Q1tsim", "q1tsim"),
    239: ("Q1TSIM", "q1tsim"),
    512: ("test_permutation. py", "test_permutation.py"),
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _needs_space_preserve(txt):
    return bool(txt) and (txt.startswith(" ") or txt.endswith(" "))


def make_omath(text):
    """Build and return an m:oMath element containing a single styled run."""
    omath = etree.Element(m("oMath"))
    r = etree.SubElement(omath, m("r"))
    rpr = etree.SubElement(r, w("rPr"))
    rfonts = etree.SubElement(rpr, w("rFonts"))
    rfonts.set(w("ascii"), "Cambria Math")
    rfonts.set(w("hAnsi"), "Cambria Math")
    lang = etree.SubElement(rpr, w("lang"))
    lang.set(w("eastAsia"), "es-ES")
    mt = etree.SubElement(r, m("t"))
    if _needs_space_preserve(text):
        mt.set(f"{{{XML_NS}}}space", "preserve")
    mt.text = text
    return omath


def make_text_run(rpr_copy, txt):
    """Create a w:r with an optional deep-copied rPr and a w:t holding txt."""
    r = etree.Element(w("r"))
    if rpr_copy is not None:
        r.append(copy.deepcopy(rpr_copy))
    t = etree.SubElement(r, w("t"))
    if _needs_space_preserve(txt):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = txt
    return r


def _run_text(run):
    """Concatenate the text of all w:t children of a run."""
    parts = []
    for t in run.findall(w("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _omath_text(omath):
    """Concatenate all m:t texts inside an m:oMath element."""
    parts = []
    for t in omath.iter(m("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)


def _has_omath_ancestor(node):
    anc = node.getparent()
    while anc is not None:
        if anc.tag == m("oMath"):
            return True
        anc = anc.getparent()
    return False


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------
def convert_to_equations_in_para(para, identifier):
    """
    Convert the FIRST plain-text occurrence of `identifier` in `para` into an
    inline equation.

    Returns:
      -1  if `identifier` already appears inside an existing m:oMath
       1  if a conversion was made
       0  if `identifier` was not found as plain text
    """
    # Idempotency / skip-existing check.
    for omath in para.iter(m("oMath")):
        if identifier in _omath_text(omath):
            return -1

    runs = list(para.iter(w("r")))
    for run in runs:
        if _has_omath_ancestor(run):
            continue

        text = _run_text(run)
        if identifier not in text:
            continue

        rpr = run.find(w("rPr"))
        rpr_copy = copy.deepcopy(rpr) if rpr is not None else None

        idx = text.index(identifier)
        before = text[:idx]
        after = text[idx + len(identifier):]

        omath = make_omath(identifier)

        new_nodes = []
        if before:
            new_nodes.append(make_text_run(rpr_copy, before))
        new_nodes.append(omath)
        if after:
            new_nodes.append(make_text_run(rpr_copy, after))

        parent = run.getparent()
        index_in_parent = parent.index(run)
        for offset, node in enumerate(new_nodes):
            parent.insert(index_in_parent + offset, node)
        parent.remove(run)

        return 1

    return 0


# ---------------------------------------------------------------------------
# DeepL text fixes
# ---------------------------------------------------------------------------
def _apply_text_fix_to_para(para, old, new):
    """Apply a string replace to every w:t in `para`. Return True if any hit."""
    changed = False
    for t in para.iter(w("t")):
        if t.text and old in t.text:
            t.text = t.text.replace(old, new)
            changed = True
    return changed


def apply_deepl_fixes(paragraphs):
    """Apply the DeepL text fixes, with a +/-10 paragraph fallback scan."""
    fixes_applied = 0
    for expected_idx, (old, new) in sorted(DEEPL_FIXES.items()):
        if 0 <= expected_idx < len(paragraphs) and _apply_text_fix_to_para(
            paragraphs[expected_idx], old, new
        ):
            print(f"P{expected_idx}: fixed '{old}' -> '{new}'")
            fixes_applied += 1
            continue

        found = False
        for i in range(expected_idx - 10, expected_idx + 11):
            if i == expected_idx:
                continue
            if i < 0 or i >= len(paragraphs):
                continue
            if _apply_text_fix_to_para(paragraphs[i], old, new):
                print(
                    f"P{i}: fixed '{old}' -> '{new}' "
                    f"(found near expected P{expected_idx})"
                )
                fixes_applied += 1
                found = True
                break

        if not found:
            print(f"P{expected_idx}: '{old}' NOT FOUND")
    return fixes_applied


# ---------------------------------------------------------------------------
# Equation injection loop
# ---------------------------------------------------------------------------
def inject_equations(paragraphs):
    total = 0
    for expected_idx, identifiers in sorted(EQUATIONS_BY_PARA.items()):
        for identifier in sorted(identifiers, key=len, reverse=True):
            if 0 <= expected_idx < len(paragraphs):
                r = convert_to_equations_in_para(paragraphs[expected_idx], identifier)
            else:
                r = 0

            if r == -1:
                print(f"P{expected_idx}: '{identifier}' already an equation, skipping")
                continue
            if r >= 1:
                print(f"P{expected_idx}: converted '{identifier}'")
                total += 1
                continue

            # r == 0: fallback scan +/-10 paragraphs.
            found = False
            for i in range(expected_idx - 10, expected_idx + 11):
                if i == expected_idx:
                    continue
                if i < 0 or i >= len(paragraphs):
                    continue
                ri = convert_to_equations_in_para(paragraphs[i], identifier)
                if ri == -1:
                    print(
                        f"P{i}: '{identifier}' already an equation, skipping "
                        f"(found near expected P{expected_idx})"
                    )
                    found = True
                    break
                if ri >= 1:
                    print(
                        f"P{i}: converted '{identifier}' "
                        f"(found near expected P{expected_idx})"
                    )
                    total += 1
                    found = True
                    break

            if not found:
                print(f"P{expected_idx}: '{identifier}' NOT FOUND")
    return total


# ---------------------------------------------------------------------------
# P217 malformed "controlled" equation fix
# ---------------------------------------------------------------------------
def fix_p217_controlled(paragraphs):
    target_text = "cirq.Z.controlled(num_controls=n-1)"

    def find_controlled_omath(para):
        for omath in para.iter(m("oMath")):
            if "controlled" in _omath_text(omath).lower():
                return omath
        return None

    omath = None
    where = None
    if 0 <= 217 < len(paragraphs):
        omath = find_controlled_omath(paragraphs[217])
        if omath is not None:
            where = 217

    if omath is None:
        for i in range(217 - 5, 217 + 6):
            if i == 217:
                continue
            if i < 0 or i >= len(paragraphs):
                continue
            omath = find_controlled_omath(paragraphs[i])
            if omath is not None:
                where = i
                break

    if omath is None:
        print("P217: malformed 'controlled' equation NOT FOUND")
        return False

    # Clear existing children.
    for child in list(omath):
        omath.remove(child)

    # Rebuild interior.
    r = etree.SubElement(omath, m("r"))
    rpr = etree.SubElement(r, w("rPr"))
    rfonts = etree.SubElement(rpr, w("rFonts"))
    rfonts.set(w("ascii"), "Cambria Math")
    rfonts.set(w("hAnsi"), "Cambria Math")
    lang = etree.SubElement(rpr, w("lang"))
    lang.set(w("eastAsia"), "es-ES")
    mt = etree.SubElement(r, m("t"))
    if _needs_space_preserve(target_text):
        mt.set(f"{{{XML_NS}}}space", "preserve")
    mt.text = target_text

    if where == 217:
        print("P217: fixed malformed 'controlled' equation")
    else:
        print(
            f"P217: fixed malformed 'controlled' equation (found near expected P217, at P{where})"
        )
    return True


# ---------------------------------------------------------------------------
# Repack
# ---------------------------------------------------------------------------
def repack(new_document_bytes):
    """Rewrite the docx in place, replacing word/document.xml."""
    tmp_path = DOCX_PATH + ".tmp"
    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        infos = zin.infolist()
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in infos:
                data = zin.read(info.filename)
                if info.filename == DOCUMENT_XML:
                    data = new_document_bytes
                zout.writestr(info, data)
    os.replace(tmp_path, DOCX_PATH)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Backup.
    if not os.path.exists(BACKUP_PATH):
        shutil.copy(DOCX_PATH, BACKUP_PATH)
        print(f"Backup created: {BACKUP_PATH}")
    else:
        print(f"Backup already exists, skipping: {BACKUP_PATH}")

    # 2. Load.
    with zipfile.ZipFile(DOCX_PATH, "r") as zin:
        document_bytes = zin.read(DOCUMENT_XML)

    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.fromstring(document_bytes, parser=parser)
    paragraphs = tree.findall(f".//{w('p')}")
    print(f"Loaded document with {len(paragraphs)} paragraphs")

    # 3. DeepL text fixes (FIRST).
    print("\n--- DeepL text fixes ---")
    deepl_count = apply_deepl_fixes(paragraphs)

    # 6. Equation injection.
    print("\n--- Equation injection ---")
    eq_count = inject_equations(paragraphs)

    # 7. P217 malformed equation fix.
    print("\n--- P217 fix ---")
    p217_fixed = fix_p217_controlled(paragraphs)

    # 8. Save / repack.
    print("\n--- Saving ---")
    new_bytes = etree.tostring(
        tree, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    repack(new_bytes)
    print(f"Saved: {DOCX_PATH}")

    # 9. Summary.
    print("\n=== Summary ===")
    print(f"Equation conversions: {eq_count}")
    print(f"DeepL text fixes:     {deepl_count}")
    print(f"P217 fix status:      {'fixed' if p217_fixed else 'NOT FOUND'}")


if __name__ == "__main__":
    main()
