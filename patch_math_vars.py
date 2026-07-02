import copy, os, re, shutil, zipfile
from lxml import etree

DOCX_PATH = "/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx"
DOCUMENT_XML = "word/document.xml"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML_NS = "http://www.w3.org/XML/1998/namespace"

def w(t): return f"{{{W}}}{t}"
def m(t): return f"{{{M}}}{t}"

def _needs_space_preserve(txt):
    return bool(txt) and (txt.startswith(" ") or txt.endswith(" "))

def make_omath(text):
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
    r = etree.Element(w("r"))
    if rpr_copy is not None:
        r.append(copy.deepcopy(rpr_copy))
    t = etree.SubElement(r, w("t"))
    if _needs_space_preserve(txt):
        t.set(f"{{{XML_NS}}}space", "preserve")
    t.text = txt
    return r

def _run_text(run):
    parts = []
    for t in run.findall(w("t")):
        if t.text:
            parts.append(t.text)
    return "".join(parts)

def _omath_text(omath):
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

def convert_to_equations_in_para(para, identifier):
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

def repack(new_document_bytes):
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


def _next_word(text, end_idx):
    """Return the next alphabetic word (lowercased) in text after position
    end_idx, skipping intervening non-alpha chars. Empty if none."""
    j = end_idx
    n = len(text)
    while j < n and not text[j].isalpha():
        j += 1
    start = j
    while j < n and text[j].isalpha():
        j += 1
    return text[start:j].lower()


def _find_allowed_match(text, pattern, forbidden):
    """Find the first regex match in text whose following word is NOT in the
    forbidden set. Returns the match object or None."""
    for match in pattern.finditer(text):
        if not forbidden:
            return match
        nxt = _next_word(text, match.end())
        if nxt in forbidden:
            continue
        return match
    return None


# Spanish words that, when they immediately follow a standalone "A", mean the
# "A" is the Spanish preposition/article — NOT a math variable. These must be
# skipped so phrases like "A diferencia", "A partir", "A continuación" survive.
A_FORBIDDEN_NEXT = {
    "diferencia", "partir", "continuación", "continuacion", "menudo", "su",
    "sus", "esto", "esta", "este", "estos", "estas", "veces", "través",
    "traves", "medida", "cambio", "pesar", "favor", "fin", "causa", "lo",
    "la", "las", "los", "él", "el", "ella", "cada", "menos", "mano", "raíz",
    "raiz", "lomejor", "propósito", "proposito", "grandes", "partirde",
}


def convert_var_to_equation(para, var_char, forbidden_next_words=None):
    """Convert ALL word-boundary plain-text occurrences of a single-char
    variable var_char in para into equations. Returns count converted.

    If forbidden_next_words is given, an occurrence whose immediately-following
    word (lowercased) is in that set is skipped (used to protect the Spanish
    article "A" in phrases like "A diferencia")."""
    pattern = re.compile(
        r"(?<![A-Za-záéíóúÁÉÍÓÚñÑü_\d])"
        + re.escape(var_char)
        + r"(?![A-Za-záéíóúÁÉÍÓÚñÑü_\d])"
    )
    forbidden = forbidden_next_words or set()
    count = 0
    # Re-collect runs each pass because the tree mutates.
    changed = True
    while changed:
        changed = False
        runs = [r for r in para.iter(w("r")) if not _has_omath_ancestor(r)]
        for run in runs:
            text = _run_text(run)
            if not text:
                continue
            match = _find_allowed_match(text, pattern, forbidden)
            if not match:
                continue
            i = match.start()
            before = text[:i]
            after = text[i + 1:]
            rpr = run.find(w("rPr"))
            rpr_copy = copy.deepcopy(rpr) if rpr is not None else None
            omath = make_omath(var_char)
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
            count += 1
            changed = True
            break  # restart scan after mutation
    return count


CHANGES = [
    (186, ["X", "Z"]),
    (187, ["X", "Z"]),
    (188, ["n"]),
    (193, ["r/2", "N", "A", "r"]),
    (194, ["N", "A"]),
    (195, ["4n + 2", "2n", "n + 2", "N", "n"]),
    (196, ["A", "N"]),
    (198, ["2n", "r", "A", "N"]),
    (199, ["2n/r", "r", "N"]),
    (213, ["N"]),
    (217, ["Z"]),
    (220, ["n"]),
    (228, ["Toffoli", "CNOT", "SWAP", "CX", "H", "X", "Y", "Z", "S", "T"]),
    (240, ["n"]),
    (447, ["n", "N"]),
    (450, ["n"]),
    (484, ["n-1", "Z"]),
    (487, ["N"]),
    (489, ["X"]),
    (494, ["MCZ"]),
    (495, ["n - 1 - i"]),
    (496, ["MCZ"]),
    (498, ["2n", "n + 2", "n"]),
    (500, ["N"]),
    (501, ["n-1-i"]),
    (509, ["A", "N"]),
    (510, ["CNOT"]),
    (512, ["n", "A", "N"]),
    (513, ["2^(m-1)", "2^m/r", "A", "m", "r"]),
]


def main():
    parser = etree.XMLParser(remove_blank_text=False)
    with zipfile.ZipFile(DOCX_PATH, "r") as z:
        xml_bytes = z.read(DOCUMENT_XML)
    tree = etree.fromstring(xml_bytes, parser)

    paragraphs = tree.findall(f".//{w('p')}")
    print(f"Total paragraphs: {len(paragraphs)}")

    total_conversions = 0

    for para_idx, terms in CHANGES:
        if para_idx >= len(paragraphs):
            print(f"P{para_idx}: index out of range (only {len(paragraphs)} paragraphs)")
            continue
        para = paragraphs[para_idx]
        for term in terms:
            if len(term) == 1 and term.isalpha():
                fw = A_FORBIDDEN_NEXT if term == "A" else None
                count = convert_var_to_equation(para, term, fw)
                if count == 0:
                    print(f"P{para_idx}: single-var '{term}' NOT FOUND")
                else:
                    print(f"P{para_idx}: converted single-var '{term}' x{count}")
                total_conversions += count
            else:
                count = 0
                result = convert_to_equations_in_para(para, term)
                if result == -1:
                    print(f"P{para_idx}: '{term}' already an equation, skipping")
                elif result == 0:
                    print(f"P{para_idx}: multi-char '{term}' NOT FOUND")
                else:
                    count += result
                    while True:
                        result = convert_to_equations_in_para(para, term)
                        if result <= 0:
                            break
                        count += result
                    if count > 0:
                        print(f"P{para_idx}: converted multi-char '{term}' x{count}")
                    total_conversions += count

    new_bytes = etree.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    repack(new_bytes)

    print("=== Summary ===")
    print(f"Total conversions: {total_conversions}")


if __name__ == "__main__":
    main()
