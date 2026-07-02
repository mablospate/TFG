# Mathematize

Corrige errores de DeepL y convierte prosa matemática española a notación simbólica en texto plano. **Nunca escribe m:oMath.**

## Argumento

Opcional: rango `P516-P540` o índice de inicio `P516`.  
Sin argumento: escanea desde el último párrafo procesado hasta el final.

---

## Qué hace

### 1. Correcciones de daño DeepL (texto → texto)

- Espacio antes de extensión: `grover. rs` → `grover.rs`, `test_permutation. py` → `test_permutation.py`, etc.
- Capitalización incorrecta de crates: `Q1tsim` → `q1tsim`, `Q1TSIM` → `q1tsim`, `Quantr` → `quantr` (solo mid-sentence), `Quantrs2` → `quantrs2`, `Qcgpu` → `qcgpu`
- Snake_case partido: `build_ time` → `build_time`, `startup _time` → `startup_time`
- Cualquier identificador técnico alterado por DeepL

### 2. Prosa matemática española → símbolo en texto plano

Sustituir la frase completa por el símbolo, sin m:oMath, solo texto:

| Frase en prosa | Reemplazo en texto plano |
|---|---|
| `dos elevado a n` | `2^n` |
| `X elevado a Y` | `X^Y` |
| `A elevado a r/2` | `A^(r/2)` |
| `X módulo N` | `X mod N` |
| `módulo N` | `mod N` |
| `módulo quince` | `mod 15` |
| `raíz cuadrada de X` | `√(X)` |
| `raíz cuadrada de dos` | `√(2)` |
| `pi dividido por cuatro` | `π/4` |
| `pi dividido por dos` | `π/2` |
| `logaritmo en base dos de N` | `log₂(N)` |
| `el techo de X` | `⌈X⌉` |
| `gcd mayor que uno` | `gcd > 1` |
| Cardinal aislado en contexto matemático: `dos`, `tres`, `quince`, `cuatro`, `ocho`, `dieciséis` | `2`, `3`, `15`, `4`, `8`, `16` |

---

## Ejecución

### Paso 1 — Determinar rango

Sin argumento: haiku agent abre el docx, encuentra el último párrafo con alguna de las frases objetivo y usa ese índice como inicio.

### Paso 2 — Escanear (agente Explore)

El agente:
1. Abre el docx como ZIP, extrae `word/document.xml`
2. Para cada párrafo en el rango, extrae el texto plano concatenando **todos** los `w:t` y `m:t` (incluyendo dentro de m:oMath, para no perder contexto)
3. Busca ocurrencias de las frases de la tabla de arriba y errores de DeepL
4. Devuelve:

```python
{
  "deepl": {
    para_idx: [("texto_erróneo", "texto_correcto"), ...],
  },
  "prose": {
    para_idx: [("frase original en prosa", "símbolo en texto plano"), ...],
  }
}
```

### Paso 3 — Patch (agente Sonnet)

El agente usa únicamente `replace_text_in_para(para, old, new)` de `patch_doc_equations.py`. **No usa make_omath_eq ni ninguna función que cree m:oMath.**

1. Abre el docx como ZIP
2. Para cada entrada en `deepl`: aplica `replace_text_in_para(para, old, new)`
3. Para cada entrada en `prose`: aplica `replace_text_in_para(para, frase, símbolo)`
4. Guarda el docx modificado en el mismo path
5. Verifica integridad: `zipfile.ZipFile(...).testzip()` debe devolver `None`

### Paso 4 — Resumen

- Número de correcciones DeepL
- Número de conversiones de prosa
- Términos NOT_FOUND

---

## Referencia de ficheros

- **Documento**: `/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx`
- **Backup**: `/Users/pablomateos/TFG/Plantilla_TFG-PROYECTO(1).docx.backup`
- **Script base**: `/Users/pablomateos/TFG/patch_doc_equations.py` (solo `replace_text_in_para`)
- **Scratchpad**: `/private/tmp/claude-501/-Users-pablomateos-TFG/eb9aa27b-4155-4805-aebe-7e9c8d3b95ac/scratchpad/`
