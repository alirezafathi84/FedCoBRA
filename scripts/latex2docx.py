"""Convert the IEEEtran paper to an editable IEEE-style .docx (native Word equations).

Pipeline: flatten \\input -> resolve cites/refs/numbering from main.aux -> pandoc -> python-docx/XML
post-processing (IEEE fonts, two-column body, full-width floats, ruled tables, references).
"""
import copy, re, subprocess, sys
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Inches
from lxml import etree

PAPER = "/home/claude/paper"
WORK = "/home/claude/docx"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{WORK}/FedCoBRA_IEEE_paper.docx"


def read(name):
    return open(f"{PAPER}/{name}.tex").read()


def flatten(text):
    return re.sub(r"\\input\{([^}]+)\}", lambda m: flatten(read(m.group(1))), text)


def balanced(s, i):
    """s[i] == '{' -> index after matching '}'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced")


ROMAN = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

# ------------------------------------------------------------------ inputs
main = read("main")
body = flatten(main[main.index("\\maketitle") + len("\\maketitle"):main.index("\\end{document}")])
body = "\n".join(l for l in body.split("\n") if not l.lstrip().startswith("%"))

aux = open(f"{PAPER}/main.aux").read()
labels = {}
for m in re.finditer(r"\\newlabel\{([^}]+)\}\{", aux):
    i = m.end()                      # at the first '{' of {{NUM}{PAGE}...}
    j = balanced(aux, i)             # NUM group spans aux[i:j]
    raw = aux[i + 1:j - 1]
    labels[m.group(1)] = re.sub(r"\\mbox|[{}\s]", "", raw)

refs_tex = read("refs")
bib_keys = re.findall(r"\\bibitem\{([^}]+)\}", refs_tex)
bib_num = {k: i + 1 for i, k in enumerate(bib_keys)}

# ------------------------------------------------------------------ citations and references
def cite_repl(m):
    nums = sorted({bib_num[k.strip()] for k in m.group(1).split(",")})
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"[{nums[i]}]--[{nums[j]}]" if j - i >= 2 else ", ".join(f"[{n}]" for n in nums[i:j + 1]))
        i = j + 1
    return ", ".join(out)


body = re.sub(r"\\cite\{([^}]+)\}", cite_repl, body)
body = re.sub(r"\\eqref\{([^}]+)\}", lambda m: f"({labels[m.group(1)]})", body)
body = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m.group(1)], body)

# ------------------------------------------------------------------ front matter
body = re.sub(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
              lambda m: "@@ABSTRACT@@\\textbf{\\emph{Abstract}---" + m.group(1) + "}", body, flags=re.S)
body = re.sub(r"\\begin\{IEEEkeywords\}\s*(.*?)\s*\\end\{IEEEkeywords\}",
              lambda m: "@@KEYWORDS@@\\textbf{\\emph{Index Terms}---" + m.group(1) + "}", body, flags=re.S)
body = re.sub(r"\\item\[\\textbf\{(RQ\d)\}\]", r"\\item \\textbf{\1:}", body)

# ------------------------------------------------------------------ section numbering (IEEE style)
out, pos, sec, sub = [], 0, 0, 0
for m in re.finditer(r"\\(section|subsection)(\*?)\{", body):
    if m.start() < pos:
        continue
    end = balanced(body, m.end() - 1)
    title = body[m.end():end - 1]
    if m.group(2):
        new = f"\\{m.group(1)}*{{{title}}}"
    elif m.group(1) == "section":
        sec += 1; sub = 0
        new = f"\\section*{{{ROMAN[sec]}. {title}}}"
    else:
        sub += 1
        new = f"\\subsection*{{{chr(64 + sub)}. {title}}}"
    out.append(body[pos:m.start()]); out.append(new); pos = end
body = "".join(out) + body[pos:]

# ------------------------------------------------------------------ theorem-like environments and proofs
counters = {"definition": 0, "proposition": 0}
def thm_repl(m):
    env, opt, content = m.group(1), m.group(2), m.group(3)
    counters[env] += 1
    content = re.sub(r"\\label\{[^}]*\}", "", content).strip()
    head = f"\\textbf{{{env.capitalize()} {counters[env]}}}" + (f" (\\emph{{{opt}}})" if opt else "")
    return f"{head}\\textbf{{:}} \\emph{{{content}}}"
body = re.sub(r"\\begin\{(definition|proposition)\}(?:\[([^\]]*)\])?(.*?)\\end\{\1\}", thm_repl, body, flags=re.S)
body = re.sub(r"\\begin\{IEEEproof\}(.*?)\\end\{IEEEproof\}",
              lambda m: "\\emph{Proof:} " + m.group(1).strip() + " $\\blacksquare$", body, flags=re.S)

# ------------------------------------------------------------------ equations (sequential numbering, as LaTeX)
eqn = 0
def fix_math(s):
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\(t?frac)(\w)(\w)", r"\\\1{\2}{\3}", s)
    s = re.sub(r"\\[bB]ig(?=[\[\]()|.]|\\\|)", "", s)
    s = s.replace("\\textstyle", "").replace("\\tfrac", "\\frac")
    s = re.sub(r"\(([^()]*)\)_\+", r"{(\1)}_+", s)
    s = re.sub(r"\[([^\[\]]*)\]_\+", r"{[\1]}_+", s)
    s = re.sub(r"\{\(([^()]*)\)\}_\+", r"\\max(0,\1)", s)          # (a)_+ == max(0, a)
    s = re.sub(r"\{\[([^\[\]]*)\]\}_\+", r"\\max(0,\1)", s)
    s = re.sub(r"\\sum_n\[(.*)\]", r"\\sum_n\\left(\1\\right)", s)
    return s.replace("\\gets", "\\leftarrow")
def eq_repl(m):
    global eqn
    env, content = m.group(1), m.group(2)
    if env == "equation":
        eqn += 1
        return f"\\[{fix_math(content).strip()}\\qquad({eqn})\\]"
    # split aligned rows into separate display equations (continuation rows joined), without '&'
    rows = [r for r in re.split(r"\\\\(?!\[)", content) if r.strip()]
    eqs, buf = [], ""
    for r in rows:
        r = fix_math(r.replace("&", "").replace("\\qquad", " ")).strip()
        if "\\nonumber" in r:
            r = r.replace("\\nonumber", "").strip()
            cand = (buf + " " + r).strip()
            if cand.count("[") != cand.count("]"):
                buf = cand                      # keep bracket-spanning lines together
            else:
                eqs.append(f"\\[{cand}\\]"); buf = ""
        else:
            eqn += 1
            eqs.append(f"\\[{(buf + ' ' + r).strip()}\\qquad({eqn})\\]"); buf = ""
    return "\n".join(eqs)
body = re.sub(r"\\begin\{(equation|align)\}(.*?)\\end\{\1\}", eq_repl, body, flags=re.S)
body = re.sub(r"\$([^$]*)\$", lambda m: "$" + fix_math(m.group(1)) + "$", body)

# ------------------------------------------------------------------ algorithm (no algorithmic support in pandoc)
algo = r"""@@ALGO@@\textbf{Algorithm 1:} FedCoBRA

\emph{Require:} clients with $a_k\in\{\mathrm{P},\mathrm{B}\}$, rounds $T$, local steps $E$.

\begin{enumerate}
\item Polygon clients send DP shape statistics; the server broadcasts $(\mu_c,\sigma_c^2)$.
\item Initialise $\theta^0=(\theta_S^0,\theta_C^0)$.
\item \textbf{for} $t=0,\dots,T-1$ \textbf{do}
\item \quad Broadcast $\theta^t$ (it also serves as the frozen teacher).
\item \quad \textbf{for} each client $k$ \textbf{in parallel do}
\item \quad\quad $\theta_k\leftarrow\theta^t$
\item \quad\quad \textbf{for} $e=1,\dots,E$ \textbf{do}
\item \quad\quad\quad Sample a batch; use $\mathcal{L}_\mathrm{P}$ if $a_k=\mathrm{P}$, else $\mathcal{L}_\mathrm{B}$ (Eq.~(""" + labels["eq:lb"] + r"""))
\item \quad\quad\quad $\theta_k\leftarrow\mathrm{OptStep}(\theta_k,\nabla\mathcal{L})$
\item \quad\quad Upload $\theta_k-\theta^t$
\item \quad $\theta_S^{t+1}\leftarrow\sum_k\frac{n_k}{n}\theta_{S,k}$; \quad $\theta_C^{t+1}\leftarrow$ RADA (Eq.~(""" + labels["eq:rada"] + r"""))
\end{enumerate}
@@ALGOEND@@
"""
body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", lambda m: algo, body, flags=re.S)

# ------------------------------------------------------------------ figures and tables
FIGS = {"figs/samples.pdf": "figs/samples.png", "figs/curves.pdf": "figs/curves.png"}
def float_repl(m):
    env, star, content = m.group(1), m.group(2), m.group(3)
    lab = re.search(r"\\label\{([^}]+)\}", content)
    num = labels[lab.group(1)] if lab else "?"
    c0 = content.index("\\caption{")
    c1 = balanced(content, c0 + len("\\caption"))
    cap = content[c0 + len("\\caption{"):c1 - 1]
    width = "7in" if star else "3.45in"
    if env == "figure":
        if "tikzpicture" in content:
            img = "figs/overview.png"
        else:
            src = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", content).group(1)
            img = FIGS[src]
        inner = f"\\begin{{figure}}\n\\includegraphics[width={width}]{{{img}}}\n\\caption{{Fig. {num}. {cap}}}\n\\end{{figure}}"
    else:
        tab = content[content.index("\\begin{tabular}"):content.index("\\end{tabular}") + len("\\end{tabular}")]
        tab = re.sub(r"\\cmidrule(\([a-z]*\))?\{[^}]*\}", "", tab)
        inner = f"\\begin{{table}}\n\\caption{{TABLE {num}. {cap}}}\n{tab}\n\\end{{table}}"
    inner = re.sub(r"\\label\{[^}]*\}", "", inner)
    return f"\n\n@@FW_START@@\n\n{inner}\n\n@@FW_END@@\n\n" if star else f"\n\n{inner}\n\n"
body = re.sub(r"\\begin\{(figure|table)(\*?)\}(?:\[[a-z]*\])?(.*?)\\end\{\1\2\}", float_repl, body, flags=re.S)

# ------------------------------------------------------------------ bibliography
items = re.findall(r"\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem\{|\\end\{thebibliography\})", refs_tex, flags=re.S)
bib = "\\section*{References}\n\n" + "\n\n".join(f"@@REF@@[{bib_num[k]}] {t.strip()}" for k, t in items)
body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", lambda m: bib, body, flags=re.S)

# ------------------------------------------------------------------ misc cleanup
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = body.replace("\\noindent", "")
body = body.replace("$>$", ">").replace("$<$", "<")
body = body.replace("$\\phi=|M|/|B|\\in(0,1]$", "$\\phi=|M|/|B|$, where $0<\\phi\\le 1$")
body = body.replace("$^\\ast$", "\\textsuperscript{*}").replace("$^\\dagger$", "\\textsuperscript{\u2020}")
body = body.replace("y^\\ast", "\\tilde{y}").replace("^\\ast", "^{*}")
body = body.replace("$^\\dagger$", "\\textsuperscript{\u2020}").replace("$^\\ast$", "\\textsuperscript{*}")
body = re.sub(r"\\textcolor\{[a-z]+\}\{([^}]*)\}", r"\\emph{\1}", body)
open(f"{WORK}/flat.tex", "w").write("\\documentclass{article}\n\\usepackage{amsmath,amssymb}\n\\begin{document}\n"
                                     + body + "\n\\end{document}\n")
open(f"{WORK}/meta.yaml", "w").write(
    'title: "' + "Polygons or Boxes? Annotation Granularity in Federated Instance Segmentation and FedCoBRA, "
    "a Contour--Box Reliability-Aware Learning Framework" + '"\n'
    "author:\n"
    '  - "First A. Author, Dept. of Computer Science, University Name, City, Country, first.author@university.edu"\n'
    '  - "Second B. Author, Dept. of Computer Science, University Name, City, Country, second.author@university.edu"\n')
r = subprocess.run(["pandoc", "flat.tex", "-f", "latex", "-t", "docx", "-o", "raw.docx",
                    "--metadata-file", "meta.yaml", "--resource-path", WORK], cwd=WORK, capture_output=True, text=True)
print("pandoc:", r.returncode, r.stderr.strip()[:800])

# ------------------------------------------------------------------ post-processing
doc = Document(f"{WORK}/raw.docx")
TNR = "Times New Roman"

def set_style(name, size, bold=None, italic=None, align=None, before=None, after=None, indent=None, caps=None):
    st = next((x for x in doc.styles if x.name is not None and x.name.lower() == name.lower()), None)
    if st is None:
        st = next((x for x in doc.styles if x.style_id == name.replace(" ", "")), None)
    if st is None:
        print("style not found:", name); return
    rpr0 = st.element.get_or_add_rPr()
    for el in rpr0.findall(qn("w:color")):
        rpr0.remove(el)
    st.font.name = TNR; st.font.size = Pt(size)
    rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = etree.SubElement(rpr, qn("w:rFonts"))
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rf.set(qn(a), TNR)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    if bold is not None: st.font.bold = bold
    if italic is not None: st.font.italic = italic
    if caps is not None: st.font.small_caps = caps
    st.font.color.rgb = None
    pf = st.paragraph_format
    if align is not None: pf.alignment = align
    if before is not None: pf.space_before = Pt(before)
    if after is not None: pf.space_after = Pt(after)
    if indent is not None: pf.first_line_indent = Inches(indent)
    pf.line_spacing = 1.0

J, C, L = WD_ALIGN_PARAGRAPH.JUSTIFY, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT
set_style("Normal", 10)
set_style("Body Text", 10, align=J, before=0, after=0, indent=0.14)
set_style("First Paragraph", 10, align=J, before=0, after=0, indent=0.14)
set_style("Compact", 10, align=J, before=0, after=0)
set_style("Title", 22, bold=False, align=C, before=0, after=8)
set_style("Author", 10, align=C, before=0, after=2)
set_style("Heading 1", 10, bold=False, italic=False, caps=True, align=C, before=8, after=4)
set_style("Heading 2", 10, bold=False, italic=True, align=L, before=6, after=3)
set_style("Table Caption", 8, align=C, before=6, after=3, caps=True)
set_style("Image Caption", 8, align=J, before=3, after=6)
set_style("Captioned Figure", 10, align=C, before=6, after=0)
set_style("Figure", 10, align=C)
for st in doc.styles:
    if st.style_id in ("Heading1", "Heading2"):
        st.paragraph_format.keep_with_next = True

# page setup on the final section
sec = doc.sections[-1]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(0.75), Inches(1.0)
sec.left_margin = sec.right_margin = Inches(0.625)
final = sec._sectPr

def make_sect(cols, continuous):
    sp = copy.deepcopy(final)
    for t in sp.findall(qn("w:type")):
        sp.remove(t)
    for c in sp.findall(qn("w:cols")):
        sp.remove(c)
    if continuous:
        t = etree.SubElement(sp, qn("w:type")); t.set(qn("w:val"), "continuous"); sp.insert(0, t)
    c = etree.SubElement(sp, qn("w:cols")); c.set(qn("w:num"), str(cols)); c.set(qn("w:space"), "360")
    return sp

def strip_marker(p, marker):
    for r in p.runs:
        if marker in r.text:
            r.text = r.text.replace(marker, ""); return

body_el = doc.element.body
for p in list(doc.paragraphs):
    t = p.text
    if "@@ABSTRACT@@" in t or "@@KEYWORDS@@" in t:
        mk = "@@ABSTRACT@@" if "@@ABSTRACT@@" in t else "@@KEYWORDS@@"
        strip_marker(p, mk)
        p.paragraph_format.alignment = J; p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.space_after = Pt(6)
        for r in p.runs:
            r.font.size = Pt(9)
    elif t.strip() in ("@@FW_START@@", "@@FW_END@@"):
        for r in p.runs:
            r.text = ""
        pf = p.paragraph_format; pf.space_before = pf.space_after = Pt(0)
        p._p.get_or_add_pPr().append(make_sect(2 if t.strip() == "@@FW_START@@" else 1, continuous=True))
    elif "@@REF@@" in t:
        strip_marker(p, "@@REF@@")
        pf = p.paragraph_format; pf.left_indent = Inches(0.3); pf.first_line_indent = Inches(-0.3)
        pf.alignment = J; pf.space_after = Pt(1)
        for r in p.runs:
            r.font.size = Pt(8)
    elif "@@ALGO@@" in t:
        strip_marker(p, "@@ALGO@@")
        pPr = p._p.get_or_add_pPr(); bdr = etree.SubElement(pPr, qn("w:pBdr"))
        for side in ("top", "bottom"):
            e = etree.SubElement(bdr, qn(f"w:{side}")); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6")
            e.set(qn("w:space"), "1"); e.set(qn("w:color"), "000000")
        p.paragraph_format.first_line_indent = Inches(0); p.paragraph_format.space_before = Pt(6)
    elif "@@ALGOEND@@" in t:
        for r in p.runs:
            r.text = ""
        pPr = p._p.get_or_add_pPr(); bdr = etree.SubElement(pPr, qn("w:pBdr"))
        e = etree.SubElement(bdr, qn("w:top")); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6")
        e.set(qn("w:space"), "1"); e.set(qn("w:color"), "000000")

# title block (title + authors) is single-column: end that section after the last author paragraph
authors = [p for p in doc.paragraphs if p.style is not None and p.style.name == "Author"]
(authors[-1] if authors else doc.paragraphs[0])._p.get_or_add_pPr().append(make_sect(1, continuous=False))

# final section: two columns, continuous
for c in final.findall(qn("w:cols")):
    final.remove(c)
for t in final.findall(qn("w:type")):
    final.remove(t)
t = etree.SubElement(final, qn("w:type")); t.set(qn("w:val"), "continuous"); final.insert(0, t)
c = etree.SubElement(final, qn("w:cols")); c.set(qn("w:num"), "2"); c.set(qn("w:space"), "360")

# tables: IEEE-like rules, 8 pt, centred numbers
for tb in doc.tables:
    tblPr = tb._tbl.tblPr
    for b in tblPr.findall(qn("w:tblBorders")):
        tblPr.remove(b)
    bd = etree.SubElement(tblPr, qn("w:tblBorders"))
    for side, sz in (("top", "8"), ("bottom", "8")):
        e = etree.SubElement(bd, qn(f"w:{side}")); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), sz)
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
    jc = tblPr.find(qn("w:jc"))
    if jc is None:
        jc = etree.SubElement(tblPr, qn("w:jc"))
    jc.set(qn("w:val"), "center")
    for ri, row in enumerate(tb.rows):
        for ci, cell in enumerate(row.cells):
            for p in cell.paragraphs:
                p.paragraph_format.alignment = L if ci == 0 else C
                p.paragraph_format.first_line_indent = Inches(0)
                for r in p.runs:
                    r.font.size = Pt(8)
        if ri == 0:
            for cell in row.cells:
                tcPr = cell._tc.get_or_add_tcPr(); tb_ = etree.SubElement(tcPr, qn("w:tcBorders"))
                e = etree.SubElement(tb_, qn("w:bottom")); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4")
                e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")

doc.core_properties.title = "Polygons or Boxes? FedCoBRA"
doc.core_properties.author = "First A. Author; Second B. Author"
doc.save(OUT)
print("saved", OUT, "| sections:", len(doc.sections), "| tables:", len(doc.tables),
      "| equations:", eqn, "| inline images:", len(doc.inline_shapes))
