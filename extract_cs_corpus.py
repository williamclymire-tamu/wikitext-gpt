"""extract_cs_corpus.py -- build a large CS training corpus from five public,
real, license-clean sources:

1. patmorin/ods -- "Open Data Structures" textbook (CC-BY), LaTeX source,
   cleaned to plain text. Genuine rigorous textbook prose: proofs,
   complexity analysis, real explanations -- not just code comments.
2. trekhleb/javascript-algorithms -- README.md theory/definitions
   (graph theory, Big-O, algorithmic paradigms) + JS code.
3-5. TheAlgorithms/{Python,Java,C-Plus-Plus} -- large multi-language
   algorithm implementations with docstring/comment explanations.

Expects all five repos cloned alongside this script:
    git clone --depth 1 https://github.com/patmorin/ods.git
    git clone --depth 1 https://github.com/trekhleb/javascript-algorithms.git
    git clone --depth 1 https://github.com/TheAlgorithms/Python.git
    git clone --depth 1 https://github.com/TheAlgorithms/Java.git
    git clone --depth 1 https://github.com/TheAlgorithms/C-Plus-Plus.git
"""
import os
import re

def clean_latex(text):
    """Lightweight LaTeX -> plain text cleanup. Not perfect, but strips the
    bulk of markup noise so training tokens aren't dominated by LaTeX
    command syntax rather than actual prose."""
    text = re.sub(r'%.*', '', text)                                  # comments
    text = re.sub(r'\\index\{[^}]*\}', '', text)                      # index markers
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\(emph|textbf|textit)\{([^}]*)\}', r'\2', text)  # unwrap emphasis
    text = re.sub(r'\\(chapter|section|subsection)\{([^}]*)\}', r'\n\n\2\n', text)
    text = re.sub(r'\\begin\{(itemize|enumerate)\}', '', text)
    text = re.sub(r'\\end\{(itemize|enumerate)\}', '', text)
    text = re.sub(r'\\item\s*', '- ', text)
    text = re.sub(r'\\[a-zA-Z]+(\{[^}]*\})?', '', text)               # remaining commands
    text = re.sub(r'[{}]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)                            # collapse blank lines
    return text

def collect_ods():
    base = "ods/latex"
    out = []
    if not os.path.isdir(base):
        print(f"warning: {base} not found, skipping ODS textbook")
        return out
    for fn in sorted(os.listdir(base)):
        if fn.endswith(".tex"):
            with open(os.path.join(base, fn), encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            out.append((fn, clean_latex(raw)))
    return out

def collect_js_pairs():
    pairs = []
    src_root = os.path.join("javascript-algorithms", "src")
    if not os.path.isdir(src_root):
        print(f"warning: {src_root} not found, skipping JS repo")
        return pairs
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != "__test__" and d != "node_modules"]
        readme_path = os.path.join(root, "README.md")
        readme_content = None
        if os.path.isfile(readme_path):
            with open(readme_path, encoding="utf-8", errors="ignore") as f:
                readme_content = f.read()
        js_files = []
        for fn in sorted(files):
            if fn.endswith(".js") and ".test." not in fn:
                with open(os.path.join(root, fn), encoding="utf-8", errors="ignore") as f:
                    js_files.append((fn, f.read()))
        if readme_content or js_files:
            pairs.append((root, readme_content, js_files))
    return pairs

def collect_code_repo(repo_dir, extensions):
    files = []
    if not os.path.isdir(repo_dir):
        print(f"warning: {repo_dir} not found, skipping")
        return files
    for root, dirs, fnames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if "test" not in d.lower() and d != "__pycache__" and not d.startswith(".")]
        for fn in fnames:
            if fn.endswith(extensions) and "test" not in fn.lower():
                fp = os.path.join(root, fn)
                with open(fp, encoding="utf-8", errors="ignore") as f:
                    files.append((fp, f.read()))
    return files

if __name__ == "__main__":
    ods_chapters = collect_ods()
    js_pairs = collect_js_pairs()
    py_files = collect_code_repo("Python", (".py",))
    java_files = collect_code_repo("Java", (".java",))
    cpp_files = collect_code_repo("C-Plus-Plus", (".cpp", ".hpp", ".h"))

    print(f"ODS textbook: {len(ods_chapters)} chapters")
    print(f"JS repo: {sum(1 for _, r, _ in js_pairs if r)} theory READMEs, {sum(len(j) for _, _, j in js_pairs)} code files")
    print(f"Python: {len(py_files)} files, Java: {len(java_files)} files, C++: {len(cpp_files)} files")

    out_path = "cs_corpus.txt"
    total_bytes = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fn, content in ods_chapters:
            out.write(f"\n=== TEXTBOOK THEORY: Open Data Structures / {fn} ===\n")
            out.write(content)
            total_bytes += len(content)

        for dirpath, readme, js_files in js_pairs:
            rel = os.path.relpath(dirpath, "javascript-algorithms")
            if readme:
                out.write(f"\n=== THEORY: {rel} ===\n")
                out.write(readme)
                total_bytes += len(readme)
            for fn, content in js_files:
                out.write(f"\n=== CODE: {rel}/{fn} ===\n")
                out.write(content)
                total_bytes += len(content)

        for repo_name, files in [("Python", py_files), ("Java", java_files), ("C-Plus-Plus", cpp_files)]:
            for fp, content in files:
                rel = os.path.relpath(fp, repo_name)
                out.write(f"\n=== CODE: {repo_name}/{rel} ===\n")
                out.write(content)
                total_bytes += len(content)

    print(f"\nwrote {out_path}: {total_bytes:,} bytes (~{total_bytes // 4:,} tokens at ~4 chars/token)")