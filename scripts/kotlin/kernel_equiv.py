import re, sys, difflib
from collections import Counter

VAR = re.compile(r"\b(?:[a-z]+)_[0-9]+\b")
DECL_PREFIXES = ("int ", "float ", "long ", "bool ", "half ", "tornado_ptr_t", "char ", "short ", "double ", "ulong ", "uint ", "unsigned ")

def parse(p, pattern):
    t = re.sub(r"\x1b\[[0-9;]*m", "", open(p, errors="replace").read())
    return {m.group(1): m.group(0) for m in re.finditer(pattern, t, re.S)}

def equivalent(a, b):
    """True if b is a with variables renamed (a bijection) and lines reordered."""
    la, lb = a.split("\n"), b.split("\n")
    mapping = {}
    sm = difflib.SequenceMatcher(None, la, lb, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for x, y in zip(la[i1:i2], lb[j1:j2]):
                if VAR.sub("V", x) == VAR.sub("V", y):
                    for u, v in zip(VAR.findall(x), VAR.findall(y)):
                        mapping.setdefault(u, v)
    for x, y in zip(la, lb):
        if x != y and VAR.sub("V", x) == VAR.sub("V", y) and not x.strip().startswith(DECL_PREFIXES):
            for u, v in zip(VAR.findall(x), VAR.findall(y)):
                mapping.setdefault(u, v)
    if len(set(mapping.values())) != len(mapping):
        return False, mapping
    renamed = VAR.sub(lambda m: mapping.get(m.group(0), m.group(0)), a)
    DECL = ("int ", "float ", "long ", "bool ", "half ", "tornado_ptr_t", "char ", "short ", "double ", "ulong ", "uint ", "unsigned ")

    def decl(l):
        kind, _, names = l.strip().partition(" ")
        return kind + " " + ",".join(sorted(x.strip() for x in names.rstrip("; ").split(",")))
    norm = lambda s: Counter(decl(l) if l.strip().startswith(DECL) else l.strip() for l in s.split("\n"))
    return norm(renamed) == norm(b), mapping

if __name__ == "__main__":
    ja, ko, pattern = sys.argv[1], sys.argv[2], sys.argv[3]
    jk, kk = parse(ja, pattern), parse(ko, pattern)
    same = [n for n in jk if jk[n] == kk.get(n)]
    print("kernels: java %d, kotlin %d; byte-identical: %d" % (len(jk), len(kk), len(same)))
    for n in jk:
        if n in kk and jk[n] != kk[n]:
            ok, m = equivalent(jk[n], kk[n])
            renames = {k: v for k, v in m.items() if k != v}
            print("  %-100s equivalent up to renaming/reordering: %s (%d renamed variables)" % (n.split("_kernels_")[-1][:100], ok, len(renames)))
    missing = set(jk) ^ set(kk)
    if missing:
        print("  only in one build:", sorted(missing))
