#!/usr/bin/env python3
"""Java -> Kotlin converter for TornadoVM kernel classes (static methods only).

Covers the subset used by jitllm's kernel files: static methods, primitive/array/TornadoVM-typed
locals, for loops (turned into while loops so the loop shape TornadoVM sees is unchanged, or into
parallelFor for @Parallel loops), casts, shifts, ternaries and `new HalfFloat(...)`. Whatever it
does not cover is left for the Kotlin compiler to point at.

usage: j2k_kernels.py <Java file> <Kotlin file> <JvmName>
"""
import re
import sys

PRIM = {"int": "Int", "float": "Float", "long": "Long", "double": "Double", "boolean": "Boolean",
        "byte": "Byte", "short": "Short", "char": "Char"}
ARR = {"float[]": "kotlin.FloatArray", "int[]": "kotlin.IntArray", "long[]": "kotlin.LongArray",
       "double[]": "kotlin.DoubleArray", "byte[]": "kotlin.ByteArray", "short[]": "kotlin.ShortArray",
       "boolean[]": "kotlin.BooleanArray"}
TYPE_RE = r"(?:void|int|float|long|double|boolean|byte|short|char|[A-Z][A-Za-z0-9_]*)(?:\[\])?"


def ktype(t):
    t = t.strip()
    if t in ARR:
        return ARR[t]
    if t.endswith("[]"):
        return "Array<%s>" % ktype(t[:-2])
    return PRIM.get(t, t)


def read_operand(s, i):
    """Operand starting at s[i]: a balanced (...) group, or a name/call/index chain."""
    if s[i] == "(":
        depth = 0
        for j in range(i, len(s)):
            depth += s[j] == "("
            depth -= s[j] == ")"
            if depth == 0:
                return s[i:j + 1], j + 1
    j = i
    if j < len(s) and s[j] in "+-":
        j += 1
    while j < len(s):
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fFlL]?", s[j:])
        if not m:
            break
        j += m.end()
        while j < len(s) and s[j] in "([":
            close = ")" if s[j] == "(" else "]"
            depth = 0
            for k in range(j, len(s)):
                depth += s[k] in "(["
                depth -= s[k] in ")]"
                if depth == 0:
                    j = k + 1
                    break
        if j < len(s) and s[j] == ".":
            j += 1
            continue
        break
    return s[i:j], j


def convert_casts(s):
    out = s
    for jt, kt in [("float", "toFloat"), ("int", "toInt"), ("long", "toLong"), ("double", "toDouble"),
                   ("byte", "toByte"), ("short", "toShort")]:
        pat = "(%s) " % jt
        pat2 = "(%s)" % jt
        while True:
            i = out.find(pat)
            skip = len(pat)
            if i < 0:
                i = out.find(pat2)
                skip = len(pat2)
                if i < 0 or not (i + skip < len(out) and (out[i + skip].isalnum() or out[i + skip] in "(_")):
                    break
            operand, end = read_operand(out, i + skip)
            wrapped = operand if operand.startswith("(") else "(%s)" % operand if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*(\[[^\]]*\])?", operand) else operand
            out = out[:i] + "%s.%s()" % (wrapped, kt) + out[end:]
    return out


def convert_expr(s):
    s = re.sub(r"\bnew HalfFloat\(", "HalfFloat(", s)
    s = convert_casts(s)
    # compound shift assignments
    s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_\[\]]*) >>>= ([^;]+)", r"\1 = \1 ushr (\2)", s)
    s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_\[\]]*) >>= ([^;]+)", r"\1 = \1 shr (\2)", s)
    s = re.sub(r"(\b[A-Za-z_][A-Za-z0-9_\[\]]*) <<= ([^;]+)", r"\1 = \1 shl (\2)", s)
    s = s.replace(" >>> ", " ushr ").replace(" >> ", " shr ").replace(" << ", " shl ")
    s = re.sub(r" & (0x[0-9A-Fa-f]+|\d+)", r" and \1", s)
    s = s.replace(" | ", " or ").replace(" ^ ", " xor ")
    # ternary: (cond) ? a : b  ->  if (cond) a else b   (single level, as used in these files)
    m = re.search(r"(\([^?]*\)) \? (.+) : (.+?)(;?)$", s)
    if m:
        s = s[:m.start()] + "if %s %s else %s%s" % (m.group(1), m.group(2), m.group(3), m.group(4))
    return s


def split_top_level(s, sep=","):
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += ch in "(["
        depth -= ch in ")]"
        if ch == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def split_multi_declarations(lines):
    out, n = [], 0
    while n < len(lines):
        line = lines[n]
        code = line.split("//", 1)[0].rstrip()
        m = re.match(r"^(\s*)(final )?(%s) ([A-Za-z0-9_]+ = .*)$" % TYPE_RE, code)
        if m and (code.endswith(",") or (code.endswith(";") and len(split_top_level(m.group(4).rstrip(";"))) > 1)):
            stmt = m.group(4)
            while not stmt.endswith(";") and n + 1 < len(lines):
                n += 1
                stmt += " " + lines[n].split("//", 1)[0].strip()
            for d in split_top_level(stmt.rstrip(";")):
                out.append("%s%s %s;" % (m.group(1), m.group(3), d.strip()))
            n += 1
            continue
        out.append(line)
        n += 1
    return out


def join_continuations(lines):
    """Joins Java statements that span several lines into one line (Kotlin ends a statement at a
    newline before a binary operator). Method signatures are left to the signature parser."""
    out, n = [], 0
    while n < len(lines):
        line = lines[n]
        code = line.split("//", 1)[0].rstrip()
        s = code.strip()
        if s.startswith("public static"):
            while not line.split("//", 1)[0].rstrip().endswith("{") and n + 1 < len(lines):
                out.append(line)
                n += 1
                line = lines[n]
            out.append(line)
            n += 1
            continue
        if s and not s.startswith(("*", "/*", "@")) and not s.endswith((";", "{", "}")) and not s.startswith("}"):
            indent = line[:len(line) - len(line.lstrip())]
            joined = s
            while n + 1 < len(lines):
                n += 1
                nxt = lines[n].split("//", 1)[0].strip()
                if not nxt:
                    continue
                joined += " " + nxt
                if nxt.endswith((";", "{")):
                    break
            out.append(indent + joined)
            n += 1
            continue
        out.append(line)
        n += 1
    return out


KOTLIN_KEYWORDS = ("val", "fun", "in", "is", "object", "when", "typealias", "as")


def convert(java, jvm_name):
    # identifiers that are keywords in Kotlin, and bitwise complement of a literal
    java = re.sub(r"\b(%s)\b(?=\s*[=;),*+\-/<>\]])" % "|".join(KOTLIN_KEYWORDS), r"`\1`", java)
    java = re.sub(r"\b(%s)\b(?= [=*+\-/<>])" % "|".join(KOTLIN_KEYWORDS), r"`\1`", java)
    java = re.sub(r"~(\d+)", r"\1.inv()", java)
    lines = java.split("\n")
    out = ['@file:JvmName("%s")' % jvm_name, ""]
    i = 0
    # header: package + imports
    while i < len(lines) and not re.match(r"public (final )?class ", lines[i]):
        line = lines[i]
        if line.startswith("import uk.ac.manchester.tornado.api.annotations.Parallel"):
            out.append("import uk.ac.manchester.tornado.kotlin.api.parallelFor")
        elif line.startswith("package ") or line.startswith("import "):
            out.append(line.rstrip(";"))
        else:
            out.append(line)
        i += 1
    i += 1  # skip class line
    body = lines[i:]
    # drop the final class brace
    for k in range(len(body) - 1, -1, -1):
        if body[k].strip() == "}":
            body = body[:k]
            break
    body = [l[4:] if l.startswith("    ") else l for l in body]
    text = "\n".join(body)
    # drop the default constructor (and its javadoc)
    text = re.sub(r"\n?(/\*\*[^\n]*\*/\n)?public %s\(\) \{\}\n" % re.escape(jvm_name), "\n", text)

    res = []
    blines = join_continuations(split_multi_declarations(text.split("\n")))
    n = 0
    scopes = {}  # indentation -> names declared in the block at that indentation

    def visible(name):
        return any(name in names for names in scopes.values())

    def declare(ind, name):
        scopes.setdefault(len(ind), set()).add(name)

    # (indent, update, wrapped): while loops (update appended before the closing brace), parallelFor
    # loops (update None), and whether the loop is wrapped in run { } to scope its variable as in Java.
    loop_stack = []

    def emit(s):
        extra = 4 * sum(1 for e in loop_stack if e[2])
        res.append(" " * extra + s if s.strip() else s)
    while n < len(blines):
        line = blines[n]
        # method signature (may span lines)
        m = re.match(r"^(\s*)public static (?:final )?(%s) ([A-Za-z0-9_]+)\((.*)$" % TYPE_RE, line)
        if m:
            indent, ret, name, rest = m.groups()
            sig = rest
            while not re.search(r"\)\s*\{\s*$", sig):
                n += 1
                sig += "\n" + blines[n]
            params_text = re.sub(r"\)\s*\{\s*$", "", sig)
            params = []
            for pl in params_text.split("\n"):
                comment = ""
                if "//" in pl:
                    pl, comment = pl.split("//", 1)
                    comment = " //" + comment
                for p in [x.strip() for x in pl.split(",") if x.strip()]:
                    p = re.sub(r"^final ", "", p)
                    pm = re.match(r"(%s) ([A-Za-z0-9_]+)$" % TYPE_RE, p)
                    params.append(("%s: %s" % (pm.group(2), ktype(pm.group(1))), comment))
                    comment = ""
            scopes.clear()
            retk = "" if ret == "void" else ": %s" % ktype(ret)
            if len(params) <= 3 and not any(c for _, c in params):
                emit("%sfun %s(%s)%s {" % (indent, name, ", ".join(p for p, _ in params), retk))
            else:
                emit("%sfun %s(" % (indent, name))
                for k, (p, c) in enumerate(params):
                    emit("%s        %s,%s" % (indent, p, c))
                emit("%s)%s {" % (indent, retk))
            n += 1
            continue

        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())]
        if stripped and not stripped.startswith(("//", "*", "/*")):
            for level in [l for l in scopes if l > len(indent)]:
                del scopes[level]

        # close a loop opened below
        if loop_stack and stripped.startswith("}") and indent == loop_stack[-1][0]:
            lind, update, wrapped = loop_stack[-1]
            if update is not None:
                emit("%s    %s" % (lind, update))
            emit(line)
            loop_stack.pop()
            if wrapped:
                emit("%s}" % lind)
            n += 1
            continue

        # for loops
        loop_comment = ""
        cm_ = re.match(r"^(for \(.*\) \{)\s*(//.*)$", stripped)
        if cm_:
            stripped, loop_comment = cm_.group(1), "  " + cm_.group(2)
        em = re.match(r"^for \(; ([^;]+); (.+)\) \{$", stripped)
        if em:
            cond, update = convert_expr(em.group(1)), convert_expr(em.group(2))
            emit("%swhile (%s) {%s" % (indent, cond, loop_comment))
            loop_stack.append((indent, update, False))
            n += 1
            continue
        fm = re.match(r"^for \((@Parallel )?(?:(%s) )?([A-Za-z0-9_]+) = ([^;]+); ([^;]+); (.+)\) \{$" % TYPE_RE, stripped)
        if fm:
            par, jtype, var, init, cond, update = fm.groups()
            init, cond, update = convert_expr(init), convert_expr(cond), convert_expr(update)
            if par:
                cm = re.match(r"^%s < (.+)$" % re.escape(var), cond)
                assert cm and update == "%s++" % var, stripped
                emit("%sparallelFor(%s, %s) { %s ->" % (indent, init, cm.group(1), var))
                loop_stack.append((indent, None, False))
            elif jtype:
                # Java scopes the loop variable to the loop: run { } does the same (inline, no object)
                emit("%srun {" % indent)
                loop_stack.append((indent, update, True))
                emit("%svar %s: %s = %s" % (indent, var, ktype(jtype), init))
                emit("%swhile (%s) {%s" % (indent, cond, loop_comment))
            else:
                emit("%s%s = %s" % (indent, var, init))
                emit("%swhile (%s) {%s" % (indent, cond, loop_comment))
                loop_stack.append((indent, update, False))
            n += 1
            continue

        # local declarations
        dm = re.match(r"^(final )?(%s) (`?[A-Za-z0-9_]+`?)( = (.*))?;(\s*//.*)?$" % TYPE_RE, stripped)
        if dm and dm.group(2) not in ("return",):
            _, jt, var, _, expr, comment = dm.groups()
            kt = ktype(jt)
            comment = comment or ""
            if visible(var):
                # a second declaration of the same name in a sibling scope: reuse the variable
                emit("%s%s = %s%s" % (indent, var, convert_expr(expr), comment) if expr is not None else "%s// (re-declared %s)" % (indent, var))
            elif expr is None:
                emit("%svar %s: %s%s" % (indent, var, kt, comment))
                declare(indent, var)
            else:
                e = convert_expr(expr)
                if kt in ("Float",) and re.fullmatch(r"-?\d+", e):
                    e += "f"
                emit("%svar %s: %s = %s%s" % (indent, var, kt, e, comment))
                declare(indent, var)
            n += 1
            continue

        # other statements
        if stripped.endswith(";") and not stripped.startswith("//") and not stripped.startswith("*"):
            code = line.rstrip()
            cpos = code.find(" //")
            comment = ""
            if cpos >= 0 and code[:cpos].rstrip().endswith(";"):
                code, comment = code[:cpos], code[cpos:]
            emit(convert_expr(code.rstrip()[:-1]) + comment)
        elif re.search(r"\)\s*\?", stripped) or " ? " in stripped:
            emit(convert_expr(line))
        else:
            # control-flow lines: if (...) {, } else {, etc.
            emit(convert_expr(line) if stripped.startswith(("if", "} else", "while", "return")) else line)
        n += 1

    out.extend(res)
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    src, dst, name = sys.argv[1:4]
    kt = convert(open(src).read(), name)
    open(dst, "w").write(kt)
    print("wrote", dst, len(kt.splitlines()), "lines")
