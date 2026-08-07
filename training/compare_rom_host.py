#!/usr/bin/env python3
"""Compare a GAME12_PROBE ares log's greedy arm against the numpy oracle for the
same blob, character for character."""
import re, sys
import eval12 as E
log, blob = sys.argv[1], sys.argv[2]
m = E.load_seq(blob) if open(blob,'rb').read(3) == b"SEQ" else E.load_seai(blob)
rom = {}
for ln in open(log, errors="replace"):
    g = re.match(r"^G12 (\d+)\|([^|]*)\|(.*)$", ln.rstrip("\n"))
    if g: rom[int(g.group(1))] = (g.group(2), g.group(3))
same = 0
for i in sorted(rom):
    key, r = rom[i]
    h = E.generate(m, key.encode("ascii"), len(r))[:len(r)]
    ok = (h == r)
    same += ok
    print(f"  [{'SAME' if ok else 'DIFF'}] {key:32s}\n     ROM  {r!r}\n     HOST {h!r}")
print(f"ROM vs numpy-oracle: {same}/{len(rom)} character-identical")
