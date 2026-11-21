#!/usr/bin/env python3
# Group a PSP ELF's import stubs by module, then fill the import table.
#
# psp-fixup-imports walks .sceStub.text in address order and appends each stub
# to the .lib.stub record its first word points at. The moment one module's
# stubs are not contiguous it warns and keeps writing, so one record's range
# covers another module's stubs and the loader resolves the wrong NID. The
# link order that produces this is normal: libcglue and libpspuser both import
# from the same kernel modules, and --as-needed pulls them in separately.
#
# This regroups the stub bodies and the NID words together, zeroes each
# record's function count so psp-fixup-imports rebuilds the table, and rewrites
# every j/jal in the executable sections whose target moved. The symbol table is
# left at the original addresses; only the instructions change.

import struct
import subprocess
import sys


def sections(path):
    out = subprocess.check_output(["psp-readelf", "-S", "-W", path], text=True)
    found = {}
    for line in out.splitlines():
        # the section index is "[ 7]" for the first ten and "[12]" after that, so
        # it cannot be split reliably; the name is the first "." token instead
        parts = line.split()
        if "PROGBITS" not in parts:
            continue
        name = next((p for p in parts if p.startswith(".")), None)
        if name is None:
            continue
        i = parts.index("PROGBITS")
        found[name] = (int(parts[i + 1], 16), int(parts[i + 2], 16), int(parts[i + 3], 16))
    return found


def executable_sections(path):
    out = subprocess.check_output(["psp-readelf", "-S", "-W", path], text=True)
    result = []
    for line in out.splitlines():
        parts = line.split()
        if "PROGBITS" not in parts or "AX" not in parts:
            continue
        i = parts.index("PROGBITS")
        result.append((int(parts[i + 1], 16), int(parts[i + 2], 16), int(parts[i + 3], 16)))
    return result


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: psp-group-imports.py <elf>")
    path = sys.argv[1]

    data = bytearray(open(path, "rb").read())
    sec = sections(path)
    stub_addr, stub_off, stub_size = sec[".sceStub.text"]
    lib_addr, lib_off, lib_size = sec[".lib.stub"]
    nid_addr, nid_off, nid_size = sec[".rodata.sceNid"]

    count = stub_size // 8
    if nid_size // 4 != count:
        sys.exit("psp-group-imports: stub section and nid section differ in length")

    # word 0 of an unfixed stub is the address of the owning .lib.stub record
    owner = []
    for i in range(count):
        word = struct.unpack_from("<I", data, stub_off + 8 * i)[0]
        if word == 0x03E00008:
            sys.exit("psp-group-imports: stubs are already fixed up; relink first")
        if not (lib_addr <= word < lib_addr + lib_size) or (word - lib_addr) % 20:
            sys.exit("psp-group-imports: stub %d does not point at a .lib.stub record" % i)
        owner.append(word)

    order = sorted(range(count), key=lambda i: (owner[i], i))
    if order != list(range(count)):
        moved = {old: new for new, old in enumerate(order)}

        data[stub_off:stub_off + stub_size] = b"".join(
            data[stub_off + 8 * i:stub_off + 8 * i + 8] for i in order)
        data[nid_off:nid_off + nid_size] = b"".join(
            data[nid_off + 4 * i:nid_off + 4 * i + 4] for i in order)

        # j and jal encode the low 28 bits of the target. Retarget the ones that
        # land on a stub whose slot changed; everything else is left alone.
        retargeted = 0
        for addr, off, size in executable_sections(path):
            for disp in range(0, size - size % 4, 4):
                insn = struct.unpack_from("<I", data, off + disp)[0]
                opcode = insn >> 26
                if opcode not in (2, 3):
                    continue
                target = ((addr + disp) & 0xF0000000) | ((insn & 0x03FFFFFF) << 2)
                slot = (target - stub_addr) // 8
                if stub_addr <= target < stub_addr + stub_size and (target - stub_addr) % 8 == 0 \
                        and moved[slot] != slot:
                    new_target = stub_addr + 8 * moved[slot]
                    insn = (insn & 0xFC000000) | ((new_target >> 2) & 0x03FFFFFF)
                    struct.pack_into("<I", data, off + disp, insn)
                    retargeted += 1

        sys.stderr.write("psp-group-imports: regrouped %d stubs, retargeted %d branches\n"
                         % (count, retargeted))

    # func_count lives at offset 10 of each 20-byte record. Zeroing it makes
    # psp-fixup-imports treat every record as unfixed and refill it.
    for rec in range(lib_size // 20):
        struct.pack_into("<H", data, lib_off + rec * 20 + 10, 0)

    open(path, "wb").write(data)

    result = subprocess.run(["psp-fixup-imports", path], stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or "Warning" in result.stderr or "Error" in result.stderr:
        sys.stderr.write(result.stderr)
        sys.exit("psp-group-imports: psp-fixup-imports did not accept the regrouped stubs")


if __name__ == "__main__":
    main()
