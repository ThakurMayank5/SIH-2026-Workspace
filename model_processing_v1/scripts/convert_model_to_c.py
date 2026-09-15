"""
Convert vaani_dscnn_v1_int8.tflite into a C array that can be compiled
straight into the ESP32 firmware (no filesystem/SPIFFS needed).

Usage:
    python convert_model_to_c.py path/to/vaani_dscnn_v1_int8.tflite ../src

Produces, in the given output directory:
    model_data.h
    model_data.cc
"""

import sys
from pathlib import Path

BYTES_PER_LINE = 12
VAR_NAME = "g_vaani_model_data"


def to_c_array(data: bytes, var_name: str) -> str:
    lines = [f"alignas(16) const unsigned char {var_name}[] = {{"]
    for i in range(0, len(data), BYTES_PER_LINE):
        chunk = data[i:i + BYTES_PER_LINE]
        row = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    lines.append(f"const unsigned int {var_name}_len = {len(data)};")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python convert_model_to_c.py <model.tflite> <output_dir>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = model_path.read_bytes()

    header = f"""#ifndef VAANI_MODEL_DATA_H_
#define VAANI_MODEL_DATA_H_

extern const unsigned char {VAR_NAME}[];
extern const unsigned int {VAR_NAME}_len;

#endif  // VAANI_MODEL_DATA_H_
"""

    source = f'#include "model_data.h"\n\n' + to_c_array(data, VAR_NAME) + "\n"

    (out_dir / "model_data.h").write_text(header)
    (out_dir / "model_data.cc").write_text(source)

    print(f"Wrote {out_dir / 'model_data.h'}")
    print(f"Wrote {out_dir / 'model_data.cc'}")
    print(f"Model size: {len(data):,} bytes")


if __name__ == "__main__":
    main()