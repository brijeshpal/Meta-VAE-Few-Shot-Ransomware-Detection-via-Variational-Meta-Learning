"""
feature_extractor.py
--------------------
Extracts STATIC features from a file (no execution required).

Detection is done on static features because we want to judge a file the moment
it lands on the system, BEFORE it is ever run. We never execute the sample.

The feature set mirrors what real ML-based AV / EDR engines use:
  * file size + structural header fields
  * number of sections and their entropy (packed/encrypted code -> high entropy)
  * import statistics (how many DLLs / functions it pulls in)
  * count of commonly-abused Windows APIs imported (used purely as a risk signal)
  * security flags (ASLR/DEP, whether the binary is digitally signed)

For non-PE files (or unparseable ones) we still emit a feature vector using
file-level statistics (size + byte entropy) so nothing crashes the pipeline.
"""

from __future__ import annotations
import math
import os
from collections import Counter

try:
    import pefile  # parses Windows PE (.exe/.dll) headers
    _HAVE_PEFILE = True
except Exception:
    _HAVE_PEFILE = False


# APIs that are disproportionately common in malware (injection, persistence,
# anti-analysis, etc.). We only COUNT how many appear -- this is a detection
# signal, not instructions. Sourced from widely published threat intel.
SUSPICIOUS_APIS = {
    "VirtualAlloc", "VirtualProtect", "WriteProcessMemory", "CreateRemoteThread",
    "ReadProcessMemory", "SetWindowsHookEx", "GetAsyncKeyState", "OpenProcess",
    "LoadLibraryA", "LoadLibraryW", "GetProcAddress", "WinExec", "ShellExecuteA",
    "RegSetValueExA", "RegCreateKeyExA", "InternetOpenA", "InternetOpenUrlA",
    "URLDownloadToFileA", "CreateServiceA", "CryptEncrypt", "IsDebuggerPresent",
    "CreateProcessA", "NtUnmapViewOfSection", "ResumeThread", "QueueUserAPC",
}

# Order matters: training and detection must use the SAME column order.
FEATURE_NAMES = [
    "file_size",
    "byte_entropy",
    "is_pe",
    "num_sections",
    "mean_section_entropy",
    "max_section_entropy",
    "num_imported_dlls",
    "num_imported_funcs",
    "num_suspicious_apis",
    "num_exports",
    "has_signature",
    "aslr_enabled",
    "dep_enabled",
    "size_of_code",
    "entry_in_last_section",
]


def _shannon_entropy(data: bytes) -> float:
    """Shannon entropy of a byte string (0 = uniform, 8 = max randomness)."""
    if not data:
        return 0.0
    counts = Counter(data)
    length = len(data)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _empty_pe_features() -> dict:
    return {
        "is_pe": 0, "num_sections": 0, "mean_section_entropy": 0.0,
        "max_section_entropy": 0.0, "num_imported_dlls": 0,
        "num_imported_funcs": 0, "num_suspicious_apis": 0, "num_exports": 0,
        "has_signature": 0, "aslr_enabled": 0, "dep_enabled": 0,
        "size_of_code": 0, "entry_in_last_section": 0,
    }


def _extract_pe_features(path: str) -> dict:
    """Parse PE structure. Returns _empty_pe_features() if it isn't a PE."""
    feats = _empty_pe_features()
    if not _HAVE_PEFILE:
        return feats
    try:
        pe = pefile.PE(path, fast_load=True)
        pe.parse_data_directories()
    except Exception:
        return feats  # not a valid PE -> treated as non-PE

    feats["is_pe"] = 1

    # Sections + entropy
    entropies = []
    for section in pe.sections:
        try:
            entropies.append(section.get_entropy())
        except Exception:
            pass
    feats["num_sections"] = len(pe.sections)
    if entropies:
        feats["mean_section_entropy"] = sum(entropies) / len(entropies)
        feats["max_section_entropy"] = max(entropies)

    # Imports
    dll_count, func_count, suspicious = 0, 0, 0
    if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_count += 1
            for imp in entry.imports:
                func_count += 1
                if imp.name:
                    name = imp.name.decode(errors="ignore")
                    if name in SUSPICIOUS_APIS:
                        suspicious += 1
    feats["num_imported_dlls"] = dll_count
    feats["num_imported_funcs"] = func_count
    feats["num_suspicious_apis"] = suspicious

    # Exports
    if hasattr(pe, "DIRECTORY_ENTRY_EXPORT"):
        feats["num_exports"] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)

    # Digital signature (presence of a security directory)
    try:
        sec_dir = pe.OPTIONAL_HEADER.DATA_DIRECTORY[
            pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_SECURITY"]
        ]
        feats["has_signature"] = 1 if sec_dir.VirtualAddress != 0 else 0
    except Exception:
        pass

    # Security mitigations from DllCharacteristics flags
    try:
        dch = pe.OPTIONAL_HEADER.DllCharacteristics
        feats["aslr_enabled"] = 1 if dch & 0x0040 else 0  # DYNAMIC_BASE
        feats["dep_enabled"] = 1 if dch & 0x0100 else 0   # NX_COMPAT
    except Exception:
        pass

    try:
        feats["size_of_code"] = int(pe.OPTIONAL_HEADER.SizeOfCode)
    except Exception:
        pass

    # Entry point sitting in the last section is a classic packer tell
    try:
        ep = pe.OPTIONAL_HEADER.AddressOfEntryPoint
        last = pe.sections[-1]
        if last.VirtualAddress <= ep < last.VirtualAddress + last.Misc_VirtualSize:
            feats["entry_in_last_section"] = 1
    except Exception:
        pass

    pe.close()
    return feats


def extract_features(path: str) -> dict:
    """Return a dict of all features (FEATURE_NAMES) for a file on disk."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        # Read up to 2 MB for entropy -- enough to characterize, bounds memory.
        blob = f.read(2 * 1024 * 1024)

    feats = {"file_size": size, "byte_entropy": _shannon_entropy(blob)}
    feats.update(_extract_pe_features(path))
    return feats


def features_to_vector(feats: dict) -> list:
    """Convert a feature dict into an ordered numeric vector for the model."""
    return [feats.get(name, 0) for name in FEATURE_NAMES]


if __name__ == "__main__":
    import sys, json
    if len(sys.argv) != 2:
        print("usage: python feature_extractor.py <file>")
        sys.exit(1)
    print(json.dumps(extract_features(sys.argv[1]), indent=2))
