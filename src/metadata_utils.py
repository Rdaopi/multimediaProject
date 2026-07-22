"""
Utilità condivise per etichettare gli embedding con i metadati del dataset
ITASpoof (reale/sintetico, sistema, speaker, sentence).

Fonte primaria: i CSV in protocols/{split}_metadata.csv (filename, system, label),
che contengono l'etichettatura ufficiale del dataset. Se un file non compare in
nessun CSV di protocols (es. dataset diverso, o file locali di test), si ricade
sul parsing del nome file come fallback, per non rompere l'uso in locale.

Formato nomi file: detection_{generatore}_speaker{speakerID}_sentence{sentenceID}.wav
'natural' = reale (system 0, label 0); ogni altro generatore = sintetico (system 1-7, label 1).
"""

import os
import re
import pandas as pd

FILENAME_RE = re.compile(
    r"detection_(?P<generator>.+?)_speaker(?P<speaker>.+?)_sentence(?P<sentence>[^_.]+)",
    re.IGNORECASE,
)

# Percorso dei protocols del dataset ITASpoof. Sovrascrivibile con la variabile
# d'ambiente ITASPOOF_PROTOCOLS_DIR se il dataset si trova altrove.
PROTOCOLS_DIR = os.environ.get(
    "ITASPOOF_PROTOCOLS_DIR",
    "/media/SSD_new/ITASpoof/detection/protocols",
)


def _parse_filename(filename):
    """Fallback: ricava generator/speaker/sentence/voice_type dal nome file."""
    base = os.path.splitext(os.path.basename(str(filename)))[0]
    m = FILENAME_RE.match(base)
    if not m:
        return {
            "generator": None,
            "speaker_id": None,
            "sentence_id": None,
            "system": None,
            "label": None,
            "voice_type": "Unknown",
        }
    gen = m.group("generator").lower()
    return {
        "generator": gen,
        "speaker_id": m.group("speaker"),
        "sentence_id": m.group("sentence"),
        "system": None,
        "label": None,
        "voice_type": "Real" if gen == "natural" else "Fake",
    }


def load_protocols(protocols_dir=PROTOCOLS_DIR):
    """Carica e concatena tutti i CSV di protocols (train/val/test) in un unico
    DataFrame indicizzato per filename. Ritorna un DataFrame vuoto (colonne corrette,
    zero righe) se la cartella non esiste, così il merge a valle resta un no-op sicuro.
    """
    columns = ["filename", "system", "label"]
    if not os.path.isdir(protocols_dir):
        return pd.DataFrame(columns=columns)

    frames = []
    for entry in sorted(os.listdir(protocols_dir)):
        if entry.endswith("_metadata.csv"):
            frames.append(pd.read_csv(os.path.join(protocols_dir, entry)))

    if not frames:
        return pd.DataFrame(columns=columns)

    protocols = pd.concat(frames, ignore_index=True)
    protocols["filename"] = protocols["filename"].apply(lambda x: os.path.basename(str(x)))
    # Alcuni file compaiono in più split se il dataset è stato ricombinato: teniamo
    # la prima occorrenza, l'etichettatura è comunque identica per lo stesso filename.
    return protocols.drop_duplicates(subset="filename", keep="first")


def attach_metadata(df, protocols_dir=PROTOCOLS_DIR):
    """Arricchisce df (che deve avere una colonna 'filename') con generator,
    speaker_id, sentence_id, system, label, voice_type.

    Priorità: CSV di protocols. Per i file assenti dai CSV, fallback sul parsing
    del nome file (utile per dataset locali di test senza protocols/).
    """
    df = df.copy()
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(str(x)))

    # 1. Parsing del nome file: fornisce sempre generator/speaker_id/sentence_id
    #    (servono per l'accoppiamento reale<->sintetico anche quando system/label
    #    arrivano dal CSV).
    parsed = df["filename"].apply(lambda f: pd.Series(_parse_filename(f)))
    df = pd.concat([df, parsed], axis=1)

    # 2. Merge con i protocols ufficiali: system/label/voice_type del CSV
    #    sovrascrivono quelli (assenti) del parsing, dove disponibili.
    protocols = load_protocols(protocols_dir)
    if not protocols.empty:
        protocols = protocols.rename(columns={"system": "system_csv", "label": "label_csv"})
        df = df.merge(protocols[["filename", "system_csv", "label_csv"]], on="filename", how="left")

        has_csv = df["system_csv"].notna()
        df.loc[has_csv, "system"] = df.loc[has_csv, "system_csv"]
        df.loc[has_csv, "label"] = df.loc[has_csv, "label_csv"]
        df.loc[has_csv, "voice_type"] = df.loc[has_csv, "label_csv"].apply(
            lambda lbl: "Real" if int(lbl) == 0 else "Fake"
        )
        df = df.drop(columns=["system_csv", "label_csv"])

    n_unknown = int((df["voice_type"] == "Unknown").sum())
    if n_unknown:
        print(f"⚠️  {n_unknown} file non etichettabili (né da protocols né dal nome file).")
        sample = df.loc[df["voice_type"] == "Unknown", "filename"].head(3).tolist()
        if sample:
            print(f"    Esempi: {sample}")

    return df
