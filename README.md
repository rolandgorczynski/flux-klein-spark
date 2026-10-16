# FLUX.2 Klein Multi-LoRA auf DGX Spark

Lokale Anpassung des [Hugging-Face-Spaces](https://huggingface.co/spaces/M3st3rJ4k3l/FLUX.2-Klein-Multi-LoRA-v2).
Quellstand und Herkunft stehen in UPSTREAM.md. Oberfläche: http://192.168.1.224:7861

## Start

```sh
docker compose build
docker compose up -d
docker compose logs -f
```

Benötigt NVIDIA DGX Spark ARM64, NVIDIA Container Toolkit und den kompatiblen
580er-Treiber. Das NVIDIA-Image liefert PyTorch und torchvision für Blackwell;
die Anwendung verwendet eine eigene Python-Umgebung.

## Modellzugriff

Die Anwendung verwendet standardmäßig FLUX.2 Klein 9B. Der Token unter
`~/.cache/huggingface/token` wird ausschließlich zur Laufzeit schreibgeschützt
eingebunden. Keine Tokens in Compose, Image oder Git speichern.

1. Unter https://huggingface.co/black-forest-labs/FLUX.2-klein-9B die
   Modellbedingungen prüfen und den Zugang für das eigene Konto freischalten.
2. Der lokale Token muss den Download dieses freigeschalteten Modells erlauben.
3. In der Oberfläche **Check access** wählen und anschließend generieren.

Das Modell lädt erst bei der ersten Generierung; fehlender Modellzugriff
verhindert den Start der Oberfläche nicht. Fehlgeschlagene Ladeversuche können
erneut versucht werden. Nach Austausch des Token-Inhalts im selben Dateipfad
ist normalerweise kein Neustart nötig; nach atomarem Dateiaustausch den
Container mit `docker compose up -d --force-recreate` neu erstellen.

`MODEL_VARIANT: 9B-KV` in compose.yaml wählt die optionale KV-Variante. Sie
benötigt eigenen Modellzugriff und passende Adapter. 4B-LoRAs und 9B-LoRAs
sind nicht austauschbar.

## Lokaler Betrieb

- Modelle und Adapter: `data/hf-cache/`.
- Generierte Dateien und Gradio-Dateien: `data/output/`.
- Upload- und Prompt-Logging des ursprünglichen Spaces ist entfernt.
- Keine Hugging-Face-ZeroGPU-Abhängigkeit; GPU-Aufrufe laufen lokal und seriell.
- Adapter werden nach Auswahl aktiviert; ohne Auswahl werden geladene Adapter deaktiviert.
- Das Modell nutzt BF16 und VAE-Tiling; keine bitsandbytes-Quantisierung erforderlich.
- Prompt-Verbesserung, Tiefe/Pose und Upscaler laden Zusatzmodelle erst bei Benutzung.

Die Anwendung läuft im LAN ohne Anmeldung. Das erste Laden kann mehrere
Minuten dauern und benötigt mehrere zehn GB freien Speicher. Die Oberfläche
wird auch ohne freigeschaltetes Modell als gesund gemeldet; **Check access**
prüft den gesonderten Modellzugriff. Ein gesunder Webdienst ist kein
Nachweis einer erfolgreichen Bildgenerierung.

## Prüfung

```sh
docker compose run --rm --no-deps --entrypoint python flux -m pip check
docker compose run --rm --no-deps --entrypoint python flux check_runtime.py
```

Die Herkunftslizenz des Anwendungscodes ist laut Space-Metadaten Apache-2.0;
die Lizenzen des Basismodells und einzelner LoRAs gelten zusätzlich.
