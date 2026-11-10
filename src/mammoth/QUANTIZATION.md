# Warum quantisiertes Mammoth für den Schema-Linking-Test wenig Sinn macht

Stand: 2026-11-09. Betrifft Mammoth-1 (27B) auf der V100 (`gpu-v100s-01`, 32 GB, sm_70), wo das Skript
automatisch auf 4-bit NF4 + float16 ausweicht (`mammoth_decider.py`, `--quant auto`).

**Status der Aussagen:** Die Begründung unten ist hergeleitet, nicht gemessen. Einen direkten Vergleich
NF4 gegen BF16 auf denselben Fragen gibt es noch nicht (siehe "Wie man es belegt").

## Kurzfassung

Der Test misst Precision/Recall/F6 über Schwellen auf den Wahrscheinlichkeiten des Modells. Genau diese
Wahrscheinlichkeiten werden durch die Quantisierung verändert, und die V100 verzerrt zusätzlich die Zeitmessung.
Zahlen aus dem NF4-Lauf sind deshalb nur eingeschränkt mit Jev, mit dem unquantisierten Mammoth und mit ExSL vergleichbar.

## Gründe

1. **Der Wert des Modells sind kalibrierte Wahrscheinlichkeiten.**
   Mammoth liefert pro Option eine Wahrscheinlichkeit, die wir gegen Schwellen (0.1 bis 0.9) auswerten.
   Das Schwestermodell Elephant-1 wurde laut Modellkarte mit Cross-Entropy plus Brier-Kalibrierungsterm trainiert;
   für Mammoth gilt nach unseren Notizen ("kalibrierte Wahrscheinlichkeiten pro Option") dasselbe Prinzip.
   NF4 rundet die Gewichte auf 4 Bit. Das verschiebt die Logits leicht und damit die Kalibrierung. Eine kleine
   Verschiebung genügt, damit eine Schwelle wie 0.3 (Stufe 1) oder 0.5 (Stufe 2) andere Kandidaten durchlässt.

2. **Abweichung vom vorgesehenen Betrieb.**
   Vorgesehen ist BF16 ohne Quantisierung (so läuft das Modell bei Baseten und auf A100/H100). Die V100 hat kein
   natives BF16, deshalb rechnen wir in float16 mit NF4-Gewichten. Gewichte und Rechenformat unterscheiden sich
   also gleichzeitig vom Training. Fehler lassen sich dann nicht mehr sauber dem Modell zuschreiben.

3. **Schwellen und F6 sind sensibel.**
   F6 gewichtet Recall sehr stark. Stufe 1 (Tabellenauswahl mit Schwelle 0.3) entscheidet, welche Tabellen
   überhaupt in Stufe 2 kommen; eine verpasste Gold-Tabelle ist dort nicht mehr korrigierbar. Kleine
   Wahrscheinlichkeitsverschiebungen können sich so auf das Endergebnis auswirken.

4. **Die Zeitmessung ist nicht übertragbar.**
   - Die Engine meldet auf der V100 `batched branches out of memory at prefix 3423 tokens; retrying with chunk
     29 / 14`: Bei vielen Fragen pro Request reicht der Speicher nicht für die parallelen Branches, die Engine
     halbiert und wiederholt. Das kostet Zeit, die auf einer 80-GB-GPU nicht anfällt.
   - `causal_conv1d` hat keine sm_70-Kernel; transformers nutzt den langsameren PyTorch-Fallback.
   - Dequantisierung der NF4-Gewichte kostet zusätzlich Rechenzeit.
   Die Sekunden pro Frage von der V100 sagen daher wenig über den realen Betrieb.

5. **Kein Vergleich mit der Referenz möglich.**
   Jev und die Baseten-Läufe von Mammoth sind unquantisiert. Ein Unterschied zwischen "Mammoth NF4" und Jev
   wäre eine Mischung aus Modellunterschied und Quantisierungseffekt.

## Was der NF4-Lauf trotzdem taugt

- Funktionstest: Pipeline, Chunking, Antwortformat und Auswertung lassen sich auf der V100 prüfen.
- Grobe Untergrenze: Zeigt, was das 27B-Modell auf kleiner Hardware leisten kann.
- Ergebnisdateien tragen `_nf4` im Namen, damit sie nicht mit unquantisierten verwechselt werden.

## Empfehlung

- Zahlen für den Vergleich mit Jev/ExSL: unquantisiert auf H100 (`gpu-h100-bil`) oder A100-80GB,
  `--quant auto` wählt dort BF16 ohne Quantisierung. A100 mit 40 GB reicht für BF16 nicht (Gewichte ca. 54 GB).
- Zeitmessung ebenfalls nur auf der H100/A100.
- V100-Läufe nur als gekennzeichneter NF4-Referenzpunkt, nicht als Mammoth-Ergebnis.
- Elephant-1 (9B, ca. 19 GB) läuft auf der V100 ohne Quantisierung (`--quant auto`); das Problem betrifft
  nur Mammoth.

## Wie man es belegt

Dieselben 30 Fragen (Seed 42) einmal auf der H100 unquantisiert und einmal mit `--quant nf4` laufen lassen
(`bash run_mammoth_test.sh --quant nf4` erzwingt NF4 auch dort). Dann vergleichen:

- F6/Precision/Recall pro Schwelle,
- Tabellen-Recall in Stufe 1,
- Differenz der Wahrscheinlichkeiten pro Kandidat (`column_scores` in den Ergebnis-JSONs).

Erst dieser Vergleich zeigt, wie groß der Quantisierungseffekt tatsächlich ist. Er könnte auch klein ausfallen;
dann wäre die V100 für Mammoth brauchbarer, als diese Notiz annimmt.
