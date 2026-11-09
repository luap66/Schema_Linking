"""Mammoth-Test auf Spider-Ent, analog zu src/analysis/jev_test.ipynb (zweistufiges Schema Linking).

    Stufe 1: kompaktes Schema (Tabellen, Spalten, FK-Ziele)   -> ein Noul pro Tabelle
    Stufe 2: volle DDL nur der ausgewaehlten Tabellen          -> ein Noul pro Spalte

Laeuft lokal im Mammoth-venv auf dem GPU-Host (MammothDecider aus mammoth_decider.py, kein API-Aufruf).
Starten am besten ueber run_mammoth_test.sh, das auch Installation/Modell prueft.

Chunking (Mammoth: max. ~16k Tokens pro Request, laengere Requests werfen ValueError):
  * Standard: dieselben Chunk-Groessen wie beim Jev-Test (12000 Zeichen kompaktes Schema / 6000 Tokens DDL
    inkl. Kandidatenliste) -> Ergebnisse direkt mit Jev vergleichbar. Dazu kommen pro Request noch die
    Fragen (Name + Instruktion je Kandidat) und die Regeln; das passt mit grossem Abstand in 16k.
  * Sicherheitsnetz: wirft das Modell trotzdem ValueError (Request zu lang), werden die Fragen des
    Requests halbiert und einzeln neu gestellt (gleiche Tabellen, andere Kandidaten). Die Anzahl dieser
    Splits steht im Ergebnis (n_splits). Wird nie gekuerzt.
  * Groessere Chunks (weniger Requests, mehr Kontext pro Request) gehen ueber --stage1-max-chars und
    --stage2-max-tokens, sind dann aber nicht mehr 1:1 mit Jev vergleichbar.
"""
import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # fuer mammoth_decider.py, falls hier abgelegt

from src.data_loaders.spider_ent_data import get_spider_ent_data  # noqa: E402
from src.utils import _extract_fk_targets, _pack_tables_into_chunks  # noqa: E402

# Regeln/Prompts identisch zu jev_test.ipynb
RULES = ("A column counts as referenced if it appears in SELECT, WHERE, a JOIN condition, GROUP BY, "
         "HAVING, ORDER BY, or inside an aggregation function of the SQL query that answers the question. "
         "If a table is used only via * or COUNT(*) and no other column of it is referenced, "
         "its first column counts as referenced.")
TABLE_RULES = ("A table is needed if at least one of its columns is referenced in the SQL query that answers the question, "
               "or if it is needed to join other needed tables. "
               "The schema shown is only one part of a large database; the needed tables may be in another part.")


# ---------------------------------------------------------------- Mammoth-Aufruf

class Mammoth:
    """Duenner Wrapper um MammothDecider: einheitliches Rueckgabeformat, Zeitmessung, Split bei ValueError."""

    def __init__(self, quant=None):
        from mammoth_decider import MammothDecider  # erst hier, damit --help ohne venv funktioniert
        t0 = time.perf_counter()
        self.decider = MammothDecider(quant=quant)  # laedt das Modell (1-3 min)
        self.load_seconds = time.perf_counter() - t0
        self.tokenizer = getattr(self.decider, "tokenizer", None)
        self.n_requests = 0
        self.n_splits = 0
        self.request_seconds = []
        self.request_tokens = []  # gezaehlte Tokens (State + Fragen) je Request, nur mit Tokenizer

    def count_request_tokens(self, state, payload):
        # Naeherung: Rohtext von State + Fragen, ohne Chat-Template/Sonderzeichen des Modells
        text = (json.dumps(state) if isinstance(state, dict) else str(state)) + json.dumps(payload)
        return len(self.tokenizer.encode(text))

    @staticmethod
    def _parse(raw, names):
        """Zieht {name: p} und Usage aus der Decider-Antwort. Akzeptiert {'answers': {k: {'noul': p}}},
        {k: {'noul': p}} und {k: p}."""
        answers = raw.get("answers", raw) if isinstance(raw, dict) else None
        if not isinstance(answers, dict) or not set(names) <= set(answers):
            raise RuntimeError(f"Unerwartetes Antwortformat von MammothDecider.decide: {str(raw)[:500]}")
        probs = {}
        for k in names:
            a = answers[k]
            probs[k] = float(a["noul"] if isinstance(a, dict) else a)
        usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
        return probs, usage.get("input_tokens", 0), usage.get("output_tokens", 0)

    def ask(self, state, questions):
        """questions: {name: instruction}. Gibt (probs, in_tokens, out_tokens) zurueck.
        Zu lange Requests (ValueError) werden ueber die Fragen halbiert."""
        payload = {n: {"type": "noul", "instructions": i} for n, i in questions.items()}
        if self.tokenizer is not None:
            self.request_tokens.append(self.count_request_tokens(state, payload))
        t0 = time.perf_counter()
        try:
            raw = self.decider.decide(state, payload)
        except ValueError:
            if len(questions) < 2:
                raise
            self.n_splits += 1
            items = list(questions.items())
            half = len(items) // 2
            probs, tin, tout = {}, 0, 0
            for part in (items[:half], items[half:]):
                p, i, o = self.ask(state, dict(part))
                probs.update(p)
                tin, tout = tin + i, tout + o
            return probs, tin, tout
        self.request_seconds.append(time.perf_counter() - t0)
        self.n_requests += 1
        return self._parse(raw, list(questions))


# ---------------------------------------------------------------- Pipeline (wie jev_test.ipynb)

def compact_table(t):
    # 'tabelle(spalte1, spalte2) FK -> ziel1, ziel2' - ohne Typen, Backticks und Beispielwerte
    line = f"{t['candidates']['table']}({', '.join(t['candidates']['columns'])})"
    fks = _extract_fk_targets(t['ddl'])
    if fks:
        line += f" FK -> {', '.join(sorted(fks))}"
    return line


def chunk_compact(tables, max_chars):
    """Jev-identisch: greedy nach Zeichen (--no-tokenizer)."""
    chunks, current, size = [], [], 0
    for t in tables:
        line = compact_table(t)
        n = len(line) + 1
        if current and size + n > max_chars:
            chunks.append(current)
            current, size = [], 0
        current.append((t['candidates']['table'], line))
        size += n
    if current:
        chunks.append(current)
    return chunks


# Fester Overhead pro Request (JSON-Klammern, Feldnamen, Chat-Template) - grosszuegig geschaetzt
REQUEST_OVERHEAD_TOKENS = 150
# JSON-Geruest je Noul-Frage: {"type": "noul", "instructions": "..."} plus Trennzeichen
QUESTION_OVERHEAD_TOKENS = 14


def _question_cost(enc, name, instruction):
    # Name steht zweimal im Request: als Schluessel und in der Instruktion
    return enc(name) + enc(instruction) + QUESTION_OVERHEAD_TOKENS


def _pack_by_tokens(costs, fixed, budget):
    """Greedy: costs = [(item, tokens)], Budget = max. Tokens des ganzen Requests inkl. fixed.
    Ein Item, das allein nicht passt, bekommt einen eigenen Chunk (nie zerschneiden)."""
    chunks, current, size = [], [], fixed
    for item, c in costs:
        if current and size + c > budget:
            chunks.append(current)
            current, size = [], fixed
        current.append(item)
        size += c
    if current:
        chunks.append(current)
    return chunks


def chunk_stage1_tokens(tables, question, tokenizer, budget):
    """Stufe 1 nach echten Mammoth-Request-Tokens: kompakte Zeile + Noul-Frage je Tabelle."""
    enc = lambda text: len(tokenizer.encode(text))
    fixed = enc(question) + enc(TABLE_RULES) + REQUEST_OVERHEAD_TOKENS
    costs = []
    for t in tables:
        name, line = t['candidates']['table'], compact_table(t)
        costs.append(((name, line), enc(line) + _question_cost(enc, name, f"Is table {name} needed for the SQL query?")))
    return _pack_by_tokens(costs, fixed, budget)


def chunk_stage2_tokens(tables, question, tokenizer, budget):
    """Stufe 2 nach echten Mammoth-Request-Tokens: volle DDL + Noul-Frage je Spalte."""
    enc = lambda text: len(tokenizer.encode(text))
    fixed = enc(question) + enc(RULES) + REQUEST_OVERHEAD_TOKENS
    costs = []
    for t in tables:
        name = t['candidates']['table']
        cands = [f"{name}.{c}" for c in t['candidates']['columns']]
        c = enc(t['ddl']) + sum(_question_cost(enc, x, f"Is column {x} referenced in the SQL query?") for x in cands)
        costs.append(((t['ddl'], cands), c))
    return [{"ddl": "\n".join(d for d, _ in grp), "candidate_strings": [x for _, cs in grp for x in cs]}
            for grp in _pack_by_tokens(costs, fixed, budget)]


def fk_neighbors(tables, selected):
    neighbors = set()
    for t in tables:
        name = t['candidates']['table'].lower()
        targets = {x.lower() for x in _extract_fk_targets(t['ddl'])}
        if name in selected:
            neighbors |= targets
        if targets & selected:
            neighbors.add(name)
    return neighbors - selected


def run_question(m, q, table_threshold, include_fk_neighbors, args):
    tok = m.tokenizer
    t_start = time.perf_counter()
    splits_before = m.n_splits
    req_before = len(m.request_tokens)
    tok_in = tok_out = 0

    # Stufe 1: Tabellen
    t1 = time.perf_counter()
    table_scores = {}
    if tok is not None:
        stage1_chunks = chunk_stage1_tokens(q['db_schema'], q['question'], tok, args.max_request_tokens)
    else:
        stage1_chunks = chunk_compact(q['db_schema'], args.stage1_max_chars)
    for chunk in stage1_chunks:
        probs, i, o = m.ask(
            {"question": q['question'], "schema": "\n".join(line for _, line in chunk), "rules": TABLE_RULES},
            {name: f"Is table {name} needed for the SQL query?" for name, _ in chunk})
        table_scores.update(probs)
        tok_in, tok_out = tok_in + i, tok_out + o
    stage1_seconds = time.perf_counter() - t1

    selected = {t.lower() for t, p in table_scores.items() if p >= table_threshold}
    if include_fk_neighbors:
        selected |= fk_neighbors(q['db_schema'], selected)
    selected_tables = [t for t in q['db_schema'] if t['candidates']['table'].lower() in selected]

    # Stufe 2: Spalten der ausgewaehlten Tabellen
    t2 = time.perf_counter()
    col_scores = {}
    if tok is not None:
        stage2_chunks = chunk_stage2_tokens(selected_tables, q['question'], tok, args.max_request_tokens)
    else:
        stage2_chunks = _pack_tables_into_chunks(selected_tables, q['question'], args.stage2_max_tokens, q['db'])
    for chunk in stage2_chunks:
        probs, i, o = m.ask(
            {"question": q['question'], "schema": chunk['ddl'], "rules": RULES},
            {c: f"Is column {c} referenced in the SQL query?" for c in chunk['candidate_strings']})
        col_scores.update(probs)
        tok_in, tok_out = tok_in + i, tok_out + o
    stage2_seconds = time.perf_counter() - t2

    return {"question": q['question'], "db_id": q['db'], "sql": q['sql'], "gold_schema": q['gold_schema'],
            "table_scores": table_scores, "selected_tables": sorted(selected),
            "n_chunks_stage1": len(stage1_chunks), "n_chunks_stage2": len(stage2_chunks),
            "n_splits": m.n_splits - splits_before,
            "max_request_tokens": max(m.request_tokens[req_before:], default=None),
            "seconds": time.perf_counter() - t_start,
            "seconds_stage1": stage1_seconds, "seconds_stage2": stage2_seconds,
            "usage": {"input_tokens": tok_in, "output_tokens": tok_out},
            "column_scores": col_scores}


# ---------------------------------------------------------------- Eval (wie jev_test.ipynb)

def _gold_cols(gold):
    return {(t.lower(), c.lower()) for t, cols in gold.items() for c in (cols or []) if t and c}


def _scores(r):
    return {tuple(k.lower().split(".", 1)): p for k, p in r["column_scores"].items()}


def eval_results(results, threshold=0.5, beta=6):
    tp = n_pred = n_gold = perfect = 0
    for r in results:
        gold = _gold_cols(r["gold_schema"])
        pred = {k for k, p in _scores(r).items() if p >= threshold}
        tp += len(pred & gold)
        n_pred += len(pred)
        n_gold += len(gold)
        perfect += gold <= pred
    p = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    b2 = beta ** 2
    f = (1 + b2) * p * rec / (b2 * p + rec) if (b2 * p + rec) else 0.0
    return {"precision": p, "recall": rec, f"f{beta}": f, "perfect_recall_rate": perfect / len(results),
            "avg_pred_cols": n_pred / len(results)}


def stage1_table_recall(results, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5)):
    print(f"{'thr':>5} {'R_tab':>7} {'R=1':>7} {'#tab':>7}")
    for thr in thresholds:
        hit = total = n_sel = all_found = 0
        for r in results:
            gold = {t.lower() for t in r['gold_schema'] if t}
            sel = {t.lower() for t, p in r['table_scores'].items() if p >= thr}
            hit += len(gold & sel)
            total += len(gold)
            n_sel += len(sel)
            all_found += gold <= sel
        print(f"{thr:>5.2f} {hit / total:>7.4f} {all_found / len(results):>7.4f} {n_sel / len(results):>7.2f}")


def present_results(results):
    print(f"Fragen: {len(results)}")
    missing = sum(len(_gold_cols(r["gold_schema"]) - set(_scores(r))) for r in results)
    print(f"Gold-Spalten ohne Kandidat: {missing}")
    print(f"Tokens: input={sum(r['usage']['input_tokens'] for r in results):,}  "
          f"output={sum(r['usage']['output_tokens'] for r in results):,}\n")
    print(f"{'thr':>5} {'P':>7} {'R':>7} {'F6':>7} {'R=1':>7} {'#pred':>7}")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        m = eval_results(results, thr)
        print(f"{thr:>5.2f} {m['precision']:>7.4f} {m['recall']:>7.4f} {m['f6']:>7.4f} "
              f"{m['perfect_recall_rate']:>7.4f} {m['avg_pred_cols']:>7.2f}")


def print_timing(results, mammoth):
    n = len(results)
    secs = sorted(r['seconds'] for r in results)
    p95 = secs[min(n - 1, int(0.95 * n))]
    print(f"Chunks/Frage: Stufe 1 = {sum(r['n_chunks_stage1'] for r in results) / n:.2f}, "
          f"Stufe 2 = {sum(r['n_chunks_stage2'] for r in results) / n:.2f}; "
          f"Splits wegen Laenge: {sum(r['n_splits'] for r in results)}")
    print(f"Sekunden/Frage: Mittel = {statistics.mean(secs):.2f}, Median = {statistics.median(secs):.2f}, "
          f"p95 = {p95:.2f}, max = {secs[-1]:.2f}  (Stufe 1 Mittel {statistics.mean(r['seconds_stage1'] for r in results):.2f}, "
          f"Stufe 2 Mittel {statistics.mean(r['seconds_stage2'] for r in results):.2f})")
    if mammoth.request_tokens:
        rt = sorted(mammoth.request_tokens)
        print(f"Tokens/Request (State+Fragen, ca.): Mittel = {statistics.mean(rt):.0f}, p95 = {rt[min(len(rt) - 1, int(0.95 * len(rt)))]}, "
              f"max = {rt[-1]} (Limit ~16384)")
    print(f"Gesamtzeit Fragen: {sum(secs):.1f} s; Modell laden: {mammoth.load_seconds:.1f} s (nicht in Sekunden/Frage enthalten)")
    if mammoth.request_seconds:
        print(f"Sekunden/Request: Mittel = {statistics.mean(mammoth.request_seconds):.2f} ({len(mammoth.request_seconds)} Requests)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=30, help="Anzahl zufaelliger Fragen (Jev-Test: 30); 0 = alle Spider-Ent-Fragen")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--table-threshold", type=float, default=0.3)
    ap.add_argument("--fk-neighbors", action="store_true", help="FK-Nachbarn der gewaehlten Tabellen mit in Stufe 2 nehmen")
    ap.add_argument("--max-request-tokens", type=int, default=8000,
                    help="Chunking mit Tokenizer (Standard): max. Tokens je Request (State + Noul-Fragen), Mammoth-Limit ~16k")
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="Chunking exakt wie im Jev-Test: Stufe 1 nach Zeichen (--stage1-max-chars), Stufe 2 ExSL-Prompt "
                         "mit Schaetzung chars/2.0 (--stage2-max-tokens)")
    ap.add_argument("--stage1-max-chars", type=int, default=12000, help="nur mit --no-tokenizer")
    ap.add_argument("--stage2-max-tokens", type=int, default=6000, help="nur mit --no-tokenizer")
    ap.add_argument("--quant", choices=["auto", "nf4", "none"], default="auto",
                    help="auto: V100 -> NF4, A100/H100 -> unquantisiert; nf4: immer 4-bit; none: nie quantisieren")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results" / "mammoth"))
    ap.add_argument("--name", default=None, help="Dateiname der Ergebnisse (Standard aus sample/seed)")
    ap.add_argument("--resume", action="store_true", help="bereits fertige Fragen aus der .jsonl uebernehmen")
    ap.add_argument("--probe", action="store_true", help="nur eine Testanfrage stellen und die Rohantwort ausgeben")
    args = ap.parse_args()

    if args.probe:
        m = Mammoth(args.quant)
        raw = m.decider.decide({"question": "How many singers are there?", "schema": "singer(id, name)"},
                               {"x": {"type": "noul", "instructions": "Is column singer.id referenced in the SQL query?"}})
        print("Rohantwort:", raw)
        print(f"Laden: {m.load_seconds:.1f} s")
        return

    data = get_spider_ent_data(max_tokens=None)  # input-Chunks werden nicht genutzt, wir chunken pro Frage selbst
    if args.sample:
        data = random.Random(args.seed).sample(data, args.sample)
    m = Mammoth(args.quant)
    if args.no_tokenizer:
        m.tokenizer = None
    # NF4-Ergebnisse im Dateinamen kennzeichnen, damit sie nicht mit unquantisierten verwechselt werden
    suffix = "_nf4" if m.decider.quantized else ""
    name = args.name or (f"spider_ent_mammoth_local_{args.sample}_samples_seed_{args.seed}{suffix}" if args.sample
                         else f"spider_ent_mammoth_local_full{suffix}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{name}.jsonl"

    done = []
    if args.resume and jsonl.exists():
        done = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Resume: {len(done)} Fragen uebernommen")
    elif jsonl.exists():
        jsonl.unlink()

    print(f"Chunking: {'Tokenizer des Modells' if m.tokenizer is not None else 'Schaetzung chars/2.0 (Jev-identisch)'}")
    print(f"Modell geladen in {m.load_seconds:.1f} s; {len(data) - len(done)} Fragen offen")
    results = list(done)
    errors = []
    with open(jsonl, "a", encoding="utf-8") as f:
        for idx, q in enumerate(data[len(done):], start=len(done) + 1):
            try:
                r = run_question(m, q, args.table_threshold, args.fk_neighbors, args)
            except Exception as e:  # z.B. ein Request, der auch nach dem Split zu lang ist -> Frage auslassen, Lauf fortsetzen
                errors.append({"question": q['question'], "db_id": q['db'], "error": repr(e)})
                print(f"[{idx}/{len(data)}] FEHLER (Frage ausgelassen): {e!r}")
                continue
            results.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"[{idx}/{len(data)}] {r['seconds']:.1f}s  chunks {r['n_chunks_stage1']}+{r['n_chunks_stage2']}  {q['question'][:70]}")

    (out_dir / f"{name}.json").write_text(json.dumps(results, indent=4), encoding="utf-8")

    print("\nStufe 1 (Tabellen):")
    stage1_table_recall(results)
    print("\nStufe 2 (Spalten):")
    present_results(results)
    print()
    print_timing(results, m)
    print(f"\nErgebnisse: {out_dir / (name + '.json')}")


if __name__ == "__main__":
    main()
