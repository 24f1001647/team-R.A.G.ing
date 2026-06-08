# -*- coding: utf-8 -*-
"""
Elden Ring Lore RAG — Dataset Parser
=====================================
Parses three data sources into chunked, tagged documents ready for Qdrant ingestion.

Data Sources:
  1. Carian-Archive/Master.html  → EVIDENCE  (actual in-game item descriptions, dialogue)
  2. elden ring lore part 1.txt  → THEORY    (YouTube lore analysis with timestamps)
  3. Carian-Archive GameText XML → EVIDENCE  (raw XML game data, parsed to structured text)

Output: JSON Lines file with chunks tagged as evidence/theory, ready for embedding.
"""

import json
import re
import os
import hashlib
from pathlib import Path
from html.parser import HTMLParser
from dataclasses import dataclass, field, asdict
from typing import Optional


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

@dataclass
class LoreChunk:
    """A single chunk of lore text ready for embedding."""
    id: str                          # unique hash-based id
    text: str                        # the actual text content
    source_type: str                 # "evidence" | "theory" | "inference"
    source_file: str                 # original filename
    category: str                    # e.g. "weapon", "talisman", "npc_dialogue", "lore_analysis"
    item_name: Optional[str] = None  # e.g. "Crimson Amber Medallion"
    item_id: Optional[int] = None    # game internal ID if available
    section: Optional[str] = None    # section/chapter heading
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ─────────────────────────────────────────────
# 1. Parse Master.html (Evidence — In-Game Text)
# ─────────────────────────────────────────────

class MasterHTMLParser(HTMLParser):
    """
    Parses the Master.html file produced by the Carian-Archive parser.
    Structure:
        <h2>FileName.fmg</h2>        → section (AccessoryName, WeaponName, etc.)
        <h3>Item Name [ID]</h3>      → item with name and ID
        <h4>Section NN</h4>          → NPC dialogue section
        <p>description text</p>      → item description / dialogue line
        ### NPC Name [ID]            → NPC header (in TalkMsg sections)
    """
    def __init__(self):
        super().__init__()
        self.chunks: list[LoreChunk] = []
        self.current_section = ""       # h2 — file category
        self.current_item_name = None   # h3 — item name
        self.current_item_id = None     # extracted from [ID]
        self.current_subsection = ""    # h4 — NPC dialogue section
        self.current_texts: list[str] = []
        self.current_tag = ""
        self.in_content = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag in ("h2", "h3", "h4"):
            self.in_content = True
            # Flush previous item if we're starting a new h2 or h3
            if tag in ("h2", "h3"):
                self._flush_item()
        elif tag == "p":
            self.in_content = True

    def handle_endtag(self, tag):
        if tag == "p" and self.in_content:
            self.in_content = False
        elif tag in ("h2", "h3", "h4"):
            self.in_content = False
        self.current_tag = ""

    def handle_data(self, data):
        if not self.in_content:
            return
        text = data.strip()
        if not text:
            return

        if self.current_tag == "h2":
            self._flush_item()
            self.current_section = text.replace(".fmg", "")
            self.current_item_name = None
            self.current_item_id = None
            self.current_texts = []

        elif self.current_tag == "h3":
            # Parse "Item Name [ID]" or "### NPC Name [ID]"
            match = re.match(r'^(?:###\s*)?(.+?)\s*\[(\d+)\]\s*$', text)
            if match:
                self.current_item_name = match.group(1).strip()
                self.current_item_id = int(match.group(2))
            else:
                self.current_item_name = text
                self.current_item_id = None

        elif self.current_tag == "h4":
            self.current_subsection = text

        elif self.current_tag == "p":
            # Check for inline [ID] text pattern (dialogue lines)
            id_match = re.match(r'^\[(\d+)\]\s*(.+)', text)
            if id_match:
                self.current_texts.append(text)
            else:
                self.current_texts.append(text)

    def _flush_item(self):
        """Save the current accumulated item as a LoreChunk."""
        if not self.current_texts:
            return

        full_text = "\n".join(self.current_texts)

        # Skip very short or useless entries
        if len(full_text.strip()) < 10:
            self.current_texts = []
            return

        # Determine category from section name
        category = self._categorize_section(self.current_section)

        # Build the chunk text with context
        if self.current_item_name:
            chunk_text = f"[{self.current_section}] {self.current_item_name}\n{full_text}"
        else:
            chunk_text = f"[{self.current_section}]\n{full_text}"

        chunk = LoreChunk(
            id=_make_id(chunk_text),
            text=chunk_text,
            source_type="evidence",
            source_file="Master.html",
            category=category,
            item_name=self.current_item_name,
            item_id=self.current_item_id,
            section=self.current_section,
            metadata={
                "subsection": self.current_subsection if self.current_subsection else None,
            }
        )
        self.chunks.append(chunk)
        self.current_texts = []

    def _categorize_section(self, section: str) -> str:
        """Map section names to human-readable categories."""
        section_lower = section.lower()
        category_map = {
            "weapon": "weapon",
            "protector": "armor",
            "accessory": "talisman",
            "goods": "item",
            "magic": "spell",
            "arts": "skill",
            "gem": "ash_of_war",
            "npc": "npc",
            "talk": "npc_dialogue",
            "tutorial": "tutorial",
            "loading": "loading_screen",
            "event": "event_text",
            "action": "ui_text",
            "blood": "message",
        }
        for key, cat in category_map.items():
            if key in section_lower:
                return cat
        return "misc"

    def finalize(self):
        """Flush any remaining item."""
        self._flush_item()


def parse_master_html(filepath: str) -> list[LoreChunk]:
    """Parse Master.html into evidence chunks."""
    parser = MasterHTMLParser()
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    parser.feed(content)
    parser.finalize()
    return parser.chunks


# ─────────────────────────────────────────────
# 2. Parse YouTube Lore Transcript (Theory)
# ─────────────────────────────────────────────

def parse_lore_transcript(filepath: str) -> list[LoreChunk]:
    """
    Parse the timestamped YouTube lore transcript.

    Format: Each line has a timestamp prefix like "0:088 seconds" or "1:021 minute, 2 seconds"
    mixed in with the actual text. We:
      1. Strip timestamps
      2. Merge into paragraphs
      3. Split into topic-based chunks using topic markers
      4. Tag as "theory" source type
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip timestamp prefixes from each line
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove timestamp patterns like "0:088 seconds", "1:021 minute, 2 seconds"
        # Pattern: starts with digits:digits then time description
        cleaned = re.sub(
            r'^\d+:\d+[\d]*\s*(?:seconds?|minutes?,?\s*\d*\s*seconds?|hours?,?\s*\d+\s*minutes?,?\s*\d*\s*seconds?)\s*',
            '',
            line
        )
        if cleaned:
            cleaned_lines.append(cleaned)

    # Join all text
    full_text = " ".join(cleaned_lines)

    # Split into logical sections using topic markers
    # These are common section-starting phrases in the transcripts
    topic_markers = [
        r'(?:the\s+)?(?:one\s+great|golden\s+star|primordial\s+era|age\s+of\s+the\s+dragons)',
        r'(?:ancient\s+dragons?|beastmen|fire\s+giants?|war\s+of\s+the\s+giants)',
        r'(?:the\s+crucible|earth\s*tree|erd\s*tree|age\s+of\s+plenty)',
        r'(?:golden\s+order|golden\s+lineage|godfrey|radagon|marica|rennala)',
        r'(?:night\s+of\s+the\s+black\s+knives|shattering|demigods?)',
        r'(?:melina|ranni|miquella|malenia|radahn|rykard|mohg|morgott)',
        r'(?:nox\s+heresy|eternal\s+cit(?:y|ies)|godsk[iy]n|gloam)',
        r'(?:tarnished|elden\s+lord|elden\s+ring|elden\s+beast)',
        r'(?:frenzied\s+flame|three\s+fingers|outer\s+god)',
        r'(?:carian|raya\s+lucaria|academy|sorcery|astrologer)',
    ]

    # Use paragraph-based chunking with overlap
    chunks = _chunk_text_by_paragraphs(
        full_text,
        max_chunk_size=1500,  # ~375 tokens at 4 chars/token
        overlap_size=200,
        source_file=os.path.basename(filepath),
        source_type="theory",
        category="lore_analysis",
    )

    return chunks


def _chunk_text_by_paragraphs(
    text: str,
    max_chunk_size: int = 1500,
    overlap_size: int = 200,
    source_file: str = "",
    source_type: str = "theory",
    category: str = "lore_analysis",
) -> list[LoreChunk]:
    """
    Split text into overlapping chunks at sentence boundaries.
    """
    # Split into sentences (rough but effective)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk_sentences = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_chunk_size and current_chunk_sentences:
            # Save current chunk
            chunk_text = " ".join(current_chunk_sentences)
            chunks.append(LoreChunk(
                id=_make_id(chunk_text),
                text=chunk_text.strip(),
                source_type=source_type,
                source_file=source_file,
                category=category,
                section=_extract_topic_heading(chunk_text),
            ))

            # Overlap: keep last few sentences
            overlap_sentences = []
            overlap_len = 0
            for s in reversed(current_chunk_sentences):
                if overlap_len + len(s) > overlap_size:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_chunk_sentences = overlap_sentences
            current_length = overlap_len

        current_chunk_sentences.append(sentence)
        current_length += sentence_len

    # Don't forget the last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        if len(chunk_text.strip()) > 50:  # skip trivially small leftovers
            chunks.append(LoreChunk(
                id=_make_id(chunk_text),
                text=chunk_text.strip(),
                source_type=source_type,
                source_file=source_file,
                category=category,
                section=_extract_topic_heading(chunk_text),
            ))

    return chunks


def _extract_topic_heading(text: str) -> str:
    """Try to extract a topic heading from the first ~100 chars of a chunk."""
    # Look for common Elden Ring proper nouns and topics
    topic_patterns = {
        r'\b(?:one\s+great)\b': "The One Great",
        r'\b(?:elden\s+beast|elden\s+star)\b': "The Elden Beast",
        r'\b(?:ancient\s+dragon)': "Ancient Dragons",
        r'\b(?:placidusax|placu\s*sax)\b': "Placidusax",
        r'\b(?:farum\s+azula|far\s*m?aula|fire?\s*m[ao]ula|farul?la)\b': "Farum Azula",
        r'\b(?:beastm[ae]n)\b': "Beastmen",
        r'\b(?:fire\s+giant)': "Fire Giants",
        r'\b(?:the\s+crucible)\b': "The Crucible",
        r'\b(?:erd?\s*tree|earth?\s*tree|er\s+tree|UR\s+Trey)\b': "The Erdtree",
        r'\b(?:golden\s+order)\b': "Golden Order",
        r'\b(?:golden\s+lineage)\b': "Golden Lineage",
        r'\b(?:godfrey|horah?\s*loux|hu\b)\b': "Godfrey/Hoarah Loux",
        r'\b(?:radagon|rigan)\b': "Radagon",
        r'\b(?:marica|marika|america)\b': "Queen Marika",
        r'\b(?:rennala|renala|ranala|rala)\b': "Rennala",
        r'\b(?:ranni|rannie|rany)\b': "Ranni",
        r'\b(?:miquella|micha|mika)\b': "Miquella",
        r'\b(?:malenia|melenia|melania)\b': "Malenia",
        r'\b(?:radahn|redan|redon)\b': "Radahn",
        r'\b(?:rykard|reichard)\b': "Rykard",
        r'\b(?:mohg|moog|morg)\b': "Mohg",
        r'\b(?:morgott|morot|margit)\b': "Morgott",
        r'\b(?:godwyn|godwin)\b': "Godwyn",
        r'\b(?:maliketh|malakith|malaketh)\b': "Maliketh",
        r'\b(?:frenzied\s+flame)\b': "Frenzied Flame",
        r'\b(?:three\s+fingers)\b': "Three Fingers",
        r'\b(?:godskin|god-skin|gloam)\b': "Godskin Apostles",
        r'\b(?:shattering)\b': "The Shattering",
        r'\b(?:black\s+knives)\b': "Night of the Black Knives",
        r'\b(?:nox|eternal\s+cit)': "Nox/Eternal Cities",
        r'\b(?:carian|caren|carion)\b': "Carian Dynasty",
        r'\b(?:raya\s+lucaria|real?\s+larian?)\b': "Raya Lucaria",
        r'\b(?:dragon\s+communion)\b': "Dragon Communion",
        r'\b(?:death\s+rite|death\s+bird)': "Death Rites",
        r'\b(?:tarnished)\b': "The Tarnished",
    }

    # Check first 300 characters for topic
    snippet = text[:300].lower()
    for pattern, topic in topic_patterns.items():
        if re.search(pattern, snippet, re.IGNORECASE):
            return topic
    return "General Lore"


# ─────────────────────────────────────────────
# 3. Parse Clean YouTube Transcript (Theory)
# ─────────────────────────────────────────────

def parse_clean_transcript(filepath: str) -> list[LoreChunk]:
    """
    Parse the 'elden ring youtube to text.txt' — a single-line clean transcript.
    Same as lore transcript but without timestamps.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    chunks = _chunk_text_by_paragraphs(
        text,
        max_chunk_size=1500,
        overlap_size=200,
        source_file=os.path.basename(filepath),
        source_type="theory",
        category="lore_analysis",
    )
    return chunks


# ─────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────

def _make_id(text: str) -> str:
    """Generate a deterministic ID from text content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def deduplicate_chunks(chunks: list[LoreChunk]) -> list[LoreChunk]:
    """Remove duplicate chunks based on text similarity."""
    seen_ids = set()
    unique = []
    for chunk in chunks:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            unique.append(chunk)
    return unique


def print_stats(chunks: list[LoreChunk], label: str = ""):
    """Print statistics about parsed chunks."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    print(f"  Total chunks: {len(chunks)}")

    # By source type
    by_type = {}
    for c in chunks:
        by_type.setdefault(c.source_type, []).append(c)
    for t, cs in sorted(by_type.items()):
        print(f"    {t}: {len(cs)} chunks")

    # By category
    by_cat = {}
    for c in chunks:
        by_cat.setdefault(c.category, []).append(c)
    print(f"  Categories:")
    for cat, cs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"    {cat}: {len(cs)}")

    # Text stats
    lengths = [len(c.text) for c in chunks]
    if lengths:
        print(f"  Chunk size — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")
        print(f"  Total text: {sum(lengths):,} characters (~{sum(lengths)//4:,} tokens)")


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def main():
    base_dir = Path(__file__).parent
    output_path = base_dir / "chunks.jsonl"

    all_chunks: list[LoreChunk] = []

    # ── 1. Parse Master.html (Evidence) ──
    master_path = base_dir / "database" / "Carian-Archive" / "Master.html"
    if master_path.exists():
        print(f"[1/3] Parsing {master_path.name}...")
        evidence_chunks = parse_master_html(str(master_path))
        all_chunks.extend(evidence_chunks)
        print_stats(evidence_chunks, "Evidence — In-Game Text (Master.html)")
    else:
        print(f"[1/3] SKIP: {master_path} not found")

    # ── 2. Parse timestamped lore transcript (Theory) ──
    lore_path = base_dir / "elden ring lore part 1.txt"
    if lore_path.exists():
        print(f"\n[2/3] Parsing {lore_path.name}...")
        theory_chunks = parse_lore_transcript(str(lore_path))
        all_chunks.extend(theory_chunks)
        print_stats(theory_chunks, "Theory — Lore Analysis (timestamped)")
    else:
        print(f"[2/3] SKIP: {lore_path} not found")

    # ── 3. Parse clean transcript (Theory) ──
    # NOTE: This file appears to be the same content as file #2 but without timestamps.
    # We parse it but deduplicate later. You may want to skip this if it's truly redundant.
    clean_path = base_dir / "elden ring youtube to text.txt"
    if clean_path.exists():
        print(f"\n[3/3] Parsing {clean_path.name}...")
        clean_chunks = parse_clean_transcript(str(clean_path))
        all_chunks.extend(clean_chunks)
        print_stats(clean_chunks, "Theory — Lore Analysis (clean)")
    else:
        print(f"[3/3] SKIP: {clean_path} not found")

    # ── Deduplicate ──
    before = len(all_chunks)
    all_chunks = deduplicate_chunks(all_chunks)
    after = len(all_chunks)
    if before != after:
        print(f"\n  Deduplication: {before} → {after} chunks (removed {before - after})")

    # ── Final Stats ──
    print_stats(all_chunks, "FINAL — All Sources Combined")

    # ── Write output ──
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    print(f"\n✅ Written {len(all_chunks)} chunks to {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    # ── Show some samples ──
    print(f"\n{'='*60}")
    print("  SAMPLE CHUNKS")
    print(f"{'='*60}")

    # Sample evidence
    evidence = [c for c in all_chunks if c.source_type == "evidence"]
    if evidence:
        sample = evidence[min(50, len(evidence)-1)]  # pick one with lore content
        print(f"\n  📜 EVIDENCE SAMPLE (from {sample.source_file}):")
        print(f"     Category: {sample.category}")
        print(f"     Item: {sample.item_name}")
        print(f"     Text: {sample.text[:200]}...")

    # Sample theory
    theory = [c for c in all_chunks if c.source_type == "theory"]
    if theory:
        sample = theory[min(10, len(theory)-1)]
        print(f"\n  🔮 THEORY SAMPLE (from {sample.source_file}):")
        print(f"     Section: {sample.section}")
        print(f"     Text: {sample.text[:200]}...")


if __name__ == "__main__":
    main()
