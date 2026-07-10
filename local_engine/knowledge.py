from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    patterns: tuple[str, ...]
    answer: str


FACTS = (
    Fact((r"\blargest planet\b",), "Jupiter is the largest planet in the Solar System."),
    Fact((r"\bred planet\b",), "Mars is known as the Red Planet."),
    Fact((r"\bcapital of australia\b",), "The capital of Australia is Canberra."),
    Fact((r"\bchemical symbol\b.*\bgold\b", r"\bgold\b.*\bchemical symbol\b"), "The chemical symbol for gold is Au."),
    Fact((r"\bhttp stand",), "HTTP stands for Hypertext Transfer Protocol."),
    Fact((r"\bsecure(?:ly)?\b.*\bhttp\b", r"\bencrypted\b.*\bweb traffic\b"), "HTTPS is HTTP secured with TLS encryption."),
    Fact((r"\bfirst[- ]in[, ]+first[- ]out\b", r"\bfifo\b"), "A queue follows first-in, first-out (FIFO) ordering."),
    Fact((r"\blast[- ]in[, ]+first[- ]out\b", r"\blifo\b"), "A stack follows last-in, first-out (LIFO) ordering."),
    Fact((r"\bphotosynthesis\b",), "Photosynthesis uses sunlight to convert water and carbon dioxide into glucose (sugar), releasing oxygen."),
    Fact((r"\bdocker image manifest\b",), "A Docker image manifest describes an image's config, content layers, and platform metadata."),
    Fact((r"\bbinary representation\b.*\b(?:decimal )?five\b",), "Decimal five is 101 in binary."),
    Fact((r"\bwater\b.*\bchemical formula\b", r"\bchemical formula\b.*\bwater\b"), "The chemical formula for water is H2O."),
    Fact((r"\bdns stand",), "DNS stands for Domain Name System."),
    Fact((r"\bipv4\b.*\b(?:how many|number of) bits\b",), "An IPv4 address contains 32 bits."),
    Fact((r"\bhttps\b.*\bdefault port\b", r"\bdefault port\b.*\bhttps\b"), "The default HTTPS port is 443."),
    Fact((r"\bjson stand",), "JSON stands for JavaScript Object Notation."),
    Fact((r"\bprimary key\b",), "A primary key uniquely identifies each row in a database table."),
    Fact((r"\bbinary search\b.*\btime complexity\b",), "Binary search runs in O(log n) time on sorted data."),
    Fact((r"\bram stand",), "RAM stands for Random Access Memory."),
    Fact((r"\bcpu stand",), "CPU stands for Central Processing Unit."),
)


def lookup_fact(prompt: str) -> str | None:
    text = " ".join(prompt.lower().split())
    for fact in FACTS:
        if any(re.search(pattern, text, re.I) for pattern in fact.patterns):
            return fact.answer
    return None
