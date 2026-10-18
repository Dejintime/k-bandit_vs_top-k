"""Lazy JSONL prompt indexing and deterministic domain routing."""

from collections import Counter, defaultdict
import json
import random


class PromptDataset:
    """Index byte offsets instead of holding the multi-gigabyte JSONL in RAM."""

    def __init__(self, path, domains):
        self.path = str(path)
        self.domains = tuple(domains)
        self.offsets = defaultdict(list)
        with open(self.path, "rb") as stream:
            line_number = 0
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("domain") in self.domains:
                    self._validate(row, line_number)
                    self.offsets[row["domain"]].append((offset, line_number))
        missing = set(self.domains) - set(self.offsets)
        if missing:
            raise ValueError(f"Training data has no rows for domains: {sorted(missing)}")

    @staticmethod
    def _validate(row, line_number):
        if row.get("usage") == "final_evaluation_only":
            raise ValueError(f"Line {line_number}: evaluation-only row in training data")
        messages = row.get("messages")
        valid = (isinstance(messages, list) and messages
                 and messages[-1].get("role") == "user"
                 and all(isinstance(message, dict)
                         and message.get("role") in ("system", "user", "assistant")
                         and isinstance(message.get("content"), str)
                         for message in messages))
        if not valid:
            raise ValueError(f"Line {line_number}: invalid chat messages")

    def counts(self):
        return Counter({domain: len(rows) for domain, rows in self.offsets.items()})

    def read(self, domain, item):
        offset, line_number = self.offsets[domain][item]
        with open(self.path, "rb") as stream:
            stream.seek(offset)
            row = json.loads(stream.readline())
        self._validate(row, line_number)
        return {key: row[key] for key in ("id", "domain", "messages")}


class DomainSampler:
    def __init__(self, dataset, weights, seed):
        self.dataset = dataset
        self.domains = list(weights)
        self.weights = [weights[domain] for domain in self.domains]
        self.rng = random.Random(seed)
        self.order = {}
        self.cursor = {}
        for domain in self.domains:
            self.order[domain] = list(range(len(dataset.offsets[domain])))
            self.rng.shuffle(self.order[domain])
            self.cursor[domain] = 0

    def next_batch(self, batch_size, is_eligible=None, domain=None):
        # One routed teacher per optimizer step keeps the reference runner bounded.
        if domain is None:
            domain = self.rng.choices(self.domains, weights=self.weights, k=1)[0]
        elif domain not in self.domains:
            raise ValueError(f"Unknown scheduled domain: {domain}")
        result = []
        attempts = 0
        while len(result) < batch_size:
            attempts += 1
            if attempts > len(self.order[domain]):
                raise ValueError(f"No eligible {domain} prompts under the configured token limit")
            if self.cursor[domain] == len(self.order[domain]):
                self.rng.shuffle(self.order[domain])
                self.cursor[domain] = 0
            index = self.order[domain][self.cursor[domain]]
            self.cursor[domain] += 1
            row = self.dataset.read(domain, index)
            if is_eligible is None or is_eligible(row):
                result.append(row)
        return domain, result
