# sender_transformer.py
import re
import email.utils
from email.utils import parseaddr
from typing import Tuple, Optional, Dict, Any

class SenderTransformer:
    @staticmethod
    def transform(from_field: str) -> tuple[str, str]:
        if not from_field:
            return ("", "")

        name, addr = email.utils.parseaddr(from_field)
        addr = (addr or "").strip()

        raw = from_field
        raw = re.sub(r'\s*<[^>]+>\s*', ' ', raw)
        if addr:
            raw = re.sub(re.escape(addr), ' ', raw, flags=re.IGNORECASE)
        raw = raw.strip().strip('"').strip("'")

        raw = re.sub(r'(?<=\w)[._](?=\w)', ' ', raw)
        raw = re.sub(r'(?i)[\s\._-]*via\b[^<\n,;"]*', '', raw)
        raw = re.sub(r'\b\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b', '', raw)
        raw = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?\b', '', raw, flags=re.IGNORECASE)
        raw = re.sub(r'\b(?:am|pm|a\.m\.|p\.m\.)\b', '', raw, flags=re.IGNORECASE)
        raw = ' '.join([tok for tok in re.split(r'\s+', raw) if not re.fullmatch(r'\d+', tok)])
        raw = re.sub(r'[^\w\s]', ' ', raw)
        raw = re.sub(r'\s{2,}', ' ', raw).strip()

        if not raw:
            raw = (name or "").strip().strip('"').strip("'")
            raw = re.sub(r'[^\w\s]', ' ', raw)
            raw = re.sub(r'\s{2,}', ' ', raw).strip()

        if not raw and addr:
            local = addr.split('@', 1)[0]
            local = re.sub(r'\+.*$', '', local)
            raw = local.replace('.', ' ').replace('_', ' ').title()

        return (raw, addr)


class ReceiverTransformer:

    @staticmethod
    def transform(to_field: str) -> Tuple[str, str]:
        if not to_field:
            return "", ""

        if "undisclosed-recipients" in to_field.lower():
            return "undisclosed-recipients", ""

        name, email_addr = parseaddr(to_field)

        name = (name or "").strip().strip('"').strip("'")
        email_addr = (email_addr or "").strip()

        if not name and email_addr:
            name = email_addr.split("@")[0]

        return name, email_addr


class AuthResultsParser:
    def __init__(self, arc_auth_results: Optional[str] = None):
        self.raw = arc_auth_results or ""
        self.text = self.raw.lower()

    def parse(self) -> Dict[str, Any]:
        features = {
            "arc_instance": self._extract_instance(),
            "auth_host": self._extract_auth_host(),

            "spf_result": self._extract_result("spf"),
            "spf_domain": self._extract_spf_domain(),
            "spf_sender_ip": self._extract_ip(),
            "smtp_mailfrom": self._extract_after("smtp.mailfrom="),

            "dkim_result": self._extract_result("dkim"),
            "dkim_domain": self._extract_after("header.d="),
            "dkim_selector": self._extract_after("header.s="),
            "dkim_signature": self._extract_after("header.b="),

            "dmarc_result": self._extract_result("dmarc"),
            "dmarc_policy": self._extract_policy(),
            "header_from": self._extract_after("header.from="),
        }
        return features

    # --- Extractors ---

    def _extract_result(self, keyword: str) -> str:
        match = re.search(rf"{keyword}\s*=\s*([a-z0-9_-]+)", self.text)
        return match.group(1) if match else ""

    def _extract_after(self, prefix: str) -> str:
        match = re.search(rf"{re.escape(prefix)}([^\s;]+)", self.text)
        return match.group(1).strip('";') if match else ""

    def _extract_ip(self) -> str:
        match = re.search(r"designates\s+([\d.]+)\s+as\s+permitted\s+sender", self.text)
        if not match:
            match = re.search(r"client-ip=([\d.]+)", self.text)
        return match.group(1) if match else ""

    def _extract_spf_domain(self) -> str:
        match = re.search(r"domain\s+of\s+([^\s;]+)", self.text)
        if not match:
            match = re.search(r"envelope-from=([^\s;]+)", self.text)
        return match.group(1).strip('"') if match else ""

    def _extract_policy(self) -> str:
        match = re.search(r"policy\s*=\s*([a-z]+)", self.text)
        return match.group(1) if match else ""

    def _extract_instance(self) -> str:
        match = re.search(r"\bi\s*=\s*([0-9]+)", self.text)
        return match.group(1) if match else ""

    def _extract_auth_host(self) -> str:
        # bardziej ogólne: znajduje pierwszy hostname po "arc-authentication-results:"
        match = re.search(r"arc[- ]?authentication[- ]?results:\s*([a-z0-9.\-]+)", self.raw, re.IGNORECASE)
        if not match:
            # fallback — szuka po "authserv-id="
            match = re.search(r"authserv-id=([a-z0-9.\-]+)", self.text)
        return match.group(1) if match else ""
