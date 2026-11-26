"""Scam-language detector for call transcripts (English, Hindi/Hinglish, Telugu).

Explainable rules, not a black box: each phrase belongs to a tactic (asking for an OTP, threatening
arrest, ...) with a weight. A call's scam score combines the tactics seen so far (noisy-OR), so one
neutral mention ("transfer") stays low while typical combinations ("bank" + "account blocked" + "share
the OTP") quickly approach 1. Patterns cover Latin script (English and romanised Hindi/Telugu, as
Whisper often writes Hinglish) and Devanagari / Telugu script.
"""
import re
import unicodedata

# tactic: (weight, label shown to the user, English/romanised regexes, Indic-script substrings)
TACTICS: dict[str, tuple[float, str, list[str], list[str]]] = {
    "credentials": (0.6, "asks for OTP / PIN / password", [
        r"\bo\.?\s?t\.?\s?p\b", r"one[- ]time password", r"\bcvv\b", r"\bpin\b", r"\bupi pin\b", r"pass ?word",
        r"verification code", r"card (number|details)", r"expiry date", r"otp (batao|bataiye|bata do|share|bhejo)",
        r"\bnet ?banking (id|password)", r"login details"],
        ["ओटीपी", "ओ टी पी", "पिन नंबर", "पासवर्ड", "सीवीवी", "कार्ड नंबर", "ఓటీపీ", "ఓటిపి", "ఓ టి పి", "పిన్", "పాస్వర్డ్", "పాస్‌వర్డ్", "కార్డ్ నంబర్"]),
    "remote_access": (0.5, "asks to install a remote-access app", [
        r"any ?desk", r"team ?viewer", r"quick ?support", r"screen ?shar", r"\bapk\b",
        r"(install|download) (this|the|an?) app", r"app (install|download) (karo|kijiye|kar lo)"],
        ["एनीडेस्क", "ऐप डाउनलोड", "ऐप इंस्टॉल", "స్క్రీన్ షేర్", "యాప్ డౌన్లోడ్", "యాప్ డౌన్‌లోడ్"]),
    "payment": (0.4, "asks to send / transfer money", [
        r"\btransfer\b", r"send (the |me )?money", r"\bpay(ment)?\b", r"\bupi\b", r"google ?pay|\bg ?pay\b|phone ?pe|paytm",
        r"processing fee", r"security deposit", r"\brefund\b", r"paise? (bhejo|bhej do|bhejiye|transfer)", r"paisa bhej",
        r"dabbu (pampandi|pampu)"],
        ["ट्रांसफर", "पैसे भेज", "पैसा भेज", "पेमेंट", "भुगतान", "यूपीआई", "जमा कर", "డబ్బు పంప", "డబ్బులు పంప",
         "బదిలీ", "ట్రాన్స్ఫర్", "ట్రాన్స్‌ఫర్", "చెల్లించ", "యూపీఐ", "పేమెంట్"]),
    "authority_threat": (0.4, "threatens police / arrest / legal action", [
        r"\bpolice\b", r"\bcbi\b", r"crime branch", r"\bcustoms\b", r"income tax", r"enforcement directorate",
        r"digital arrest", r"\barrest", r"\bwarrant\b", r"\bcourt\b", r"case (has been |is )?(registered|filed)",
        r"money laundering", r"\bparcel\b", r"\bdrugs?\b", r"\bfir\b", r"giraftar"],
        ["पुलिस", "गिरफ्तार", "गिरफ़्तार", "अरेस्ट", "वारंट", "कोर्ट", "सीबीआई", "कस्टम", "पार्सल", "मनी लॉन्ड्रिंग", "एफआईआर",
         "పోలీస్", "పోలీసు", "అరెస్ట్", "అరెస్టు", "వారెంట్", "కోర్టు", "సీబీఐ", "పార్సెల్", "కస్టమ్స్"]),
    "account_threat": (0.35, "says your account / SIM / KYC will be blocked", [
        r"(account|card|sim|wallet)\b.{0,25}\b(block|suspend|frozen|freeze|deactivat|close)",
        r"(block|suspend|freeze|deactivat|close)\w*\b.{0,25}\b(account|card|sim)", r"\bk\.?\s?y\.?\s?c\b",
        r"account band", r"khata band"],
        ["खाता बंद", "अकाउंट ब्लॉक", "अकाउंट बंद", "एकाउंट ब्लॉक", "केवाईसी", "के वाई सी", "सिम ब्लॉक", "कार्ड ब्लॉक",
         "ఖాతా బ్లాక్", "అకౌంట్ బ్లాక్", "ఖాతా మూసి", "కేవైసీ", "సిమ్ బ్లాక్"]),
    "family_emergency": (0.35, "claims an emergency / accident", [
        r"\baccident\b", r"\bhospital\b", r"in (big )?trouble", r"need money (urgently|now|fast)", r"\bkidnap",
        r"bail\b", r"musibat"],
        ["एक्सीडेंट", "दुर्घटना", "अस्पताल", "हॉस्पिटल", "मुसीबत", "ప్రమాదం", "యాక్సిడెంట్", "ఆసుపత్రి", "హాస్పిటల్"]),
    "reward": (0.3, "promises a prize / cashback / lottery", [
        r"\blotter(y|ies)\b", r"won (a |the )?(prize|car|lottery|reward)", r"\bcash ?back\b", r"reward points", r"\bkbc\b",
        r"\bjackpot\b"],
        ["लॉटरी", "इनाम", "कैशबैक", "లాటరీ", "బహుమతి", "క్యాష్‌బ్యాక్", "క్యాష్ బ్యాక్"]),
    "urgency_secrecy": (0.25, "pressures you to act fast / keep it secret", [
        r"\burgent", r"\bimmediately\b", r"right now", r"within (\d+|one|two|five|ten|fifteen|thirty) (minute|hour)s?",
        r"(don'?t|do not) (tell|inform) (anyone|anybody|your family)", r"keep (this|it) (secret|confidential)",
        r"(don'?t|do not) (disconnect|cut|hang up)", r"stay on the line", r"\bturant\b", r"\bjaldi\b",
        r"kisi ko (mat|na) bata", r"phone (mat|na) kat"],
        ["तुरंत", "अभी के अभी", "जल्दी", "किसी को मत बता", "किसी को न बता", "फोन मत काट", "फ़ोन मत काट",
         "వెంటనే", "త్వరగా", "ఎవరికీ చెప్ప", "ఫోన్ కట్ చేయ"]),
    "impersonation": (0.2, "claims to be a bank / official", [
        r"bank manager", r"from (the |your )?(bank|rbi|reserve bank)", r"customer (care|support|service)",
        r"\b(rbi|sbi|hdfc|icici|axis bank|kotak)\b", r"relationship manager", r"fraud (department|team|cell)",
        r"(telecom|trai|cyber ?cell)"],
        ["बैंक से", "बैंक मैनेजर", "कस्टमर केयर", "आरबीआई", "एसबीआई", "బ్యాంక్ నుంచి", "బ్యాంకు నుంచి", "బ్యాంక్ మేనేజర్", "కస్టమర్ కేర్"]),
}

# Hinglish phrasings Whisper writes for common scam lines (added 26 Nov 2026; same tactics and weights)
TACTICS["credentials"][2].extend([r"otp (bata|bol|de)\w* (do|dijiye|dena|dijie)", r"jo otp aaya", r"code (bata|bol)\w*"])
TACTICS["payment"][2].extend([r"(paise|paisa|amount) (transfer|bhej)\w*", r"(google pay|gpay|phone ?pe|paytm) kar (do|dijiye|dena)"])
TACTICS["account_threat"][2].extend([r"(account|khata) band ho (jayega|jaega|jaayega)"])
TACTICS["urgency_secrecy"][2].extend([r"abhi ke abhi", r"(jaldi|turant) (karo|kijiye|kariye)", r"\b(mat|na) batana\b"])

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v[2]] for k, v in TACTICS.items()}

# ---------------------------------------------------------------- context layer (optional, config "scam_context")
# A genuine bank says "we will never ask for your OTP" or "OTP kisi ko mat batayein". The keyword rules alone count
# that as "asks for OTP". The context layer drops credential (and secrecy) matches that occur only inside such
# protective advice, clause by clause, so "we never ask for your PIN, but share the OTP now" still counts.
_CLAUSE = re.compile(r"[.?!;,।|]|\bbut\b|\blekin\b|\bmagar\b|\bhowever\b")
_PROTECT = [re.compile(p) for p in (
    r"\b(never|don'?t|do not|dont|must not|should not|shouldn'?t|won'?t|will not|no one|nobody|not)\s+(\w+\s+){0,3}?"
    r"(share|tell|give|disclose|reveal|send|ask|asks|asking)\b",
    r"\b(kisi ko|kisi se|kabhi( bhi)?)\b.{0,25}\b(mat|na|nahi|nahin)\b",
    r"\b(mat|na|nahi|nahin)\s+(batay\w*|batai\w*|batana|share kar\w*|de\w*)\b.{0,10}$",
)]
_PROTECT_INDIC = [("किसी", ("मत", " न ", "नहीं", "न करें")), ("कभी", ("मत", " न ", "नहीं")),
                  ("చెప్పకండి", ("",)), ("చేయవద్దు", ("",)), ("ఇవ్వకండి", ("",)), ("షేర్ చేయకండి", ("",))]
_SUPPRESSIBLE = ("credentials", "urgency_secrecy")

# named fraud scripts: explanations only, they never change the score
PATTERNS = [
    ("Fake bank / KYC script", "impersonation", ("credentials", "account_threat", "remote_access")),
    ("Digital-arrest script", "authority_threat", ("payment", "urgency_secrecy", "credentials")),
    ("Family-emergency script", "family_emergency", ("payment", "urgency_secrecy")),
    ("Prize / refund script", "reward", ("payment", "credentials")),
    ("Remote-access takeover", "remote_access", ("credentials", "payment")),
]


def _protective(clause: str) -> bool:
    if any(rx.search(clause) for rx in _PROTECT):
        return True
    return any(a in clause and any(b in clause for b in bs) for a, bs in _PROTECT_INDIC)


def _matches(name: str, text: str) -> list[str]:
    return [m.group(0) for rx in _COMPILED[name] for m in [rx.search(text)] if m] + [s for s in TACTICS[name][3] if s in text]


def patterns_for(tactics) -> list[str]:
    seen = set(tactics)
    return [label for label, anchor, partners in PATTERNS if anchor in seen and seen & set(partners)]


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[“”\"'’‘`]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def find_tactics(text: str, context: bool = False, suppressed: list | None = None) -> list[dict]:
    """All tactic matches in one piece of text: [{"tactic", "label", "weight", "phrase"}].

    context=False is the original keyword engine. context=True additionally drops credential / secrecy matches
    that occur only inside protective advice; dropped matches are appended to `suppressed` for the explanation.
    """
    t = normalize(text)
    hits = []
    kept_text = None
    for name, (weight, label, _, indic) in TACTICS.items():
        found = _matches(name, t)
        if found and context and name in _SUPPRESSIBLE:
            if kept_text is None:
                parts = [c for c in _CLAUSE.split(t) if c and c.strip()]
                kept_text = " | ".join(c for c in parts if not (_protective(c) and _matches("credentials", c)))
            if not _matches(name, kept_text):
                if suppressed is not None:
                    suppressed.append({"tactic": name, "label": label, "phrase": found[0],
                                       "reason": "protective advice (e.g. 'never share your OTP')"})
                continue
            found = _matches(name, kept_text)
        if found:
            hits.append({"tactic": name, "label": label, "weight": weight, "phrase": found[0]})
    return hits


def combine(weights) -> float:
    p = 1.0
    for w in weights:
        p *= 1.0 - w
    return 1.0 - p


class CallScam:
    """Scam evidence accumulated over one call."""

    def __init__(self, context: bool = False):
        self.tactics: dict[str, dict] = {}
        self.context = context

    def add(self, text: str) -> dict:
        new, suppressed = [], []
        hits = find_tactics(text, self.context, suppressed)
        for h in hits:
            if h["tactic"] not in self.tactics:
                self.tactics[h["tactic"]] = h
                new.append(h)
        return {"hits": hits, "new_tactics": new, "score": self.score,
                "tactics": [v["label"] for v in self.tactics.values()],
                "patterns": patterns_for(self.tactics), "suppressed": suppressed}

    @property
    def score(self) -> float:
        return combine(v["weight"] for v in self.tactics.values())
