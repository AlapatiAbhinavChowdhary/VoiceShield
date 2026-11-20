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

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v[2]] for k, v in TACTICS.items()}


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[“”\"'’‘`]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def find_tactics(text: str) -> list[dict]:
    """All tactic matches in one piece of text: [{"tactic", "label", "weight", "phrase"}]."""
    t = normalize(text)
    hits = []
    for name, (weight, label, _, indic) in TACTICS.items():
        found = [m.group(0) for rx in _COMPILED[name] for m in [rx.search(t)] if m]
        found += [s for s in indic if s in t]
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

    def __init__(self):
        self.tactics: dict[str, dict] = {}

    def add(self, text: str) -> dict:
        new = []
        for h in find_tactics(text):
            if h["tactic"] not in self.tactics:
                self.tactics[h["tactic"]] = h
                new.append(h)
        return {"hits": find_tactics(text), "new_tactics": new, "score": self.score,
                "tactics": [v["label"] for v in self.tactics.values()]}

    @property
    def score(self) -> float:
        return combine(v["weight"] for v in self.tactics.values())
