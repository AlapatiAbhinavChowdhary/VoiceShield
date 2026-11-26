"""Scam-intent frames (shadow mode): what the caller asks the listener to DO, under WHAT pretext.

Keyword rules recognise known phrases ("share the OTP"); indirect scams avoid them ("read out the verification
number", "deposit the GST so the bonus can be released"). A frame fires when one utterance contains
  a REQUEST to the listener (share / read / pay / deposit / scan / install / buy / bhejo / bataiye / भेजिए ...)
  + a SENSITIVE OBJECT (code / OTP / PIN / money / fee / gift cards / QR / app / paise / कोड ...),
and the conversation so far contains a manipulation PRETEXT: threat, authority, reward, emergency,
urgency or secrecy, or a "mistake" story. Requests inside protective advice ("never share your OTP") are ignored.
Score = 0.5 + 0.15 per extra pretext kind (max 0.95). English, romanised Hindi and Devanagari; Telugu partly.

Status: SHADOW (config scam_semantic). Reported next to the rules, never changes a decision.
"""
import re

from .scam import _CLAUSE, _protective, normalize

REQUEST = re.compile(
    r"\b(share|tell|read( out)?|send|give|pay|transfer|deposit|scan|install|download|enter|type|confirm|buy|move|"
    r"bhej\w*|bata\w*|de do|de dijiye|dijiye|jama kar\w*|daal\w*|kar do)\b|भेज|बता|दीजिए|दीजिये|जमा|పంప|చెప్ప")
OBJECT = re.compile(
    r"\b(code|codes|otp|pin|password|cvv|card number|money|amount|fee|fees|charges?|deposit|payment|savings|rupees?|"
    r"gst|duty|penalty|gift cards?|qr|app|upi|paise|paisa|rupaye|hazaar|lakh|thousand|verification number|test amount)\b|"
    r"ओटीपी|कोड|पैसे|पैसा|हज़ार|हजार|पिन|रुपये|డబ్బు|ఓటీపీ")
PRETEXT = {
    "threat": r"\b(block\w*|suspend\w*|stop\w*|disconnect\w*|cut|arrest\w*|case|fir|penalt\w*|seiz\w*|custody|warrant|legal|"
              r"hirasat|band ho\w*|ruka|virus|stealing)\b|हिरासत|बंद|गिरफ्त|వారెంట్|బ్లాక్",
    "authority": r"\b(bank|officer|police|customs|army|income tax|manager|agency|board|government|cbi|rbi|narcotics|network|"
                 r"department|desk|kyc|aadhaar|courier)\b|बैंक|पुलिस|केवाईसी|బ్యాంక్",
    "reward": r"\b(refund|bonus|prize|lottery|won|eligible|cashback|profit|returns?|approved|maturity|upgrade\w*|selected|"
              r"shortlisted|loan|lagi)\b|इनाम|लॉटरी",
    "emergency": r"\b(accident|hospital|custody|trouble|emergency)\b|अस्पताल|हिरासत|ప్రమాదం",
    "urgency": r"\b(now|right away|immediately|today|tonight|tomorrow|turant|abhi|jaldi|confidential|secret|"
               r"do not tell|dont tell|mat bata\w*|warna)\b|अभी|तुरंत|किसी को मत|వెంటనే",
    "mistake": r"\b(by mistake|wrongly|galti se|wrong account)\b",
}
_PRETEXT = {k: re.compile(v) for k, v in PRETEXT.items()}


class CallFrames:
    def __init__(self):
        self.pretexts: set[str] = set()
        self.requests: list[str] = []
        self.score = 0.0

    def add(self, text: str) -> dict:
        t = normalize(text)
        self.pretexts |= {k for k, rx in _PRETEXT.items() if rx.search(t)}
        for clause in (c for c in _CLAUSE.split(t) if c and c.strip()):
            if _protective(clause):
                continue
            r, o = REQUEST.search(clause), OBJECT.search(clause)
            if r and o:
                self.requests.append(f"{r.group(0)} … {o.group(0)}")
        if self.requests and self.pretexts:
            self.score = max(self.score, min(0.95, 0.5 + 0.15 * (len(self.pretexts) - 1)))
        return {"score": round(self.score, 2), "requests": self.requests[-3:], "pretexts": sorted(self.pretexts), "mode": "shadow"}
