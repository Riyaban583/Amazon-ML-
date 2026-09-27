import re
import unicodedata
from typing import List, Set, Tuple


LEGAL_SUFFIXES: Set[str] = {
    # US / UK / International
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "l.l.c.", 
    "ltd", "limited", "lp", "llp", "plc", "p.c.", "pc", "group", "holdings", 
    "industries", "enterprises", "enterprise", "associates", "partners", "services",
    # India specific
    "pvt", "pvt.", "private", "p limited", "pvt ltd", "private limited", "traders",
    "trading", "agency", "agencies", "proprietor", "prop", "m/s", "shree", "sri",
    # France specific
    "sa", "s.a.", "sarl", "s.a.r.l.", "sas", "s.a.s.", "sasu", "snc", "eurl", 
    "sci", "ste", "societe", "ets", "etablissement", "cie", "compagnie"
}

ADDRESS_EXPANSIONS = {
    # Street types
    r"\bst\b": "street",
    r"\brd\b": "road",
    r"\bave\b": "avenue",
    r"\bav\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bbld\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bpl\b": "place",
    r"\bsq\b": "square",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bexpy\b": "expressway",
    # Unit / Sub-address
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bflr\b": "floor",
    r"\bbldg\b": "building",
    r"\bdept\b": "department",
    r"\bno\b": "number",
    # Indian address terms
    r"\bopp\b": "opposite",
    r"\bnr\b": "near",
    r"\badj\b": "adjacent",
    r"\bbh\b": "behind",
    r"\bsec\b": "sector",
    r"\bcol\b": "colony",
    r"\brst\b": "rasta",
    r"\bpo\b": "post office",
    # French address terms
    r"\brte\b": "route",
    r"\bchem\b": "chemin",
    r"\bimp\b": "impasse",
    r"\ball\b": "allee",
}

STOPWORDS = {
    "the", "of", "and", "in", "on", "at", "to", "for", "with", "by", "from", 
    "de", "du", "des", "la", "le", "les", "et", "en", "pour"
}


def strip_accents(text: str) -> str:
    """Normalize unicode characters and strip accents (e.g. French é -> e)."""
    if not isinstance(text, str):
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def clean_text_base(text: str) -> str:
    """Base lowercase and basic character normalization."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    # Normalize ampersand
    text = re.sub(r"&", " and ", text)
    # Replace symbols and punctuation with space
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Compress multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_business_name(name: str) -> Tuple[str, List[str]]:
    """
    Cleans business name:
    - Normalizes characters and removes noise punctuation
    - Strips legal entity forms (Pvt Ltd, LLC, Inc, SARL, etc.)
    - Returns cleaned string and distinct informative tokens
    """
    text = clean_text_base(name)
    tokens = text.split()

    filtered_tokens = []
    for token in tokens:
        if token in LEGAL_SUFFIXES:
            continue
        filtered_tokens.append(token)

    cleaned_str = " ".join(filtered_tokens) if filtered_tokens else text
    informative_tokens = [t for t in filtered_tokens if t not in STOPWORDS and len(t) > 1]
    return cleaned_str, informative_tokens


def clean_address(address: str) -> Tuple[str, List[str], Set[str]]:
    """
    Cleans address:
    - Expands common abbreviations (st -> street, opp -> opposite, etc.)
    - Extracts numeric tokens (PIN codes, ZIP codes, building numbers)
    - Returns cleaned string, tokens, and numeric token set
    """
    text = clean_text_base(address)
    
    # Expand abbreviations
    for pattern, replacement in ADDRESS_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text)

    tokens = text.split()
    numbers = set(re.findall(r"\b\d+\b", text))
    informative_tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    
    return text, informative_tokens, numbers
