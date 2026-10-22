"""Persona schema, validation, the six original research personas, and a sampler for
synthetic training personas.

Persona fields are unchanged from the original TwinSim UI:
  age, gender, education, occupation, location,
  openness, conscientiousness, extraversion, agreeableness, neuroticism   (0-100)
  political (0 = far-left ... 100 = far-right), environmental (0-100),
  economic (text), commStyle (text), notes (text)
"""
from __future__ import annotations

import random

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
NUMERIC_FIELDS = TRAITS + ["political", "environmental"]
TEXT_FIELDS = ["age", "gender", "education", "occupation", "location", "economic", "commStyle", "notes"]


class PersonaError(ValueError):
    pass


def validate_persona(p: dict) -> dict:
    """Return a cleaned copy or raise PersonaError with a human-readable reason."""
    if not isinstance(p, dict):
        raise PersonaError("persona must be an object")
    out = {}
    for k in NUMERIC_FIELDS:
        v = p.get(k, 50)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise PersonaError(f"{k} must be a number between 0 and 100")
        if not 0 <= v <= 100:
            raise PersonaError(f"{k} must be between 0 and 100 (got {v})")
        out[k] = int(round(v))
    for k in TEXT_FIELDS:
        v = p.get(k, "")
        if v is None:
            v = ""
        if not isinstance(v, (str, int, float)):
            raise PersonaError(f"{k} must be text")
        v = str(v).strip()
        if len(v) > 300:
            raise PersonaError(f"{k} is too long (max 300 characters)")
        out[k] = v
    if out["age"]:
        try:
            a = int(float(out["age"]))
        except ValueError:
            raise PersonaError("age must be a number")
        if not 13 <= a <= 110:
            raise PersonaError("age must be between 13 and 110")
        out["age"] = str(a)
    out["label"] = str(p.get("label") or "").strip()[:80] or (out["occupation"] or "Custom persona").title()
    if p.get("id"):
        out["id"] = str(p["id"])
    return out


# --------------------------------------------------------------------------- presets
# The six research personas of the original study. They form the *benchmark* persona set
# and are never used as training personas.
RESEARCH_PERSONAS: list[dict] = [
    dict(id="p1", label="Young Urban Progressive", age="23", gender="female", education="bachelor's",
         occupation="barista / aspiring writer", location="Brooklyn, New York",
         openness=82, conscientiousness=42, extraversion=71, agreeableness=76, neuroticism=58,
         political=12, environmental=88, economic="worried about cost of living, supports wealth redistribution",
         commStyle="passionate, informal, uses humour, active on social media",
         notes="attends protests, vegan, student loan debt"),
    dict(id="p2", label="Conservative Rural Farmer", age="57", gender="male", education="high school",
         occupation="soy farmer", location="rural Kansas",
         openness=28, conscientiousness=84, extraversion=38, agreeableness=52, neuroticism=31,
         political=87, environmental=28, economic="self-reliant, distrusts government subsidies",
         commStyle="terse, direct, sceptical of jargon, plain-spoken",
         notes="veteran, lifelong Republican, church-going"),
    dict(id="p3", label="Suburban Moderate Nurse", age="39", gender="female", education="associate degree",
         occupation="registered nurse", location="Columbus, Ohio",
         openness=61, conscientiousness=78, extraversion=52, agreeableness=71, neuroticism=47,
         political=38, environmental=62, economic="practical, debt-averse, modest savings goals",
         commStyle="warm, thoughtful, mildly cautious, balanced",
         notes="two school-age children, concerned about healthcare costs"),
    dict(id="p4", label="Elderly Retired Veteran", age="73", gender="male", education="some college",
         occupation="retired US Army / volunteers", location="San Antonio, Texas",
         openness=24, conscientiousness=88, extraversion=34, agreeableness=47, neuroticism=28,
         political=82, environmental=21, economic="fixed income, suspicious of inflation and spending",
         commStyle="formal, patriotic, concise, values discipline",
         notes="deeply religious, attends church weekly, proud of military service"),
    dict(id="p5", label="Tech Startup Engineer", age="29", gender="non-binary", education="master's degree",
         occupation="software engineer", location="San Francisco, CA",
         openness=88, conscientiousness=68, extraversion=57, agreeableness=64, neuroticism=43,
         political=21, environmental=79, economic="high earner, invests aggressively in index funds",
         commStyle="analytical, qualifies statements, references data, uses hedging language",
         notes="vegan, student loans paid off, cares about AI safety"),
    dict(id="p6", label="Blue-Collar Union Worker", age="46", gender="male", education="vocational training",
         occupation="electrician, IBEW member", location="Detroit, Michigan",
         openness=41, conscientiousness=72, extraversion=56, agreeableness=59, neuroticism=49,
         political=52, environmental=41, economic="union wages, suspicious of Wall Street, owns home",
         commStyle="blunt, working-class idiom, distrusts elites, direct",
         notes="moderate Democrat, voted Trump 2016 then Biden 2020"),
]

DEFAULT_PERSONA = dict(
    label="Default Teacher", age="34", gender="female", education="bachelor's degree",
    occupation="high school teacher", location="suburban Ohio, USA",
    openness=60, conscientiousness=75, extraversion=45, agreeableness=70, neuroticism=40,
    political=35, environmental=65, economic="moderate, worries about rising costs",
    commStyle="thoughtful, moderately verbose, occasionally dry humour",
    notes="has two children aged 8 and 11",
)

# --------------------------------------------------------------------------- sampling pools
OCCUPATIONS = [
    "accountant", "bus driver", "chef", "construction worker", "dentist", "graphic designer",
    "high school teacher", "librarian", "mechanic", "pharmacist", "plumber", "police officer",
    "real estate agent", "retail manager", "small business owner", "social worker", "truck driver",
    "university student", "warehouse worker", "web developer", "veterinarian", "paramedic",
    "data analyst", "farmer", "hairdresser", "insurance agent", "musician", "retired teacher",
    "stay-at-home parent", "lawyer", "marketing manager", "carpenter", "cashier", "nurse",
    "pastor", "factory worker", "journalist", "architect", "barber", "electrician",
]
LOCATIONS = [
    "Phoenix, Arizona", "rural Georgia", "Portland, Oregon", "Tampa, Florida", "Denver, Colorado",
    "small town Iowa", "Seattle, Washington", "Houston, Texas", "Boston, Massachusetts", "rural Montana",
    "Atlanta, Georgia", "Minneapolis, Minnesota", "suburban Chicago", "Nashville, Tennessee",
    "Salt Lake City, Utah", "rural Ohio", "Las Vegas, Nevada", "Raleigh, North Carolina",
    "Pittsburgh, Pennsylvania", "Albuquerque, New Mexico", "Charleston, South Carolina", "Milwaukee, Wisconsin",
]
GENDERS = ["female", "male", "female", "male", "non-binary"]
EDUCATION = ["high school", "some college", "associate degree", "bachelor's degree", "master's degree",
             "vocational training", "doctorate"]
ECONOMIC = {
    "left": ["supports higher taxes on the wealthy", "worried about inequality and rent",
             "thinks wages have not kept up", "supports public healthcare and unions"],
    "center": ["moderate, worries about rising costs", "practical, likes a balanced budget",
               "cautious saver, open to both parties' ideas", "focused on stable prices and jobs"],
    "right": ["self-reliant, wants lower taxes", "distrusts government spending",
              "believes in free markets and small business", "worried about inflation and debt"],
}
NOTES = [
    "has two young kids", "cares for an elderly parent", "loves hiking and camping", "volunteers at a food bank",
    "recently moved to a new city", "plays in a local band", "goes to church on Sundays", "runs marathons",
    "is paying off student loans", "coaches youth soccer", "grows vegetables at home", "recently divorced",
    "first in family to attend college", "served in the navy", "plays video games online", "has a rescue dog",
    "is saving for a house", "reads a lot of history books", "works night shifts", "is learning to code",
]

# Communication-style families (used by the synthetic corpus and understood by the evaluator).
STYLE_PHRASES = {
    "terse": ["terse, direct, plain-spoken", "blunt and brief", "short answers, no nonsense",
              "plain-spoken, few words", "direct, sceptical of jargon"],
    "analytical": ["analytical, references data, hedging language", "precise, qualifies statements",
                   "data-driven, careful with claims", "logical, cites numbers, measured"],
    "formal": ["formal, concise, values discipline", "polite and formal", "proper, respectful, structured",
               "formal tone, avoids slang"],
    "expressive": ["passionate, informal, uses humour", "energetic, casual, lots of emotion",
                   "informal, funny, enthusiastic", "chatty, expressive, uses exclamation marks"],
    "warm": ["warm, thoughtful, balanced", "kind, gentle, empathetic", "caring, reflective, considerate",
             "friendly, patient, thoughtful"],
}


def sample_persona(rng: random.Random) -> tuple[dict, str]:
    """Random synthetic *training* persona (traits independent of demographics) + its style family."""
    def trait():
        # mixture favouring clearly high / clearly low values so the corpus contains strong signals
        r = rng.random()
        if r < 0.4:
            return rng.randint(5, 32)
        if r < 0.8:
            return rng.randint(68, 95)
        return rng.randint(40, 60)

    political = trait()
    lean = "left" if political < 40 else "right" if political > 60 else "center"
    style = rng.choice(list(STYLE_PHRASES))
    p = dict(
        age=str(rng.randint(19, 82)), gender=rng.choice(GENDERS), education=rng.choice(EDUCATION),
        occupation=rng.choice(OCCUPATIONS), location=rng.choice(LOCATIONS),
        openness=trait(), conscientiousness=trait(), extraversion=trait(), agreeableness=trait(),
        neuroticism=trait(), political=political,
        environmental=max(0, min(100, int(100 - political + rng.randint(-25, 25)))),
        economic=rng.choice(ECONOMIC[lean]), commStyle=rng.choice(STYLE_PHRASES[style]),
        notes=rng.choice(NOTES),
    )
    p["label"] = f"{p['occupation'].title()} ({p['age']})"
    return p, style
