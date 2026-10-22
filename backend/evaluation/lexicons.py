"""Cue lexicons used by the rubric evaluator.

These are general-purpose word/phrase categories in the spirit of LIWC-style dictionaries,
written for the evaluator only (the evaluator never imports the corpus generator). Each
entry is a regex fragment matched on lower-cased text at word boundaries.

LIMITATION (documented in the README): lexicon scoring measures *expressed* cues, not
meaning. It cannot detect sarcasm, and a response that uses the right words incoherently
can still score well. Human evaluation (section 31) exists to validate it.
"""

TRAIT_CUES: dict[str, dict[str, list[str]]] = {
    "openness": {
        "high": [r"new (things|ideas|approach\w*|options?)", r"curious", r"explor\w*", r"experiment\w*", r"creativ\w*",
                 r"try (it|a|something|new)", r"give it a try", r"open to", r"excit\w*", r"imagin\w*", r"innovat\w*",
                 r"mix it up", r"change (doesn'?t|does not) scare", r"different options", r"test (it|a)"],
        "low": [r"stick (with|to) what", r"what (i know|works)", r"familiar", r"old way",
                r"not (big )?(on|into) (new|trends|gyms)", r"no reason to change", r"routine", r"same as always",
                r"conventional", r"tried and true", r"don'?t see a reason"],
    },
    "conscientiousness": {
        "high": [r"plan(s|ned|ning)?\b", r"budget\w*", r"organi[sz]\w*", r"(?<!personal )responsib\w*", r"careful\w*", r"disciplin\w*",
                 r"homework", r"research\w*", r"check (the )?details", r"emergency fund", r"pay (off|down)",
                 r"debt", r"save\b|saving\w*|savings", r"stick to it", r"step by step", r"thorough\w*", r"first i"],
        "low": [r"planning ahead (isn'?t|is not)( really)? my thing", r"go with my gut", r"on the spot", r"as i go",
                r"don'?t (really )?plan", r"spontaneous\w*", r"wing it", r"whatever happens", r"not really my thing",
                r"impulse\w*", r"just keep paying", r"why not", r"life is short"],
    },
    "extraversion": {
        "high": [r"friends", r"talk (it )?over", r"tell everyone", r"post about", r"social\b", r"together",
                 r"party", r"people around", r"group", r"everyone thinks", r"get people", r"out loud",
                 r"comment (and|publicly|right)", r"call it out"],
        "low": [r"to myself", r"quietly", r"on my own", r"privately", r"alone", r"don'?t need to share",
                r"keep (it|this) private", r"scroll past", r"ignore it", r"by myself", r"quiet"],
    },
    "agreeableness": {
        "high": [r"understand", r"i respect|respect (that|other|their|people|each)", r"fair to", r"kind(ness|ly)?(?! of)",
                 r"help (others|people)", r"see the best", r"compassion\w*", r"trust (them|people)", r"gently",
                 r"privat\w* message|dm them kindly", r"family helps", r"cooperat\w*", r"empath\w*"],
        "low": [r"don'?t trust", r"skeptic\w*|sceptic\w*", r"question their", r"don'?t care (much )?what",
                r"scam", r"(they'?re|you'?re|they are) wrong", r"hype", r"motives", r"no thanks", r"not buying",
                r"call it out", r"frankly", r"waste\w*"],
    },
    "neuroticism": {
        "high": [r"anxi\w*", r"worr\w*", r"stress\w*", r"nervous\w*", r"afraid", r"scar(ed|y)", r"panic\w*",
                 r"what could go wrong", r"overwhelm\w*", r"fear\w*", r"upset", r"uneasy", r"wrong choice"],
        "low": [r"calm\w*", r"relax\w*", r"not worried", r"doesn'?t stress", r"take it as it comes",
                r"no big deal", r"easygoing", r"stable", r"at ease", r"fine with", r"(don'?t|do not) worry"],
    },
}

POLITICAL_CUES = {
    # cues of a left-leaning position
    "left": [r"living wage", r"inequalit\w*", r"fair share", r"workers?\b", r"working people", r"corporations?",
             r"wall street", r"wealthy|the rich", r"climate (action|change is real|crisis)", r"clean energy",
             r"public health", r"affordable for everyone", r"diversity", r"include everyone|inclusive",
             r"progress\w*", r"long overdue", r"protect the planet", r"accountab\w*", r"welcome more",
             r"path to citizenship", r"common sense", r"stricter", r"science", r"dignity", r"community",
             r"support raising", r"we must act", r"strong climate", r"fewer rights", r"invest in our"],
    # cues of a right-leaning position
    "right": [r"personal responsibility", r"self[- ]relian\w*", r"government overreach", r"free market",
              r"small business\w*", r"taxes are (already )?too high", r"no new taxes", r"cut spending",
              r"tradition\w*", r"faith", r"family values|faith,? family", r"freedom", r"second amendment",
              r"secure (the )?border", r"enforce the law", r"law-abiding", r"censorship", r"free speech",
              r"overhyped", r"wreck the economy", r"hard work", r"government telling", r"not the government",
              r"big agencies", r"oppose", r"cost jobs", r"washington", r"made this country strong",
              r"amen", r"respect and tradition", r"get back to work", r"living within"],
    "center": [r"both (sides|parties)", r"middle( ground)?", r"compromise", r"it depends", r"balance\w*",
               r"gradual\w*", r"somewhere in", r"practical\w*", r"sensibl\w*", r"room for both", r"hybrid",
               r"some things were better", r"reasonable", r"not extremes"],
}

STANCE_CUES = {
    "support": [r"\byes\b", r"i support", r"absolutely", r"agree\w*", r"of course", r"\bsure\b", r"i'?d (get|take|try|invest|buy|lend|jump)"],
    "oppose": [r"\bno\b", r"i oppose", r"disagree", r"not risk", r"pass\b", r"skip", r"stay away", r"avoid", r"won'?t", r"no thanks"],
}

INTENSIFIERS = [r"absolutely", r"totally", r"completely", r"always", r"never", r"extremely", r"!!+", r"definitely",
                r"no question", r"literally", r"so much", r"the worst", r"the best", r"100%"]

STYLE_KEYWORDS = {
    # communication-style description -> style family
    "terse": ["terse", "blunt", "direct", "brief", "short", "plain", "concise", "no nonsense", "few words", "jargon"],
    "analytical": ["analytical", "data", "hedg", "precise", "qualif", "logical", "numbers", "measured", "cites", "careful with claims"],
    "formal": ["formal", "polite", "proper", "discipline", "respectful", "structured", "patriotic", "avoids slang"],
    "expressive": ["passionate", "informal", "humour", "humor", "energetic", "casual", "funny", "enthusiastic",
                   "expressive", "chatty", "exclamation", "emotion", "social media"],
    "warm": ["warm", "kind", "gentle", "empath", "caring", "friendly", "thoughtful", "balanced", "considerate",
             "patient", "reflective", "cautious"],
}

ANALYTICAL_MARKERS = [r"\bdata\b", r"evidence", r"numbers?", r"roughly", r"on balance", r"trade-?off", r"estimate",
                      r"probabl\w*", r"\bseems?\b", r"i think", r"percent|%", r"studies", r"likely", r"statistic\w*"]
FORMAL_MARKERS = [r"i would\b", r"thank you", r"considered view", r"with respect", r"in my view", r"i do not",
                  r"it is\b", r"i am\b", r"respectfully"]
INFORMAL_MARKERS = [r"\blol\b", r"honestly", r"seriously", r"real talk", r"oh wow", r"\bgonna\b", r"\bkinda\b",
                    r"\bsuper\b", r"\bngl\b", r"\bomg\b"]
WARMTH_MARKERS = [r"understand", r"\bhope\b", r"\bcare\b|caring", r"\bkind\b", r"everyone involved", r"each other",
                  r"people around me", r"\bfeel\b", r"\bokay\b"]
NEGATIONS = {"not", "no", "never", "don't", "dont", "doesn't", "isn't", "wasn't", "won't", "can't", "cannot", "without"}

# Demographic archetypes used to detect *stereotype projection*: what a lazy simulator might
# assume from demographics alone (political 0-100, openness 0-100).
ARCHETYPES = [
    (lambda p: _age(p) >= 65, {"political": 70, "openness": 30}),
    (lambda p: _age(p) and _age(p) <= 28, {"political": 30, "openness": 70}),
    (lambda p: any(w in (p.get("location", "") + " " + p.get("occupation", "")).lower()
                   for w in ("rural", "farm", "kansas", "texas", "montana", "iowa")), {"political": 75}),
    (lambda p: any(w in (p.get("location", "") + " " + p.get("occupation", "")).lower()
                   for w in ("san francisco", "brooklyn", "seattle", "portland", "boston", "software", "artist",
                             "writer")), {"political": 25, "openness": 72}),
    (lambda p: any(w in p.get("occupation", "").lower() for w in ("union", "electrician", "factory", "worker")),
     {"political": 40}),
    (lambda p: any(w in (p.get("occupation", "") + p.get("notes", "")).lower()
                   for w in ("army", "veteran", "military", "navy", "police")), {"political": 75, "conscientiousness": 80}),
]


def _age(p: dict) -> int:
    try:
        return int(float(p.get("age") or 0))
    except ValueError:
        return 0


# Very common English words (function words + frequent verbs/nouns). Used only to decide whether
# a response is readable English at all (validity gate), not for scoring traits.
COMMON_WORDS = set("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its itself just let like me more most my myself
no nor not now of off on once only or other our ours out over own same she should so some such than
that the their them then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours yes ok okay really much many
well think know want need get go make take give see look come say tell feel try use work help keep
pay save spend money time people life family friend friends job business government country world
year years day good bad new old first last long great little right big high low small next own
sure maybe probably still even back way thing things lot always never every something nothing
honestly frankly agree disagree support oppose doctor health tax taxes wage wages price prices
""".split()) | {"i'd", "i'm", "don't", "it's", "i'll", "can't", "won't", "doesn't", "that's", "i've"}
