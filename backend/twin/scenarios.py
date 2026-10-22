"""Scenario library: 4 domains x several issues x several phrasings.

Each phrasing is assigned to exactly one split:
  * ``benchmark`` : the fixed test set (never used for training). Contains the four
                     original research-study prompts plus one extra prompt per domain.
  * ``validation``: used for validation loss during training.
  * ``train``     : used for the synthetic corpus and for harvesting feedback.
"""
from __future__ import annotations

import re

DOMAINS = {
    "economic": "Economic Decision",
    "political": "Political Opinion",
    "health": "Health Choice",
    "social": "Social Media Reply",
}

# issue id -> (domain, [phrasings])
ISSUES: dict[str, tuple[str, list[str]]] = {
    # ---------------- economic
    "windfall": ("economic", [
        "You have $3,000 in unexpected savings. You can: pay off remaining credit card debt, invest in a stock index fund, put it in a high-yield savings account, or spend it on a vacation. What do you do and why?",
        "You get a $3,000 bonus at work. Do you pay down debt, invest it, save it, or take a trip?",
        "An unexpected $3,000 lands in your account. How would you spend or save it?",
        "You receive a $3,000 tax refund. What do you do with the money?",
        "You win $3,000 in a raffle. Save it, invest it, pay off debt, or treat yourself?",
    ]),
    "price_hike": ("economic", [
        "A subscription service you use every day raises its price by 40%. What do you do?",
        "Your internet provider suddenly raises your bill by 40 percent. How do you respond?",
        "The gym you go to raises membership fees by 40%. Do you stay, switch, or quit?",
        "Your favorite streaming service jumps its price by 40 percent. What now?",
    ]),
    "lend_family": ("economic", [
        "A cousin asks to borrow $1,500 and promises to pay it back in six months. Do you lend it?",
        "Your brother asks you for a $1,500 loan to cover rent. What do you do?",
        "A close relative wants to borrow $1,500 from you. How do you decide?",
    ]),
    "side_business": ("economic", [
        "A friend invites you to co-found a small side business that needs $5,000 of your savings. Do you join?",
        "You could put $5,000 into starting a side hustle with a friend. Would you take the risk?",
        "Would you invest $5,000 of your savings to start a small business on the side?",
    ]),
    "buy_rent": ("economic", [
        "Housing prices are high. Should you keep renting or stretch your budget to buy a home now?",
        "Is it smarter for you to buy a house this year or keep renting? Explain.",
        "You can afford a small down payment. Do you buy a home or keep renting?",
    ]),
    # ---------------- political
    "minimum_wage": ("political", [
        "Should the US federal minimum wage be raised from $7.25 to $17 per hour? Give your honest opinion — who benefits, who loses, and what should happen.",
        "Should the minimum wage be increased to $17 per hour?",
        "What do you think about raising the federal minimum wage to $17?",
        "Congress is debating a $17 minimum wage. Do you support it?",
        "Is a $17 minimum wage good or bad for the country?",
    ]),
    "climate_policy": ("political", [
        "Should the government put a tax on carbon emissions to fight climate change?",
        "Do you support new government rules to cut carbon emissions, even if energy costs rise?",
        "Should the US spend more on clean energy and climate programs?",
        "What should be done about climate change, if anything?",
    ]),
    "gun_policy": ("political", [
        "Should there be universal background checks for all gun sales?",
        "Do you think gun laws in the US should be stricter or looser?",
        "What is your view on new gun control legislation?",
    ]),
    "immigration": ("political", [
        "Should the US increase or decrease the number of immigrants it admits each year?",
        "What should the country do about immigration and the southern border?",
        "Do you think immigration levels should go up or down?",
    ]),
    "social_media_liability": ("political", [
        "Should social media companies be legally liable for misinformation on their platforms?",
        "Should the government regulate what big tech companies allow online?",
        "Do you think tech platforms should be punished for spreading false information?",
    ]),
    # ---------------- health
    "supplement": ("health", [
        "A close friend recommends a popular all-natural supplement they say boosted their energy and immunity. It is not FDA-approved but is widely sold. Do you try it? How do you decide?",
        "Would you use an unapproved supplement if it promised significant health benefits?",
        "A coworker swears by a herbal supplement that is not FDA-approved. Would you take it?",
        "An influencer is selling a natural immunity booster that is not approved by the FDA. Do you buy it?",
    ]),
    "new_medication": ("health", [
        "Your doctor recommends a new medication with mild side effects. How do you respond?",
        "A doctor suggests you start a daily medication that might cause mild side effects. Do you take it?",
        "You are prescribed a new drug with a few possible side effects. What do you do?",
    ]),
    "vaccine": ("health", [
        "A new seasonal vaccine is recommended for adults this fall. Will you get it?",
        "Do you plan to get this year's recommended flu shot? Why or why not?",
        "Your pharmacy offers the updated vaccine for free. Do you get it?",
    ]),
    "exercise_plan": ("health", [
        "Your doctor says you should exercise 30 minutes a day. How do you respond?",
        "You are told your health would improve with a daily workout routine. What do you do?",
        "A friend invites you to train for a 10k run together. Do you join?",
    ]),
    "therapy": ("health", [
        "You have been feeling stressed for weeks. Would you see a therapist?",
        "A friend suggests you talk to a counselor about stress. Do you go?",
        "Work stress is piling up. Would you try therapy or handle it yourself?",
    ]),
    # ---------------- social
    "traditional_values": ("social", [
        'You see this tweet going viral: "America was better off 40 years ago. Bring back traditional values and stop this constant change." Write your reply.',
        'Reply to this tweet: "America was better off 40 years ago. Bring back traditional values."',
        'A post says: "Things were better in the old days. We need traditional values again." Write a reply.',
        'Someone tweets: "Modern life is ruining this country. Go back to tradition." Your reply?',
    ]),
    "friend_opinion": ("social", [
        "A friend posts a political opinion you strongly disagree with. Do you comment, DM them, or scroll past?",
        "Your old classmate shares a political post you think is wrong. What do you do?",
        "A family member posts a political rant on Facebook. Do you respond?",
    ]),
    "viral_climate_post": ("social", [
        'Reply to this viral post: "Climate change is the biggest crisis of our time and we must act now."',
        'Someone tweets: "Climate change is exaggerated by the media." Write your reply.',
        'Reply to this post: "Electric cars are the future, gas cars should be banned."',
    ]),
    "remote_work_post": ("social", [
        'Reply to this tweet: "Remote work has made people lazy. Everyone should be back in the office."',
        'A post says: "Working from home is the best thing that ever happened to workers." Your reply?',
        'Write a reply to: "Young people today do not want to work hard anymore."',
    ]),
    "local_news_post": ("social", [
        'Your town posts: "The city plans to raise property taxes to fund new schools." Write a comment.',
        'Reply to a local post: "Should our town build a new public park or cut taxes?"',
        'Comment on this post: "The city council wants to add bike lanes downtown."',
    ]),
}

BENCHMARK_TEXTS = {
    # the 4 original research-study prompts (unchanged) ...
    ISSUES["windfall"][1][0], ISSUES["minimum_wage"][1][0], ISSUES["supplement"][1][0],
    ISSUES["traditional_values"][1][0],
    # ... plus one more per domain
    ISSUES["price_hike"][1][0], ISSUES["climate_policy"][1][0], ISSUES["new_medication"][1][0],
    ISSUES["friend_opinion"][1][0],
}
VALIDATION_TEXTS = {
    ISSUES["lend_family"][1][0], ISSUES["gun_policy"][1][0], ISSUES["vaccine"][1][0],
    ISSUES["remote_work_post"][1][0],
}


def _slug(issue: str, i: int) -> str:
    return f"{issue}-{i}"


def scenario_library() -> list[dict]:
    out = []
    for issue, (domain, texts) in ISSUES.items():
        for i, t in enumerate(texts):
            split = "benchmark" if t in BENCHMARK_TEXTS else "validation" if t in VALIDATION_TEXTS else "train"
            out.append({"id": _slug(issue, i), "domain": domain, "issue": issue, "text": t, "split": split})
    return out


def scenarios_by_split(split: str) -> list[dict]:
    return [s for s in scenario_library() if s["split"] == split]


def _words(t: str) -> list[str]:
    stop = {"the", "a", "an", "to", "of", "you", "do", "your", "is", "it", "and", "or", "in", "for", "be",
            "what", "should", "would", "this", "that", "on", "with", "are", "how", "why", "if", "about"}
    return [w for w in re.findall(r"[a-z0-9$]+", t.lower()) if w not in stop and len(w) > 2]


def infer_issue(domain: str, text: str) -> str | None:
    """Best-effort mapping of a user-written scenario to a known issue (keyword overlap)."""
    words = set(_words(text))
    best, best_score = None, 0
    for issue, (d, texts) in ISSUES.items():
        if d != domain:
            continue
        vocab = set()
        for t in texts:
            vocab |= set(_words(t))
        score = len(words & vocab)
        if score > best_score:
            best, best_score = issue, score
    return best


def validate_scenario(s: dict) -> dict:
    if not isinstance(s, dict):
        raise ValueError("scenario must be an object")
    text = str(s.get("text") or "").strip()
    domain = str(s.get("domain") or "").strip()
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {list(DOMAINS)}")
    if len(text) < 10:
        raise ValueError("scenario text is too short (min 10 characters)")
    if len(text) > 600:
        raise ValueError("scenario text is too long (max 600 characters)")
    out = {"domain": domain, "text": text, "issue": s.get("issue") or infer_issue(domain, text)}
    if s.get("id"):
        out["id"] = str(s["id"])
    return out
