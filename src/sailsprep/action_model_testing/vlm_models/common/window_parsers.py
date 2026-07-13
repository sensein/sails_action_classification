from __future__ import annotations


def parse_multiclass(
    resp: str,
    all_classes: list[str],
    active_classes: list[str],
    no_label: str,
) -> str | None:
    if not resp:
        return None

    upper = resp.upper()
    if "ACTION:" in upper:
        after = (
            resp[upper.find("ACTION:") + 7:]
            .strip().split("\n")[0].strip().strip("\"'")
        )
        for cls in all_classes:
            if cls.lower() == after.lower():
                return cls
        for cls in all_classes:
            if cls.replace("_", " ").lower() == after.lower():
                return cls

    for cls in active_classes:
        if cls.lower() in resp.lower():
            return cls

    no_variants = [
        no_label.lower(),
        no_label.replace("_", " ").lower(),
    ]
    if any(v in resp.lower() for v in no_variants):
        return no_label

    return None


def parse_binary(resp: str) -> bool | None:
    if not resp:
        return None

    upper = resp.upper()
    if "ANSWER:" in upper:
        after = upper.split("ANSWER:")[-1].strip().split()[0]
        if after.startswith("YES"):
            return True
        if after.startswith("NO"):
            return False

    stripped = upper.strip().rstrip(".")
    if stripped == "YES":
        return True
    if stripped == "NO":
        return False

    return None


def parse_finegrained(
    resp: str,
    active_classes: list[str],
    task: str,
) -> str | None:
    if not resp:
        return None

    upper = resp.upper()
    if "ACTION:" in upper:
        after = (
            resp[upper.find("ACTION:") + 7:]
            .strip().split("\n")[0].strip().strip("\"'")
        )
        for cls in active_classes:
            if cls.lower() == after.lower():
                return cls
        for cls in active_classes:
            if cls.replace("_", " ").lower() == after.lower():
                return cls
        if task == "rmm" and "flap" in after.lower():
            return "Hands_flapping"

    for cls in active_classes:
        if cls.lower() in resp.lower():
            return cls

    if task == "rmm" and "flap" in resp.lower():
        return "Hands_flapping"

    return None
