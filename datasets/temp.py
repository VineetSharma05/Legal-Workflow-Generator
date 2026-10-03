import json
from collections import Counter

COMBINED_PATH = "./combined_dataset.json"
SOFTWARE_LICENSING_PATH = "./softwarelicensing.json"

with open(COMBINED_PATH, "r") as f:
    combined = json.load(f)

with open(SOFTWARE_LICENSING_PATH, "r") as f:
    sl = json.load(f)

print("combined (before):", len(combined))
print("softwarelicensing:", len(sl))

# combined_dataset.json may already contain pre-existing duplicate
# provision_ids from other domains (e.g. dpdp_act_2023). That is not
# something this merge should fix - only softwarelicensing laws are in
# scope here - so remember them to exclude from the post-merge check.
pre_existing_dupe_ids = {
    pid for pid, count in Counter(p["provision_id"] for p in combined).items()
    if count > 1
}

# Index existing combined provisions by provision_id so we can detect
# duplicates and their positions without touching entries from other
# law domains (e.g. companies_act_2013, dpdp_act_2023, posh_act_2013, ...).
combined_index = {p["provision_id"]: i for i, p in enumerate(combined)}

added = 0
updated = 0

for provision in sl:
    pid = provision["provision_id"]
    if pid in combined_index:
        # Already present (e.g. copyright_act_1957 / information_technology_act_2000
        # provisions shared with other domains) - replace with the updated
        # softwarelicensing.json version instead of keeping a duplicate.
        idx = combined_index[pid]
        if combined[idx] != provision:
            combined[idx] = provision
            updated += 1
    else:
        combined.append(provision)
        combined_index[pid] = len(combined) - 1
        added += 1

# Sanity check: the merge itself must not introduce any new duplicate
# provision_ids (pre-existing ones from other domains are left as-is).
post_merge_dupe_ids = {
    pid for pid, count in Counter(p["provision_id"] for p in combined).items()
    if count > 1
}
new_dupe_ids = post_merge_dupe_ids - pre_existing_dupe_ids
assert not new_dupe_ids, f"merge introduced duplicate provision_id(s): {new_dupe_ids}"

print("added:", added)
print("updated:", updated)
print("combined (after):", len(combined))

with open(COMBINED_PATH, "w") as f:
    json.dump(combined, f, indent=2)
    f.write("\n")
