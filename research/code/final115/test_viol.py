import os, sys, numpy as np
from pathlib import Path
W = Path(os.environ.get("DC_WORK") or Path.home() / "dc_work")
sys.path.insert(0, str(W / r"final_v7_prod\src"))
from detector_classifier.lanes import Decoder, ANCHOR
rng = np.random.default_rng(0)
bad = 0
for it in range(3000):
    n = int(rng.integers(1, 10)); L = int(rng.integers(1, 5))
    func = list(rng.choice(["Advance", "Presence", "Count", "Yellow_Red", "Other", "Mid"], n))
    anc = np.array([f in ANCHOR for f in func])
    if not anc.any() or rng.random() < 0.1:
        anc[:] = True
    role = np.array([func[i] if func[i] in ANCHOR else "" for i in range(n)], object)
    S = rng.integers(1, 1 << L, size=(int(rng.integers(1, 20)), n))
    a = np.array([Decoder._viol(list(map(int, r)), L, anc, role) for r in S], float)
    b = Decoder._viol_batch(S, L, anc, role)
    bad += int((a != b).any())
print("mismatching batches:", bad)
