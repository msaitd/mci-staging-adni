"""Compact AD-signature atrophy measures (FreeSurfer 7, ICV-normalised volumes, thickness in mm)."""
L, R = "Left", "Right"
COMPACT = ["AS_HIPP", "AS_AMYG", "AS_ENT", "AS_SIG", "AS_VENT"]


def _thk(reg, side):
    return f"fs_thk_ThicknessAverageaparcstatsof{side}{reg}"


def add_compact(X):
    X = X.copy()
    sv = "fs_vol_SubcorticalVolumeasegstatsof"
    X["AS_HIPP"] = X[f"{sv}{L}Hippocampus"] + X[f"{sv}{R}Hippocampus"]
    X["AS_AMYG"] = X[f"{sv}{L}Amygdala"] + X[f"{sv}{R}Amygdala"]
    X["AS_ENT"] = (X[_thk("Entorhinal", L)] + X[_thk("Entorhinal", R)]) / 2
    sig = [_thk(reg, s) for reg in ("Entorhinal", "InferiorTemporal", "MiddleTemporal", "Fusiform") for s in (L, R)]
    X["AS_SIG"] = X[sig].mean(axis=1, skipna=False)
    X["AS_VENT"] = sum(X[f"{sv}{s}{v}"] for s in (L, R) for v in ("LateralVentricle", "InferiorLateralVentricle"))
    return X
