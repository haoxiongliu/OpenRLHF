HINT_DICT = {
    'smt': r"smt",
    'hint': r"hint",
    'my_hint': r"try norm_num [*]; try field_simp [*] at *; try ring_nf at *; try nlinarith",
    "my_hint_v0.1": r"try field_simp [*] at *; try norm_num [*]; try nlinarith",
    'aesop': r"aesop",
    'aesop_v0.1': r"try aesop; try norm_num [*]",
    'omega': r"omega",
    'nlinarith': r"nlinarith",
    "linarith": r"linarith",
    'ring_nf': r"ring_nf",
    'simp_all': r"simp_all",
    "norm_num": r"norm_num",
    "field_simp": r"field_simp [*] at *",
    "bound": r"bound",
    "leanhammer": r"hammer",
    "leanhammer_0": r"hammer {aesopPremises := 0, autoPremises := 0}",
    "leanhammer_1": r"hammer {aesopPremises := 1, autoPremises := 1}",
    "leanhammer_2": r"hammer {aesopPremises := 2, autoPremises := 2}",
    "leanhammer_3": r"hammer {aesopPremises := 3, autoPremises := 3}",
    "leanhammer_4": r"hammer {aesopPremises := 4, autoPremises := 4}",
    "leanhammer_5": r"hammer {aesopPremises := 5, autoPremises := 5}",
    None: None,
    "": None
}

RECIPE2HAMMER_LIST = {
    "mix": ["bound", "nlinarith", "simp_all", "field_simp", "omega", "my_hint", "aesop"],
    "mix2": ["aesop", "my_hint", "omega"],
}
