"""oracle.skyportal: the SkyPortal annotation/classification shapes (no model, no torch)."""

from oracle import skyportal as S


def _cls(predicted="SN-Ia"):
    return {
        "predicted": predicted,
        "probabilities": {
            "SN-Ia": 0.75, "SN-II": 0.11, "SN-Ib/c": 0.11, "SLSN": 0.01,
            "AGN": 0.01, "CV": 0.005, "Varstar": 0.005,
        },
    }


def test_taxonomy_map_covers_bts_leaves():
    assert set(S.ORACLE_TO_TAXONOMY) == {
        "SN-Ia", "SN-II", "SN-Ib/c", "SLSN", "AGN", "CV", "Varstar"
    }
    assert S.SKYPORTAL_ORIGIN == "ORACLE"
    assert S.SKYPORTAL_TAXONOMY == "Sitewide Taxonomy"


def test_annotations_are_flat_and_carry_full_vector():
    ann = S.annotations_for(_cls())
    assert ann["oracle_class"] == "SN-Ia"
    assert ann["oracle_p_SN-Ia"] == 0.75 and ann["oracle_p_max"] == 0.75
    assert all(not isinstance(v, (dict, list)) for v in ann.values())


def test_skyportal_annotations_shape():
    ann = S.skyportal_annotations(_cls())
    assert isinstance(ann, list) and ann[0]["origin"] == "ORACLE"
    assert ann[0]["data"]["oracle_p_SN-Ia"] == 0.75


def test_skyportal_classification_maps_to_sitewide():
    cl = S.skyportal_classifications(_cls("SN-Ia"))
    assert cl == [
        {"taxonomy": "Sitewide Taxonomy", "classification": "Ia",
         "probability": 0.75, "ml": True, "origin": "ORACLE"}
    ]
    # An unmapped leaf yields no classification (annotations still carry the vector).
    assert S.skyportal_classifications({"predicted": "Junk", "probabilities": {"Junk": 1.0}}) == []
