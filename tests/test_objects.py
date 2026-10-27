"""Checks the extracted object table against hitboxes known from the game."""
import pytest

from gdgen.objects import load_objects


@pytest.fixture(scope="module")
def objects():
    return load_objects()


@pytest.mark.parametrize(
    "object_id, category, hitbox",
    [
        (1, "solid", (30, 30)),          # default block
        (8, "hazard", (6, 12)),          # spike
        (39, "hazard", (6, 5.6)),        # half spike
        (103, "hazard", (4, 7.6)),       # small spike
        (35, "pad", (25, 4)),            # yellow pad
        (36, "orb", (36, 36)),           # yellow orb
        (12, "portal_mode", (34, 86)),   # cube portal
        (10, "portal_gravity", (25, 75)),
        (99, "portal_size", (31, 90)),
    ],
)
def test_known_hitboxes(objects, object_id, category, hitbox):
    info = objects[object_id]
    assert info.category == category
    assert info.hitbox == pytest.approx(hitbox, abs=0.01)


def test_saw_is_circular_hazard(objects):
    saw = objects[88]
    assert saw.category == "hazard" and saw.radius == pytest.approx(32.3)


@pytest.mark.parametrize("object_id, radius", [(89, 21.6), (98, 11.925), (1706, 21.6)])
def test_saw_radius_derived_from_sprite_width(objects, object_id, radius):
    # customSetup computes these radii from m_width, which GameObject::init sets from the sprite.
    assert objects[object_id].radius == pytest.approx(radius, abs=1e-3)


@pytest.mark.parametrize("object_id", [919, 3610, 3611])
def test_hazards_that_look_like_decoration_or_triggers(objects, object_id):
    assert objects[object_id].category == "hazard"


@pytest.mark.parametrize("object_id, category", [(200, "speed"), (901, "trigger"), (1007, "trigger"),
                                                 (31, "start_pos"), (13, "portal_mode")])
def test_categories(objects, object_id, category):
    assert objects[object_id].category == category
