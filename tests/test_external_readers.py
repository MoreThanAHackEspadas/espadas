"""External readers on tiny synthetic files."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from espadas.external import degrade, metrica_epts  # noqa: E402

META = """<main><Metadata>
<Sessions><Session><FieldSize><Width>100</Width><Height>50</Height></FieldSize></Session></Sessions>
<Teams><Team id="TA"/><Team id="TB"/></Teams>
<Players><Player id="a1" teamId="TA"/><Player id="b1" teamId="TB"/><Player id="a2" teamId="TA"/></Players>
</Metadata>
<DataFormatSpecifications>
<DataFormatSpecification startFrame="1" endFrame="2">
 <SplitRegister><SplitRegister><PlayerChannelRef playerChannelId="c1_x"/><PlayerChannelRef playerChannelId="c1_y"/></SplitRegister>
 <SplitRegister><PlayerChannelRef playerChannelId="c2_x"/><PlayerChannelRef playerChannelId="c2_y"/></SplitRegister></SplitRegister>
</DataFormatSpecification>
<DataFormatSpecification startFrame="3" endFrame="3">
 <SplitRegister><SplitRegister><PlayerChannelRef playerChannelId="c3_x"/><PlayerChannelRef playerChannelId="c3_y"/></SplitRegister>
 <SplitRegister><PlayerChannelRef playerChannelId="c2_x"/><PlayerChannelRef playerChannelId="c2_y"/></SplitRegister></SplitRegister>
</DataFormatSpecification>
</DataFormatSpecifications>
<PlayerChannels>
<PlayerChannel id="c1_x" playerId="a1" channelId="x"/><PlayerChannel id="c1_y" playerId="a1" channelId="y"/>
<PlayerChannel id="c2_x" playerId="b1" channelId="x"/><PlayerChannel id="c2_y" playerId="b1" channelId="y"/>
<PlayerChannel id="c3_x" playerId="a2" channelId="x"/><PlayerChannel id="c3_y" playerId="a2" channelId="y"/>
</PlayerChannels></main>"""


def test_epts_substitution_moves_group_to_new_slot(tmp_path):
    meta_path = tmp_path / "m.xml"
    meta_path.write_text(META)
    raw = tmp_path / "t.txt"
    raw.write_text("1:0.1,0.2;0.5,0.5:0.3,0.4\n2:NaN,NaN;0.6,0.6:NaN,NaN\n3:0.9,0.8;0.7,0.7:0.5,0.5\n")
    meta = metrica_epts.read_metadata(str(meta_path))
    assert meta["team_of_slot"].tolist() == [0, 1, 0] and meta["pitch"] == (100.0, 50.0)
    players, ball = metrica_epts.read_tracking(str(raw), meta)
    assert players[0, 0].tolist() == [10.0, 10.0]            # a1 in slot 0
    assert np.isnan(players[2, 0]).all()                     # a1 subbed off
    assert players[2, 2].tolist() == [90.0, 40.0]            # a2 took group 0
    assert players[2, 1].tolist() == [70.0, 35.0]            # b1 unchanged
    assert ball[0].tolist() == [30.0, 20.0] and np.isnan(ball[1]).all()


def test_hide_ball_reaches_target_deterministically():
    n = 2000
    ball = np.zeros((n, 2))
    ball[::10] = np.nan                                       # 10% missing already
    players = np.zeros((n, 4, 2))
    graded = np.ones(n, dtype=bool)
    out1, hid1 = degrade.hide_ball(ball, players, graded, 0.41)
    out2, hid2 = degrade.hide_ball(ball, players, graded, 0.41)
    assert np.array_equal(hid1, hid2)
    assert abs(np.isnan(out1[:, 0]).mean() - 0.41) < 0.002
    assert not hid1[::10].any()                               # only hides present frames
