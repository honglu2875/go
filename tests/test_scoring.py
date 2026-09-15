import json
from pathlib import Path
import unittest
from gozero.scoring import direct_area


class ScoringTests(unittest.TestCase):
    def test_area_neutral_regions_and_retained_stones(self):
        self.assertEqual(direct_area('...\n...\n...',0.5),0.5)
        self.assertEqual(direct_area('...\n.X.\n...',0.5),-8.5)
        self.assertEqual(direct_area('X.O\n...\n...',0.5),0.5)
        with self.assertRaises(ValueError):direct_area('...\n.X.',0.5)

    def test_actual_katago_terminal_board_uses_declared_raw_scoring(self):
        fixture=json.loads((Path(__file__).resolve().parents[1]/'eval/fixtures/pass_alive_area.json').read_text())
        board=fixture['board']
        self.assertEqual(direct_area(board,fixture['komi']),fixture['raw_white_minus_black'])
        self.assertNotEqual(fixture['raw_white_minus_black'],fixture['katago_adjudicated_white_minus_black'])
        rows=board.splitlines()
        for _ in range(4):
            rows=[''.join(x) for x in zip(*rows[::-1])]
            self.assertEqual(direct_area('\n'.join(rows),fixture['komi']),fixture['raw_white_minus_black'])
            self.assertEqual(direct_area('\n'.join(x[::-1] for x in rows),fixture['komi']),fixture['raw_white_minus_black'])
