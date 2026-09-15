from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'eval'))
from causal_gtp import Engine,inference_contract
from gozero.snapshots import read_json


class CausalInferenceTests(unittest.TestCase):
    def test_context_reserves_every_possible_search_extension(self):
        config=read_json(ROOT/'eval/causal_student/search.json')
        model={'size':9,'komi':7.5,'max_tokens':329}
        inference_contract(config,model)
        inference_contract({**config,'max_game_moves':312},model)
        with self.assertRaisesRegex(ValueError,'context'):
            inference_contract({**config,'max_game_moves':313},model)
        with self.assertRaisesRegex(ValueError,'context'):
            inference_contract({**config,'simulations':73},model)

    def test_cap_prevents_new_search_or_external_move_without_truncating_history(self):
        engine=Engine.__new__(Engine);engine.plies=256;engine.max_game_moves=256
        # No game object exists: rejected commands cannot start/commit anything.
        for command,args in [('genmove',['B']),('play',['B','pass'])]:
            with self.assertRaisesRegex(ValueError,'move cap'):
                engine.command(command,args)
        self.assertEqual(engine.plies,256)


if __name__=='__main__':unittest.main()
