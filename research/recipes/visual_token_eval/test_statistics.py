import unittest
import numpy as np
from paired_statistics import compare

class PairedChecks(unittest.TestCase):
    def test_identical_models_and_uniform_per_position_improvement(self):
        left=[{'entry':['expert',0,i],'count':n,'ce':2*n,'target_entropy':n/2,'top1':n/2} for i,n in enumerate((2,10,100))]
        zero=compare(left,left,repetitions=31)
        self.assertEqual(zero['right_minus_left_kl'],0.)
        self.assertEqual(zero['kl_cluster_bootstrap_percentile_95'],[0.,0.])
        right=[{**x,'ce':x['ce']-.25*x['count']} for x in left]
        got=compare(left,right,repetitions=31)
        self.assertEqual(got['right_minus_left_kl'],-.25)
        np.testing.assert_array_equal(got['kl_cluster_bootstrap_percentile_95'],[-.25,-.25])
        self.assertEqual(got['fraction_games_lower_kl_on_right'],1.)

    def test_order_and_lengths_must_match(self):
        left=[{'entry':['expert',0,i],'count':n,'ce':n,'target_entropy':0,'top1':0} for i,n in enumerate((2,10))]
        with self.assertRaises(ValueError):compare(left,list(reversed(left)))
        with self.assertRaises(ValueError):compare(left,[left[0],{**left[1],'count':11}])

if __name__=='__main__':unittest.main()
