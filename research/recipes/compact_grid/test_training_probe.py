import unittest
from training_probe import select,overfit_observation


class ProbeTests(unittest.TestCase):
    def test_population_order_does_not_change_sample(self):
        population={128:[(0,i) for i in range(80)],256:[(1,i) for i in range(15)],384:[(2,i) for i in range(5)]}
        a=select(population,20)
        b=select({k:list(reversed(v)) for k,v in reversed(list(population.items()))},20)
        self.assertEqual(a,b)
        self.assertEqual({k:len(v) for k,v in a.items()},{128:16,256:3,384:1})
        self.assertEqual(sum(map(len,select(population,1000).values())),100)

    def test_sustained_divergence_but_not_one_noisy_check(self):
        def curve(values):return [dict(turn=i,metrics=dict(expert_kl=v)) for i,v in enumerate(values)]
        train=curve([.8,.5,.48,.46,.44])
        self.assertTrue(overfit_observation(curve([.9,.6,.62,.63,.64]),train)['sustained'])
        self.assertFalse(overfit_observation(curve([.9,.6,.59,.58,.59]),train)['sustained'])
        self.assertFalse(overfit_observation(curve([.9,.6,.62,.63,.64]),curve([.8,.5,.5,.5,.5]))['sustained'])


if __name__=='__main__':unittest.main()
