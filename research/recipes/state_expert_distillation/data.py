"""Expert-only sampling from the unchanged, verified board/history dataset."""
from gozero.board_sequence_batches import Dataset as BoardDataset


class Dataset(BoardDataset):
    def sample(self, random, games_per_host):
        indices = self.indices['expert', 0]
        return self.batch([('expert', *indices[int(i)]) for i in random.integers(len(indices), size=games_per_host)])

    def evaluation_entries(self, split):
        if split not in (1, 2):
            raise ValueError('Evaluation requires a held-out split')
        return [('expert', *index) for index in self.indices['expert', split]]
