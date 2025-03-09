import unittest
from src.data_preprocessing.load_data import load_dataset
from src.data_preprocessing.preprocess import preprocess_dataset

class TestMain(unittest.TestCase):

    def test_load_dataset(self):
        dataloader = load_dataset(batch_size=32)
        for i, batch in enumerate(dataloader):
            self.assertEqual(batch['images'].shape, (32, 3, 224, 224))
            self.assertEqual(batch['findings'].shape[0], 32)
            self.assertEqual(batch['boxes/bbox'].shape[0], 32)
            self.assertEqual(batch['boxes/finding'].shape[0], 32)
            for field in batch.keys():
                if field.startswith("metadata"):
                    self.assertEqual(batch[field].shape[0], 32)
            break  # Only test the first batch

    def test_preprocess_dataset(self):
        dataloader = load_dataset(batch_size=32)
        processed_data = preprocess_dataset(dataloader)
        self.assertIsInstance(processed_data, list)
        self.assertGreater(len(processed_data), 0)
        sample = processed_data[0]
        self.assertIn('images', sample)
        self.assertIn('findings', sample)
        self.assertIn('boxes/bbox', sample)
        self.assertIn('boxes/finding', sample)

if __name__ == '__main__':
    unittest.main()