import unittest
from src.main import main

class TestMain(unittest.TestCase):
    def test_plot(self):
        self.assertIsNotNone(main())

if __name__ == '__main__':
    unittest.main()