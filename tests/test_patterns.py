import unittest
import pandas as pd
from indicators.patterns import detect_engulfing

class TestPatterns(unittest.TestCase):

    def test_detect_engulfing(self):
        # Test case 1: Bullish Engulfing
        bullish_data = {
            'time': pd.to_datetime(['2023-01-01 10:00', '2023-01-01 10:05']),
            'open':  [1.1000, 1.0990],
            'high':  [1.1010, 1.1030],
            'low':   [1.0995, 1.0985],
            'close': [1.0998, 1.1020]
        }
        bullish_df = pd.DataFrame(bullish_data)
        bullish_result = detect_engulfing(bullish_df)
        self.assertEqual(len(bullish_result), 1)
        self.assertEqual(bullish_result[0]['type'], 'ENGULFING_BULL')

        # Test case 2: Bearish Engulfing
        bearish_data = {
            'time': pd.to_datetime(['2023-01-01 10:10', '2023-01-01 10:15']),
            'open':  [1.1020, 1.1030],
            'high':  [1.1025, 1.1035],
            'low':   [1.1015, 1.1000],
            'close': [1.1022, 1.1010]
        }
        bearish_df = pd.DataFrame(bearish_data)
        bearish_result = detect_engulfing(bearish_df)
        self.assertEqual(len(bearish_result), 1)
        self.assertEqual(bearish_result[0]['type'], 'ENGULFING_BEAR')

        # Test case 3: No Engulfing
        no_engulfing_data = {
            'time': pd.to_datetime(['2023-01-01 10:20', '2023-01-01 10:25']),
            'open':  [1.1000, 1.1010],
            'high':  [1.1010, 1.1020],
            'low':   [1.0990, 1.1000],
            'close': [1.1005, 1.1015]
        }
        no_engulfing_df = pd.DataFrame(no_engulfing_data)
        no_engulfing_result = detect_engulfing(no_engulfing_df)
        self.assertEqual(len(no_engulfing_result), 0)

if __name__ == '__main__':
    unittest.main()
