import os
import numpy as np
import pandas as pd
import unittest
import zipfile
from coerce import coerce_column, STRING_TO_DATE, STRING_TO_NUMBER, FLOAT_TO_INT

class TestCoerce(unittest.TestCase):

    def setUp(self):
        """Set up tests."""
        pd.options.display.max_columns = 999
        dataZip = zipfile.ZipFile("globalterrorismdb_0718dist.csv.zip")
        self.df = pd.read_csv(dataZip.open("globalterrorismdb_0718dist.csv"), encoding = "ISO-8859-1")

    def test_string_to_number(self):
        """Test string_to_number method."""
        df = coerce_column(self.df, "nkillus", STRING_TO_NUMBER)
        self.assertEqual("float64", df["nkillus"].dtype)

    def test_float_to_int(self):
        """Test float_to_int method."""
        df = coerce_column(self.df, "nwound", FLOAT_TO_INT)
        self.assertEqual("int32", df["nwound"].dtype)

    def test_string_to_date(self):
        """Test string to date method."""
        df = coerce_column(self.df, "iyear", STRING_TO_DATE)
        self.assertEqual("datetime64[ns]", df["iyear"].dtype)

if __name__ == "__main__":
    unittest.main()
