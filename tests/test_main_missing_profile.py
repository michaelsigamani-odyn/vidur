import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vidur.main import main


class MainErrorHandlingTest(unittest.TestCase):
    def test_main_exits_with_missing_profile_message(self):
        config = SimpleNamespace(seed=0)
        with patch("vidur.main.SimulationConfig.create_from_cli_args", return_value=config):
            with patch("vidur.main.set_seeds"):
                with patch(
                    "vidur.main.Simulator",
                    side_effect=FileNotFoundError("No profiling data for mi300x"),
                ):
                    with self.assertRaises(SystemExit) as context:
                        main()
        self.assertIn("No profiling data for mi300x", str(context.exception))


if __name__ == "__main__":
    unittest.main()
