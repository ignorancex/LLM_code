# -*- coding: utf-8 -*-
import pathlib
import shutil

import smefit.runner

runcard_folder = commondata_path = pathlib.Path(__file__).parents[0]


class TestRunner:
    test_runner = smefit.runner.Runner.from_file(runcard_folder / "fake_runcard.yaml")

    def test_init(self):
        assert self.test_runner.run_card["datasets"][0] == "test_dataset"
        assert self.test_runner.run_card["result_ID"] == "fake_runcard"

    def test_setup_result_folder(self):
        result_folder = pathlib.Path(self.test_runner.run_card["result_path"])
        res_folder_fit = result_folder / self.test_runner.run_card["result_ID"]
        assert res_folder_fit.exists()
        shutil.rmtree(res_folder_fit)
