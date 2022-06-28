import datetime
import os
import glob
import json
import shutil
import time
from typing import Optional
from os.path import join as pjoin

from bert4rec.utils import get_logger
from bert4rec.trainer import BERT4RecTrainer
from bert4rec.data import DatasetBuilder


class TrainManager:
    PARENT_DATA_DIR = pjoin(os.path.dirname(__file__), "data")
    PARENT_MODEL_DIR = pjoin(os.path.dirname(__file__), "model")

    def __init__(self, conf_fname: str, finetune_conf_fname: str=""):
        self._logger = get_logger()
        self._finetune_conf_fname = finetune_conf_fname
        self._set_configure(conf_fname)
    
    def _set_configure(self, conf_fname: str):
        with open(conf_fname, "r") as fin:
            self._opt = json.load(fin)
        self._is_finetune = self._opt.get("finetune") is not None
        timestamp_fname = self._opt["data"]["preprocess"].get("fname_timestamp", None)
        self._use_timestamp = True if timestamp_fname else False
        self._meta_save_dir = {}

    def _download_meta_data(self):
        for meta_id, meta_info in self._opt["data"].get("meta", {}).items():
            _, save_dir = self._download_data_from_modelhouse(
                meta_id,
                meta_info["svc"],
                meta_info["data"]
            )
            self._meta_save_dir[meta_id] = save_dir

    def _download_stream_data(self):
        return self._download_data_from_modelhouse(
            "raw",
            self._opt["data"]["download"]["svc"],
            self._opt["data"]["download"]["data"]
        )

    def _download_pretrained_model(self):
        return self._download_data_from_modelhouse(
            "pretrained_model",
            self._opt["finetune"]["svc"],
            self._opt["finetune"]["model_name"]
        )

    def _download_data_from_modelhouse(self, title: str, svc: str, data: str):
        # svc_name / model_name for this BERT4Rec model.
        svc_name = self._opt["info"]["svc"]
        model_name = self._opt["info"]["model_name"]
        save_dir = pjoin(self.PARENT_DATA_DIR, svc_name, model_name, title)
        os.makedirs(save_dir, exist_ok=True)
        downloaded = True
        return downloaded, save_dir

    def _preprocess_data(self, stream_data_dir: str, pretrained_model_dir: Optional[str]):
        self._logger.info("Preprocessing data...")
        svc_name = self._opt["info"]["svc"]
        model_name = self._opt["info"]["model_name"]

        save_dir = pjoin(self.PARENT_DATA_DIR, svc_name, model_name, "preprocessed")
        os.makedirs(save_dir, exist_ok=True)
        preprocess_opt = self._opt["data"]["preprocess"]
        model_opt = self._opt["model"]
        stream_timestamp_data_fname = pjoin(stream_data_dir, preprocess_opt["fname_timestamp"]) if self._use_timestamp else ""
        pretrained_key_fname = pjoin(pretrained_model_dir, "keys") if pretrained_model_dir is not None else ""
        pretrained_features = preprocess_opt.get("pretrained_features", [])
        for pretrained_feature in pretrained_features:
            pretrained_feature["dir_name"] = self._meta_save_dir[pretrained_feature["key"]]

        builder = DatasetBuilder(
            stream_data_fname=pjoin(stream_data_dir, preprocess_opt["fname"]),
            stream_timestamp_data_fname=stream_timestamp_data_fname,
            save_dir=save_dir,
            frequency_min_count=preprocess_opt["frequency_min_count"],
            consume_min_count=preprocess_opt["consume_min_count"],
            sequence_length=model_opt["sequence_length"],
            max_samples_per_stream=preprocess_opt.get("max_samples_per_stream", 0),
            pretrained_key_fname=pretrained_key_fname,
            mr_impl=preprocess_opt.get("mr_impl", "c"),
            num_workers=self._opt["trainer"].get("num_workers", 8),
            pretrained_features=pretrained_features
        )
        builder.build()
        return save_dir

    def _train(self, data_dir: str, pretrained_model_dir: Optional[str]):
        svc_name = self._opt["info"]["svc"]
        model_name = self._opt["info"]["model_name"]

        trainer_opt = self._opt["trainer"]
        model_opt = self._opt["model"]
        if pretrained_model_dir is not None:
            pretrained_model_file_fname = pjoin(pretrained_model_dir, "model.pt")
            pretrained_model_opt_fname = pjoin(pretrained_model_dir, "model_opt.json")
        else:
            pretrained_model_file_fname = None
            pretrained_model_opt_fname = None

        self._logger.info("Training...")
        current_timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        self.trainer = BERT4RecTrainer(**trainer_opt, model_name=model_name, current_timestamp=current_timestamp)
        self.trainer.fit(data_dir, **model_opt, 
                    pretrained_model_file_fname=pretrained_model_file_fname, 
                    pretrained_model_opt_fname=pretrained_model_opt_fname,
                    use_timestamp=self._use_timestamp)
        self._logger.info("Training finished!")

        save_dir = pjoin(self.PARENT_MODEL_DIR, svc_name, model_name, current_timestamp)
        os.makedirs(save_dir, exist_ok=True)
        self.trainer.save(save_dir)
        self._logger.info(f"Saved model to {save_dir}")
        return save_dir, current_timestamp
    
    def _inferene(self, data_dir: str, pretrained_model_dir: Optional[str]):
        svc_name = self._opt["info"]["svc"]
        model_name = self._opt["info"]["model_name"]

        trainer_opt = self._opt["trainer"]
        model_opt = self._opt["model"]
        if pretrained_model_dir is not None:
            pretrained_model_file_fname = pjoin(pretrained_model_dir, "model.pt")
            pretrained_model_opt_fname = pjoin(pretrained_model_dir, "model_opt.json")
        else:
            pretrained_model_file_fname = None
            pretrained_model_opt_fname = None

        self._logger.info("Training...")
        current_timestamp = str(int(datetime.datetime.now().timestamp() * 1000))
        self.trainer = BERT4RecTrainer(**trainer_opt, model_name=model_name, current_timestamp=current_timestamp)
        self.trainer.inference(data_dir, **model_opt, 
                    pretrained_model_file_fname=pretrained_model_file_fname, 
                    pretrained_model_opt_fname=pretrained_model_opt_fname,
                    use_timestamp=self._use_timestamp)
        
        return True
    
    def run_all_once(self):
        _, stream_data_dir = self._download_stream_data()
        self._download_meta_data()
        data_dir = self._preprocess_data(stream_data_dir, None)
        pretrained_model_dir, current_timestamp = self._train(data_dir, None)

        self._set_configure(self._finetune_conf_fname)
        _, stream_data_dir = self._download_stream_data()
        self._download_meta_data()
        data_dir = self._preprocess_data(stream_data_dir, pretrained_model_dir)

        for i in range(0, 1):
            model_dir, current_timestamp = self._train(data_dir, pretrained_model_dir)

            mrr = float(self.trainer._calculate_mrr())
            
            if self._opt["trainer"]["is_submit"]:

                self.trainer._make_leaderboard(self.trainer._leaderboard_loader, "./data/submit_leaderboard_csv", "leader")
                self.trainer._make_leaderboard(self.trainer._final_loader, "./data/submit_final_csv", "final")

            else:
                self.trainer._make_leaderboard(self.trainer._val_experiment_loader, "./data/val_csv", "val")
                self.trainer._make_leaderboard(self.trainer._test_loader, "./data/te_csv", "test")

            print("Done")


    def run_100(self):
        i = 0
        while i < 100:
            self.run_all_once()
            i += 1
            time.sleep(10.0)

    def run_forever(self):
        i = 0
        while i < 100:
            self.run_once()
            time.sleep(60.0)


if __name__ == "__main__":
    from fire import Fire
    Fire(TrainManager)
