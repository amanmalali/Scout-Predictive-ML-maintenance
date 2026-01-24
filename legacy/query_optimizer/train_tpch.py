import os
import tarfile
import time
from pathlib import Path
from typing import cast

import lightning.pytorch as pl
import pandas as pd
import pytorch_warmup as warmup
import requests
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import WeightedMeanAbsolutePercentageError
from sklearn.model_selection import train_test_split


from udao.data.extractors import PredicateEmbeddingExtractor, QueryStructureExtractor
from udao.data.extractors.tabular_extractor import TabularFeatureExtractor
from udao.data.handler.data_handler import DataHandler
from udao.data.handler.data_processor import FeaturePipeline, create_data_processor
from udao.data.iterators.query_plan_iterator import QueryPlanIterator
from udao.data.predicate_embedders import Word2VecEmbedder, Word2VecParams
from udao.data.preprocessors.normalize_preprocessor import NormalizePreprocessor
from udao.model.embedders.graph_averager import GraphAverager
from udao.model.model import FixedEmbeddingUdaoModel, UdaoModel
from udao.model.module import LearningParams, UdaoModule
from udao.model.regressors.mlp import MLP
from udao.model.utils.losses import WMAPELoss
from udao.model.utils.schedulers import UdaoLRScheduler, setup_cosine_annealing_lr
from udao.optimization import concepts
from udao.optimization.moo.progressive_frontier import SequentialProgressiveFrontier
from udao.optimization.soo.mogd import MOGD
from udao.utils.interfaces import UdaoEmbedInput
from udao.utils.logging import logger

logger.setLevel("INFO")


def transform_data(df,train_perc=0.7,latency_threshold=50):    
    df_data=df.loc[(df['latency']<latency_threshold)]
    df_train=df_data.sample(frac = train_perc)
    df_val_short=df_data.drop(df_train.index)



    df_val_long=df.loc[(df['latency'] >= latency_threshold)]
    df_train_long=df_val_long.sample(200)
    df_val_long_true=df_val_long.drop(df_train_long.index)

    df_val=pd.concat([df_val_long_true,df_val_short])

    df_train=pd.concat([df_train,df_train_long])
    
    return df_train,df_val


def download_data() -> None:
    base_dir = Path(__file__).parent / "data"

    if os.path.exists(base_dir / "TPCH/brief.csv"):
        logger.info("Data already downloaded")
        return
    logger.info("Downloading data")
    response = requests.get("https://www.lix.polytechnique.fr/~lyu/dataset/TPCH.tar.gz")
    with open(base_dir / "TPCH.tar.gz", "wb") as f:
        f.write(response.content)
    with tarfile.open(base_dir / "TPCH.tar.gz", "r:gz") as tar:
        tar.extractall(base_dir)
    os.remove(base_dir / "TPCH.tar.gz")


def train_base_model():
    tensor_dtypes = th.float32
    device = "cpu"
    batch_size = 512
    # download_data()

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    #### Data definition ####
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(
                columns=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
                + ["s1", "s2", "s3", "s4"]
                + ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["latency", "cost"]),
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
                NormalizePreprocessor(MinMaxScaler(), "graph_meta_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=8))
            ),
        ),
    )

    base_dir = Path(__file__).parent
    lqp_df = pd.read_csv(str(base_dir / "data/TPCH/LQP.csv"))
    brief_df = pd.read_csv(str(base_dir / "data/TPCH/brief.csv"))
    cols_to_use = lqp_df.columns.difference(brief_df.columns)

    df = brief_df.merge(
        lqp_df[["id", *cols_to_use]],
        on="id",
    )

    aws_cost_cpu_hour_ratio = 0.052624
    aws_cost_mem_hour_ratio = 0.0057785  # for GB*H

    def get_cloud_cost(lat: float, mem: int, cores: int, nexec: int) -> float:
        cpu_hour = (nexec + 1) * cores * lat / 3600
        mem_hour = (nexec + 1) * mem * lat / 3600
        cost = cpu_hour * aws_cost_cpu_hour_ratio + mem_hour * aws_cost_mem_hour_ratio
        return cost

    df["cost"] = df.apply(  # type: ignore
        lambda row: get_cloud_cost(row["latency"], row["k1"] * 2, row["k2"], row["k3"]),
        axis=1,
    )
    
    df_train,df_val=transform_data(df)

    df_train, df_test = train_test_split(df_train, test_size=0.2)
    df_train.to_csv('./data/train.csv')
    df_test.to_csv('./data/test.csv')
    df_val.to_csv('./data/val.csv')
    # df_val.to_csv('./val.csv')
    print("Training length :",len(df_train))
    print("Validation length :",len(df_val))
    df.to_csv('./data/full_data.csv')

    data_handler = DataHandler(
        df,
        DataHandler.Params(
            index_column="id",
            stratify_on=None,#"tid",
            dryrun=False,#True,
            data_processor=data_processor,
        ), 
    ) 


    split_iterators=data_handler.get_iterators_file(df_train,df_test,df_test) #REMEMBER


    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": 1024,
            "op_groups": ["cbo", "op_enc", "type"],
            "type_embedding_dim": 8,
            "embedding_normalizer": None,
        },
        regressor_params={"n_layers": 2, "hidden_dim": 32, "dropout": 0.1},
    )




    # print(split_iterators["train"].shape)
    try:
        module = UdaoModule.load_from_checkpoint(
            "checkpoints/1-val_WMAPE=0.98.ckpt",
            model=model,
            objectives=["latency", "cost"],
        )
        logger.info("Found checkpointed model!")
    except BaseException:
        logger.info("model not found from checkpoints!")
        module = UdaoModule(
            model,
            ["latency","cost"],
            loss=WMAPELoss(),
            learning_params=LearningParams(
                init_lr=1e-2, min_lr=1e-3, weight_decay=1e-1
            ),
            metrics=[WeightedMeanAbsolutePercentageError],
        )
        tb_logger = TensorBoardLogger("tb_logs")
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename="{epoch}-val_WMAPE={val_latency_WeightedMeanAbsolutePercentageError:.2f}",
            auto_insert_metric_name=False,
            monitor="val_latency_WeightedMeanAbsolutePercentageError",  # Metric to monitor for saving the best model
            mode="min",  # Choose 'min' or 'max' depending on the metric
            save_top_k=4, 
        )
        # train_iterator = cast(QueryPlanIterator, split_iterators["train"])
        # split_iterators["train"].set_augmentations(
        #    [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
        # )
        scheduler = UdaoLRScheduler(
            setup_cosine_annealing_lr, warmup.UntunedLinearWarmup
        )
        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=100,
            logger=tb_logger,
            callbacks=[scheduler, checkpoint_callback],
        )
        trainer.fit(
            model=module,
            train_dataloaders=split_iterators["train"].get_dataloader(batch_size),
            val_dataloaders=split_iterators["test"].get_dataloader(batch_size),
        )
        logger.info("model trained and checkpointed.")
        th.save(model, './base_model.pt')



def retrain_model(new_train_df,dir_name="checkpoints_retrain/"):

    tensor_dtypes = th.float32
    device = "cpu"
    batch_size = 512
    

    th.set_default_dtype(tensor_dtypes)  # type: ignore
    #### Data definition ####
    processor_getter = create_data_processor(QueryPlanIterator, "op_enc")
    data_processor = processor_getter(
        tensor_dtypes=tensor_dtypes,
        tabular_features=FeaturePipeline(
            extractor=TabularFeatureExtractor(
                columns=["k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8"]
                + ["s1", "s2", "s3", "s4"]
                + ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8"],
            ),
            preprocessors=[NormalizePreprocessor(MinMaxScaler())],
        ),
        objectives=FeaturePipeline(
            extractor=TabularFeatureExtractor(["latency", "cost"]),
        ),
        query_structure=FeaturePipeline(
            extractor=QueryStructureExtractor(positional_encoding_size=10),
            preprocessors=[
                NormalizePreprocessor(MinMaxScaler(), "graph_features"),
                NormalizePreprocessor(MinMaxScaler(), "graph_meta_features"),
            ],
        ),
        op_enc=FeaturePipeline(
            extractor=PredicateEmbeddingExtractor(
                Word2VecEmbedder(Word2VecParams(vec_size=8))
            ),
        ),
    )

    df_train, df_test = train_test_split(new_train_df, test_size=0.2)
    # df_val.to_csv('./val.csv')
    # df_train=pd.read_csv('./data/train.csv')
    # df_test=pd.read_csv('./data/test.csv')
    print("Training length :",len(df_train))
    df=pd.read_csv('./data/full_data.csv')

    data_handler = DataHandler(
        df,
        DataHandler.Params(
            index_column="id",
            stratify_on=None,#"tid",
            dryrun=False,#True,
            data_processor=data_processor,
        ), 
    ) 




    split_iterators=data_handler.get_iterators_file(df_train,df_test,df_test) #REMEMBER


    model = UdaoModel.from_config(
        embedder_cls=GraphAverager,
        regressor_cls=MLP,
        iterator_shape=split_iterators["train"].shape,
        embedder_params={
            "output_size": 1024,
            "op_groups": ["cbo", "op_enc", "type"],
            "type_embedding_dim": 8,
            "embedding_normalizer": None,
        },
        regressor_params={"n_layers": 2, "hidden_dim": 32, "dropout": 0.1},
    )




    # print(split_iterators["train"].shape)

    module = UdaoModule(
        model,
        ["latency","cost"],
        loss=WMAPELoss(),
        learning_params=LearningParams(
            init_lr=1e-2, min_lr=1e-3, weight_decay=1e-1
        ),
        metrics=[WeightedMeanAbsolutePercentageError],
    )
    tb_logger = TensorBoardLogger("tb_logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_name,
        filename="{epoch}-val_WMAPE={val_latency_WeightedMeanAbsolutePercentageError:.4f}",
        auto_insert_metric_name=False,
        monitor="val_latency_WeightedMeanAbsolutePercentageError",  # Metric to monitor for saving the best model
        mode="min",  # Choose 'min' or 'max' depending on the metric
        save_top_k=4, 
    )
    # train_iterator = cast(QueryPlanIterator, split_iterators["train"])
    # split_iterators["train"].set_augmentations(
    #    [train_iterator.make_graph_augmentation(random_flip_positional_encoding)]
    # )
    scheduler = UdaoLRScheduler(
        setup_cosine_annealing_lr, warmup.UntunedLinearWarmup
    )
    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=100,
        logger=tb_logger,
        callbacks=[scheduler, checkpoint_callback],
    )
    trainer.fit(
        model=module,
        train_dataloaders=split_iterators["train"].get_dataloader(batch_size),
        val_dataloaders=split_iterators["test"].get_dataloader(batch_size),
    )
    logger.info("model retrained and checkpointed.")


    return df_train,df_test,checkpoint_callback.best_model_path,model



# if __name__ == "__main__":

#     train_base_model()
    


    # data_handler = DataHandler(
    #     df,
    #     DataHandler.Params(
    #         index_column="id",
    #         stratify_on="tid",
    #         dryrun=True,
    #         data_processor=data_processor,
    #     ),
    # )

    # split_iterators = data_handler.get_iterators()
    #### Model definition and training ####
    # df.to_csv("./trial.csv")