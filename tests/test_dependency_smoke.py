"""Import and tiny I/O checks for declared runtime dependency groups."""

import importlib

import pyarrow as pa
import pyarrow.parquet as pq


def test_esm_multitask_predictor_imports():
    module = importlib.import_module("amp_ctmc_2027.models.esm_multitask")
    assert module.ESMMultiTaskPredictor is not None
    transformers = importlib.import_module("transformers")
    assert transformers.AutoTokenizer is not None
    assert transformers.EsmModel is not None


def test_parquet_read_write(tmp_path):
    path = tmp_path / "smoke.parquet"
    pq.write_table(
        pa.Table.from_pylist([{"sequence": "ACDEFGHI", "split": "train"}]), path
    )
    assert pq.read_table(path).to_pylist() == [
        {"sequence": "ACDEFGHI", "split": "train"}
    ]


def test_property_training_pipeline_imports():
    module = importlib.import_module(
        "amp_ctmc_2027.pipeline.property_training_pipeline"
    )
    assert module.PropertyTrainingPipeline is not None


def test_ctmc_training_pipeline_imports():
    module = importlib.import_module("amp_ctmc_2027.pipeline.training_pipeline")
    assert module.TrainingPipeline is not None


def test_generation_entry_point_imports():
    module = importlib.import_module("amp_ctmc_2027.generate")
    assert callable(module.generate_broad_spectrum)
