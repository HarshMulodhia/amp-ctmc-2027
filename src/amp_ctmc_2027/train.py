from __future__ import annotations

import inspect
import logging
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.pipeline.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    """CLI entry point for Phase-3 training."""
    config = AMPConfig(
        batch_size=16,#256
        learning_rate=3e-4,
        n_epochs=1,#200
        early_stopping_patience=40,
        dropout=0.1,
        log_masking_diagnostics=True,
        debug_overfit_samples=None,  # set to 64 to test whether the model can overfit
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logger.info("AMPConfig imported from: %s", inspect.getfile(AMPConfig))
    logger.info("TrainingPipeline imported from: %s", inspect.getfile(TrainingPipeline))
    logger.info("Resolved training config: %s", config.model_dump())
    pipeline = TrainingPipeline(config=config, fasta_repo=FastaRepository(Path.cwd()))
    pipeline.run()


if __name__ == "__main__":
    main()