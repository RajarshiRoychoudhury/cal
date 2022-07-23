import logging
from typing import Any, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def read_pool_data(pool_path: str) -> Tuple[Any, Any]:
    pool_data = pd.read_csv(pool_path)

    pool_data["Description"] = pool_data["Description"].map(lambda x: str(x).replace("\\", ""))

    pool = pool_data["Description"].values.tolist()

    logging.info(f"Number of instances -- pool: {len(pool)}")

    return pool
