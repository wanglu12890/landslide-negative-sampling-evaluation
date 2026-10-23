"""Uniform independent sampling of unique low-IV cells."""
import rasterio
from rasterio.windows import Window
from skills.ns1_random_sampling.src.random_sampling import sample_negative_cells


def sample_negatives(run, count, pool_size, threshold, cfg):
    """Use NS1's deterministic candidate-ordinal sampler, with NS2 provenance."""
    negative = sample_negative_cells(run / "spatial/low_susceptibility_mask.tif", pool_size, count,
        cfg["sampling"]["random_seed"], cfg["processing"]["chunk_rows"])
    negative["negative_id"] = [f"NS2_N{i+1:04d}" for i in range(count)]
    negative["strategy"] = "NS2_IV_LowSusceptibility"
    negative["iv_quantile_threshold"] = threshold
    with rasterio.open(run / "spatial/IV_total.tif") as iv:
        negative["iv_total"] = [float(iv.read(1, window=Window(int(p.col), int(p.row), 1, 1))[0, 0]) for p in negative.itertuples()]
    return negative
