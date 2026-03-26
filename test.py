import time
import dask
import malariagen_data

pf8 = malariagen_data.Pf8()
ds = pf8.variant_calls()

# Pick a single variant by slicing a small region
single = ds.isel(variants=slice(1000000, 1000001))

print(f"Single variant dataset: {dict(single.sizes)}")
print(f"GT chunk structure for this slice: {single['call_genotype'].data.chunks}")
print()

for n_threads in [128, 64, 32, 16]:
    opts = {"scheduler": "threads", "num_workers": n_threads}
    # warm up
    dask.compute(single["call_genotype"].data, single["call_AD"].data, **opts)

    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        dask.compute(single["call_genotype"].data, single["call_AD"].data, **opts)
        times.append(time.perf_counter() - t0)

        print(times, flush=True)
    print(f"threads={n_threads:3d}  min={min(times):.2f}s  mean={sum(times)/len(times):.2f}s", flush=True)