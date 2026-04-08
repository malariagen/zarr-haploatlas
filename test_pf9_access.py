"""Quick test: can we connect to the Pf9 GCS bucket?"""
import malariagen_data

print("Connecting to Pf9...")
try:
    pf9 = malariagen_data.Pf9()
    print(f"OK — Pf9 object created: {pf9}")
except Exception as e:
    print(f"FAILED to create Pf9 object: {e}")
    raise SystemExit(1)

print("Fetching variant calls (reads zarr metadata from GCS)...")
try:
    vc = pf9.variant_calls()
    print(f"OK — variant_calls shape: {dict(vc.sizes)}")
except Exception as e:
    print(f"FAILED to fetch variant_calls: {e}")
    raise SystemExit(1)

print("Reading first 5 variant positions (triggers actual GCS read)...")
try:
    pos = vc["variant_position"].isel(variants=slice(0, 5)).values
    print(f"OK — first 5 positions: {pos}")
except Exception as e:
    print(f"FAILED to read variant data: {e}")
    raise SystemExit(1)

print("\nAll checks passed — Pf9 bucket is accessible.")
