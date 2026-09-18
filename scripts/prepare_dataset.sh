#!/usr/bin/env bash
set -ETeuo pipefail
SRC=$(realpath "$(dirname "$0")/..")

APP="${1:?Must specify app to prepare dataset for}"

cd "$SRC"

if [[ "$APP" == "crc" ]] || [[ "$APP" == "all" ]]; then
	echo
	echo "==> Preparing CRC datasets"
    # TODO: This is broken in CI due to missing CL in docker images
	make -C combinational-logic/crc clean
	make -C combinational-logic/crc datasets
fi

if [[ "$APP" == "cfd" ]] || [[ "$APP" == "all" ]]; then
	echo
	echo "==> Preparing CFD datasets"
	[[ -f test/unstructured-grids/cfd/128.dat ]] ||
		python3 scripts/generate_cfd_dataset.py \
			test/unstructured-grids/cfd/fvcorr.domn.193K \
			test/unstructured-grids/cfd/128.dat \
			128
	[[ -f test/unstructured-grids/cfd/1284.dat ]] ||
		python3 scripts/generate_cfd_dataset.py \
			test/unstructured-grids/cfd/fvcorr.domn.193K \
			test/unstructured-grids/cfd/1284.dat \
			1284
	[[ -f test/unstructured-grids/cfd/45056.dat ]] ||
		python3 scripts/generate_cfd_dataset.py \
			test/unstructured-grids/cfd/fvcorr.domn.193K \
			test/unstructured-grids/cfd/45056.dat \
			45056
	[[ -f test/unstructured-grids/cfd/193474.dat ]] ||
		python3 scripts/generate_cfd_dataset.py \
			test/unstructured-grids/cfd/fvcorr.domn.193K \
			test/unstructured-grids/cfd/193474.dat \
			193474
fi

if [[ "$APP" == "swat" ]] || [[ "$APP" == "all" ]]; then
	echo
	echo "==> Preparing SWAT datasets"
	[[ -f test/dynamic-programming/swat/sampledb-tiny.data &&
		-f test/dynamic-programming/swat/sampledb-tiny.loc &&
		-f test/dynamic-programming/swat/sampledb-small.data &&
		-f test/dynamic-programming/swat/sampledb-small.loc &&
		-f test/dynamic-programming/swat/sampledb-medium.data &&
		-f test/dynamic-programming/swat/sampledb-medium.loc &&
		-f test/dynamic-programming/swat/sampledb-large.data &&
		-f test/dynamic-programming/swat/sampledb-large.loc ]] ||
		python3 scripts/generate_swat_dataset.py \
			test/dynamic-programming/swat/sampledb1K1 \
			test/dynamic-programming/swat
fi

if [[ "$APP" == "tdm" ]] || [[ "$APP" == "all" ]]; then
	echo
	echo "==> Preparing TDM datasets"
	[[ -f test/finite-state-machine/tdm/sim-64-size200-tiny.csv &&
		-f test/finite-state-machine/tdm/episodes-tiny.txt &&
		-f test/finite-state-machine/tdm/sim-64-size200-small.csv &&
		-f test/finite-state-machine/tdm/episodes-small.txt &&
		-f test/finite-state-machine/tdm/sim-64-size200-medium.csv &&
		-f test/finite-state-machine/tdm/episodes-medium.txt &&
		-f test/finite-state-machine/tdm/sim-64-size200-large.csv &&
		-f test/finite-state-machine/tdm/episodes-large.txt ]] ||
		python3 scripts/generate_tdm_dataset.py \
			test/finite-state-machine/tdm/sim-64-size200.csv \
			test/finite-state-machine/tdm/30-episodes.txt \
			--output-dir test/finite-state-machine/tdm \
			--prefix sim-64-size200 \
			--episodes-prefix episodes
fi
