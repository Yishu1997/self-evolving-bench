#!/usr/bin/env bash
python -m bench.run --base-url "$1" --api-key "$2" --model "$3" --n 20
