uv run pmarl train --config ./config/koth.json
uv run pmarl eval --run $(ls ./results/ -Art | tail -n 1) --out ./analysis/$(ls ./results/ -Art | tail -n 1)
