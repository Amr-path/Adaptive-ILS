# Moving AI Lab Benchmark Maps

These benchmark maps are from the Moving AI Lab MAPF Benchmarks.

## Source
- Website: https://movingai.com/benchmarks/grids.html
- Citation: Sturtevant, N. R. (2012). Benchmarks for grid-based pathfinding.
  IEEE Transactions on Computational Intelligence and AI in Games, 4(2), 144-148.

## Maps Included

### den312d.map
- **Source**: Dragon Age: Origins
- **Dimensions**: 65 × 81
- **Format**: Octile grid

### ht_chantry.map
- **Source**: Dragon Age 2
- **Dimensions**: 141 × 162
- **Format**: Octile grid

### random-64-64-20.map
- **Source**: Randomly generated
- **Dimensions**: 64 × 64
- **Obstacle Density**: 20%
- **Format**: Octile grid

## Map Format

```
type octile
height [H]
width [W]
map
[grid data]
```

### Cell Types
- `.` - Traversable terrain (ground)
- `G` - Traversable terrain (ground)
- `S` - Traversable terrain (swamp, same cost as ground in this work)
- `T` - Obstacle (tree/wall)
- `@` - Obstacle (out of bounds)
- `O` - Obstacle (out of bounds)
- `W` - Obstacle (water, treated as impassable)

## Usage

These maps are used for:
1. Comparing AILS with standard A* baseline
2. Validating AILS on real-world game map structures
3. Enabling direct comparison with Lee & Lee 2025 results
