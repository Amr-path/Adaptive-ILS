"""
Map File Parsers
=================

Parsers for standard benchmark map formats.

Supported formats:
- Moving AI Lab (.map files)
- Simple text-based maps
- PNG/Image-based maps
"""

import os
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import sys
sys.path.insert(0, '..')
from core.data_structures import Grid, Point


class MovingAIParser:
    """
    Parser for Moving AI Lab benchmark maps.

    Moving AI Lab format:
    - First line: "type octile"
    - Second line: "height N"
    - Third line: "width M"
    - Fourth line: "map"
    - Following lines: Map data

    Characters:
    - '.': Passable terrain
    - 'G': Passable terrain (grass)
    - '@': Out of bounds / obstacle
    - 'O': Out of bounds
    - 'T': Trees (obstacle)
    - 'S': Swamp (passable, slower)
    - 'W': Water (obstacle)
    """

    PASSABLE_CHARS = {'.', 'G', 'S'}
    OBSTACLE_CHARS = {'@', 'O', 'T', 'W'}

    def __init__(self):
        self.map_info: Dict = {}

    def parse(self, filepath: str) -> Grid:
        """
        Parse a Moving AI Lab map file.

        Args:
            filepath: Path to the .map file

        Returns:
            Parsed Grid object
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Parse header
        header_lines = 0
        width, height = 0, 0

        for line in lines:
            line = line.strip()
            if line.startswith('type'):
                self.map_info['type'] = line.split()[1] if len(line.split()) > 1 else 'octile'
                header_lines += 1
            elif line.startswith('height'):
                height = int(line.split()[1])
                header_lines += 1
            elif line.startswith('width'):
                width = int(line.split()[1])
                header_lines += 1
            elif line == 'map':
                header_lines += 1
                break

        if width == 0 or height == 0:
            raise ValueError(f"Invalid map file: could not parse dimensions from {filepath}")

        # Create grid
        grid = Grid(width, height, eight_connected=True)

        # Parse map data
        map_lines = lines[header_lines:]

        for y, line in enumerate(map_lines):
            if y >= height:
                break
            line = line.rstrip('\n\r')

            for x, char in enumerate(line):
                if x >= width:
                    break

                if char in self.OBSTACLE_CHARS:
                    grid.set_obstacle(x, y, True)
                elif char in self.PASSABLE_CHARS:
                    grid.set_obstacle(x, y, False)
                else:
                    # Unknown character - treat as obstacle for safety
                    grid.set_obstacle(x, y, True)

        self.map_info['width'] = width
        self.map_info['height'] = height
        self.map_info['filepath'] = filepath
        self.map_info['filename'] = os.path.basename(filepath)

        return grid

    def parse_scenario(self, scenario_path: str) -> List[Dict]:
        """
        Parse a Moving AI Lab scenario file.

        Scenario files contain test cases with start/goal positions
        and optimal path lengths.

        Args:
            scenario_path: Path to .scen file

        Returns:
            List of scenario dictionaries
        """
        scenarios = []

        with open(scenario_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:  # Skip header
            parts = line.strip().split()
            if len(parts) >= 9:
                scenario = {
                    'bucket': int(parts[0]),
                    'map': parts[1],
                    'width': int(parts[2]),
                    'height': int(parts[3]),
                    'start_x': int(parts[4]),
                    'start_y': int(parts[5]),
                    'goal_x': int(parts[6]),
                    'goal_y': int(parts[7]),
                    'optimal_length': float(parts[8]),
                }
                scenarios.append(scenario)

        return scenarios


class SimpleTextParser:
    """
    Parser for simple text-based maps.

    Format:
    - Each line represents a row
    - '#' or '1': Obstacle
    - '.' or '0' or ' ': Free space
    """

    def parse(self, filepath: str) -> Grid:
        """
        Parse a simple text map file.

        Args:
            filepath: Path to the map file

        Returns:
            Parsed Grid object
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Determine dimensions
        height = len(lines)
        width = max(len(line.rstrip('\n\r')) for line in lines) if lines else 0

        if width == 0 or height == 0:
            raise ValueError(f"Empty or invalid map file: {filepath}")

        # Create grid
        grid = Grid(width, height, eight_connected=True)

        # Parse map
        for y, line in enumerate(lines):
            line = line.rstrip('\n\r')
            for x, char in enumerate(line):
                if char in {'#', '1', '@', 'X'}:
                    grid.set_obstacle(x, y, True)
                # Other characters are passable

        return grid


class ImageParser:
    """
    Parser for image-based maps.

    Black pixels are obstacles, white/gray pixels are passable.
    Requires PIL/Pillow.
    """

    def __init__(self, threshold: int = 128):
        """
        Initialize parser.

        Args:
            threshold: Grayscale threshold for obstacles (0-255)
        """
        self.threshold = threshold

    def parse(self, filepath: str) -> Grid:
        """
        Parse an image map file.

        Args:
            filepath: Path to image file (PNG, JPG, BMP, etc.)

        Returns:
            Parsed Grid object
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL/Pillow required for image parsing. Install with: pip install Pillow")

        img = Image.open(filepath).convert('L')  # Convert to grayscale
        width, height = img.size

        grid = Grid(width, height, eight_connected=True)

        for y in range(height):
            for x in range(width):
                pixel = img.getpixel((x, y))
                if pixel < self.threshold:  # Dark = obstacle
                    grid.set_obstacle(x, y, True)

        return grid


def parse_map_file(filepath: str, format: Optional[str] = None) -> Grid:
    """
    Parse a map file with automatic format detection.

    Args:
        filepath: Path to map file
        format: Optional format hint ("movingai", "text", "image")

    Returns:
        Parsed Grid object
    """
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # Auto-detect format
    if format is None:
        if ext == '.map':
            format = 'movingai'
        elif ext in {'.txt', '.grid'}:
            format = 'text'
        elif ext in {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}:
            format = 'image'
        else:
            # Try to detect from content
            with open(filepath, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('type'):
                    format = 'movingai'
                else:
                    format = 'text'

    # Parse with appropriate parser
    if format == 'movingai':
        return MovingAIParser().parse(filepath)
    elif format == 'text':
        return SimpleTextParser().parse(filepath)
    elif format == 'image':
        return ImageParser().parse(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def list_benchmark_maps(directory: str, recursive: bool = True) -> List[str]:
    """
    List all map files in a directory.

    Args:
        directory: Directory to search
        recursive: Search subdirectories

    Returns:
        List of map file paths
    """
    extensions = {'.map', '.txt', '.grid', '.png', '.jpg', '.bmp'}
    maps = []

    if recursive:
        for root, dirs, files in os.walk(directory):
            for f in files:
                if os.path.splitext(f)[1].lower() in extensions:
                    maps.append(os.path.join(root, f))
    else:
        for f in os.listdir(directory):
            if os.path.splitext(f)[1].lower() in extensions:
                maps.append(os.path.join(directory, f))

    return sorted(maps)


def download_moving_ai_maps(output_dir: str, map_set: str = "dao") -> List[str]:
    """
    Download Moving AI Lab benchmark maps.

    Note: This is a placeholder. In practice, maps should be
    downloaded manually from https://movingai.com/benchmarks/

    Args:
        output_dir: Directory to save maps
        map_set: Map set to download ("dao", "sc1", "random", etc.)

    Returns:
        List of downloaded map paths
    """
    # Map sets available at movingai.com
    map_sets = {
        "dao": "Dragon Age: Origins maps",
        "da2": "Dragon Age 2 maps",
        "sc1": "StarCraft 1 maps",
        "wc3": "Warcraft 3 maps",
        "random": "Random obstacle maps",
        "room": "Room maps",
        "maze": "Maze maps",
    }

    print(f"Note: Moving AI maps should be downloaded manually from:")
    print(f"https://movingai.com/benchmarks/grids.html")
    print(f"\nAvailable map sets: {list(map_sets.keys())}")
    print(f"\nSelected: {map_set} - {map_sets.get(map_set, 'Unknown')}")

    # Return empty list - user needs to download manually
    return []
