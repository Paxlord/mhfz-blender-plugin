# FMOD Importer for Blender

A Blender addon for importing and exporting FMOD model files (.fmod) from **Monster Hunter Frontier Z**.

![Blender Version](https://img.shields.io/badge/Blender-4.4+-orange)
![Python Version](https://img.shields.io/badge/Python-3.7+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Installation

### Method 1: Download from Releases (Recommended)

1. Go to the [Releases page](../../releases)
2. Download the latest `fmod_importer.zip`
3. In Blender: `Edit` → `Preferences` → `Add-ons`
4. Click `Install...` and select the downloaded zip file
5. Enable "FMOD Importer" in the addon list

### Method 2: Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fmod-importer.git
   ```
2. Copy the `src` folder to your Blender addons directory:
   - **Windows**: `%APPDATA%\Blender Foundation\Blender\3.x\scripts\addons\`
   - **macOS**: `~/Library/Application Support/Blender/3.x/scripts/addons/`
   - **Linux**: `~/.config/blender/3.x/scripts/addons/`
3. Rename the `src` folder to `fmod_importer`
4. Enable the addon in Blender preferences

## Usage

### Importing FMOD Models

1. **File** → **Import** → **FMOD (.fmod)**
2. Navigate to your `.fmod` file
3. The plugin will automatically:
   - Load the model geometry
   - Find and load textures from adjacent folders
   - Import skeleton data (`.fskl`) if present
   - Set up materials with proper node trees
   - Create vertex groups for bone weights

### Exporting to FMOD

**Recommended Workflow for Custom Models:**

1. **Import a template**: Import a similar FMOD model from the game (e.g., import an existing Great Sword for a custom Great Sword)
2. **Replace the mesh**: Delete the imported mesh and import/create your custom model
3. **Align your model**: Position and scale your mesh to match the skeleton
4. **Setup the model**:
   - Apply transforms to your mesh object
   - Parent the mesh to the weapon armature
   - Set up bone weights accordingly
   - For multi-material weapons (Long Sword, Gunlance, Hunting Horn, Bow, Light Bowgun, Heavy Bowgun), assign faces to appropriate material slots
   - **Important**: Weapons and armor only support diffuse textures. Check the Shader Editor and remove any normal or specular map connections
   - Use the Setup Tools to automate this process, or do it manually
5. **Export**: **File** → **Export** → **FMOD (.fmod)**

**Export Options:**

- **Include Bone List**: Only enable for monster models, disable for weapons/armor
- **Force Texture Resize**: Set to 128x128 for weapons and armor compatibility
- **Write Log Files**: Only needed when using with the ReFrontier tool

### Using Setup Tools

1. Open the **FMOD Setup** panel by pressing **N** in the 3D Viewport to open the sidebar
2. Select your **target mesh** and **armature**
3. Choose **item type** and **subtype** (currently supports weapons only)
4. Click **Setup FMOD Mesh** for automatic configuration

The tool will:

- Apply all transforms
- Create proper material slots
- Set up vertex groups
- Assign bone weights
- Parent mesh to armature

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/fmod-importer.git
cd fmod-importer

# Build the addon
./build.sh

# This creates fmod_importer.zip ready for installation
```

### Project Structure

```
src/
├── __init__.py              # Main addon registration
├── binary_utils.py          # Binary file I/O utilities
├── data_classes.py          # Data structure definitions
├── file_parsers.py          # FMOD/FSKL file parsers
├── file_writers.py          # Export functionality
├── import_operators.py      # Import operators
├── export_operators.py      # Export operators
├── model_utils.py           # Mesh processing utilities
├── setup_tools.py           # Weapon setup automation
├── sgi_strips.py           # Triangle strip algorithm
└── ui_panels.py            # User interface panels
```

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test in Blender
4. Commit: `git commit -m "Add feature"`
5. Push: `git push origin feature-name`
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Silvris**, for the initial fmod.bt template outline
- **Blaze**, for testing
