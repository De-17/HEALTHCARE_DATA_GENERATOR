# Create the complete project structure for Synthetic Healthcare Data Generator
import os
import pandas as pd

# Define the project structure
project_structure = {
    'synthetic-healthcare-data-generator': {
        'data': {
            'raw': {},
            'processed': {},
            'synthetic': {}
        },
        'models': {
            'gans': {},
            'vaes': {},
            'saved': {}
        },
        'src': {
            'data': {},
            'models': {},
            'evaluation': {},
            'privacy': {},
            'utils': {},
            'visualization': {}
        },
        'notebooks': {},
        'tests': {},
        'configs': {},
        'results': {
            'reports': {},
            'figures': {},
            'metrics': {}
        },
        'docker': {},
        'docs': {}
    }
}

def create_project_structure(base_path, structure, level=0):
    """Recursively create project directory structure"""
    for name, content in structure.items():
        path = os.path.join(base_path, name) if level > 0 else name
        if isinstance(content, dict):
            print(f"{'  ' * level}📁 {name}/")
            if content:  # If dictionary has contents
                create_project_structure(path, content, level + 1)
        else:
            print(f"{'  ' * level}📄 {name}")

print("🏗️  Synthetic Healthcare Data Generator - Project Structure")
print("=" * 60)
create_project_structure('', project_structure)

# Create a sample dataset description
sample_datasets = {
    'UCI Heart Disease': {
        'features': 14,
        'samples': 303,
        'target': 'heart_disease',
        'type': 'Classification'
    },
    'CDC Diabetes': {
        'features': 21, 
        'samples': 253680,
        'target': 'diabetes_binary',
        'type': 'Classification'
    },
    'Synthetic Healthcare': {
        'features': 12,
        'samples': 10000,
        'target': 'admission_outcome',
        'type': 'Classification'
    }
}

print(f"\n📊 Available Healthcare Datasets:")
print("=" * 60)
for dataset, info in sample_datasets.items():
    print(f"• {dataset}")
    for key, value in info.items():
        print(f"  - {key}: {value}")
    print()

print("🚀 Project Components Overview:")
print("=" * 60)
components = [
    "📈 GAN Models: WGAN-GP, CTGAN, medGAN, DP-GAN",
    "🧠 VAE Models: β-VAE, Conditional VAE, WAE", 
    "🔒 Privacy: Differential Privacy, k-anonymity, l-diversity",
    "📏 Evaluation: 15+ utility & privacy metrics",
    "📊 Visualization: Interactive dashboards & reports",
    "🐳 Docker: Containerized deployment",
    "📚 Documentation: Complete API & tutorials"
]

for component in components:
    print(component)