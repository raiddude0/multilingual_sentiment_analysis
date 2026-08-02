from setuptools import setup, find_packages

setup(
    name="multilingual_sentiment_analysis",
    version="0.1.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "transformers>=4.46,<6",
        "datasets>=2.19,<4",
        "scikit-learn>=1.4,<2",
        "matplotlib>=3.8,<4",
        "seaborn>=0.13,<1",
        "pandas>=2.2,<3",
        "torch>=2.2,<3",
        "accelerate>=0.30,<2",
        "gradio>=4.0,<7",
    ],
)