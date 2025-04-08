from setuptools import setup, find_packages

setup(
    name="ml_agent",
    version="0.1.0",
    package_dir={"ml_agent": "ml_agent"},
    install_requires=[
        "langchain==0.3.21",
        "langchain-core==0.3.49",
        "langchain-groq==0.3.1",
        "langchain-community==0.3.20",
        "langgraph==0.3.20",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "scikit-learn>=1.2.0",
        "pydantic>=2.0.0",
        "typing-extensions>=4.5.0",
        "category_encoders>=2.6.0",
        "langchain-experimental>=0.0.49"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=5.0.0",
            "black>=23.0.0",
        ]
    },
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A package for machine learning agents",
    long_description=open("README.md").read(),  # Create a README.md file
    long_description_content_type="text/markdown",
    # url="https://github.com/yourusername/ml_agent",  # Replace with your repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose an appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
