from setuptools import setup, find_packages


setup(
    name='perennityai-mp-extract',
    version='0.1.0',  # Developmental release (devN)
    author='Perennity AI',
    author_email='info@perennityai.com',
    description = "A tool for extracting gesture landmarks and metadata from videos and saving them to disk using MediaPipe.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/perennityai/perennityai-mp-extract',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.8',
    install_requires=[
        'tensorflow>=2.5.0,<=2.17.0', 
        "tensorboard>=2.5.0,<=2.17.0",
        "keras",
        'matplotlib',
        "matplotlib",
        "opencv-python",
        "json5 ",
        "pandas",
        "mediapipe",
        "yt_dlp"
    ],
    entry_points={
        'console_scripts': [
            'perennityai-mp-extract = perennity_mp_extract.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        # Include any config or other necessary data files in the package
        '': ['configs/*.ini'],
    },
    license='MIT',
    keywords='mediapipe hand landmarks face landmarks pose landmarks extraction PerennityAI',
)