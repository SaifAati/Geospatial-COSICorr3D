from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='geoCosiCorr3D',
        version='2.5.3',
        author='Saif Aati',
        author_email='saif@caltech.edu, saifaati@gmail.com',
        description='Geospatial COSI-Corr 3D',
        long_description=open('README.md').read() + '\n\n' + open('NEWS.md').read(),
        long_description_content_type='text/markdown',
        url='https://github.com/SaifAati/geoCosiCorr3D.git',
        platforms=['unix', 'linux', 'win64', 'osx'],
        classifiers=[
            'Programming Language :: Python :: 3.8',
        ],
        license='GNU General Public License v3 (GPLv3)',
        license_files=('LICENSE'),
        packages=find_packages(include=['geoCosiCorr3D', 'geoCosiCorr3D.*']),
        python_requires='>=3.8',
        zip_safe=False,
        extras_require={
            'testing': [
                'pytest>=6.0',
                'pytest-cov>=2.0',
                'mypy>=0.910',
                'flake8>=3.9',
                'tox>=3.24',
            ],
        },
        package_data={
            'geoCosiCorr3D': ['py.typed'],
        },
    )
