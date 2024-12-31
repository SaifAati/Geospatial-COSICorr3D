# [December 30, 2024] geoCosiCorr 3D v2.5.3 Release Notes
We’re excited to end the year with another release: geoCosiCorr 3D v2.5.3!

This latest version introduces the Time Series Subpackage, along with improvements to enhance user experience. 
Here’s what’s new:
- geoPCAIM :Geospatial Principal Component Analysis-based Inversion Method (geoPCAIM) is a statistically
based approach applied to redundant surface displacement measurements. 
This method filters out measurement noise and extracts signals with maximum spatio-temporal coherence.
- geoICA :Independent Component Analysis (geoICA) is another statistically based approach for processing
redundant surface displacement measurements. It effectively filters out noise and highlights signals with high 
spatio-temporal coherence.


---



# [July 09, 2024] geoCosiCorr 3D v2.5.0 Release Notes

- Dynamic memory management, during orthorectification, to avoid memory overflow.
- Support state extrapolation.
- Code cleanup and refactoring.

---

# [April 15, 2024] geoCosiCorr 3D v2.4.1 Release Notes

- **Cosicorr Command Line Interface (CLI) Improvement**: cosicorr cli support GCP generation,
  RSM model refinement and orthorectification.

---

# [March 3, 2024] geoCosiCorr 3D v2.4.0 Release Notes

We're excited to announce the release of geoCosiCorr 3D v2.4.0! This latest version introduces significant updates and
improvements aimed at enhancing the user experience and expanding the tool's capabilities in satellite image processing.
Here's what's new:

## Updates and New Features

- **Conda Environment with Python 3.9**: We've updated the Python environment to 3.9 to ensure better compatibility and
  performance. This update makes it easier for users to manage dependencies and set up geoCosiCorr 3D in their work
  environment.

- **Docker Support with Python 3.10**: To provide more flexibility and ease of use, geoCosiCorr 3D now comes with Docker
  support, running on Python 3.10. This allows users to deploy and use geoCosiCorr 3D in containerized environments,
  simplifying the installation process and ensuring consistency across different platforms.

- **Improved Installation Process**: We've streamlined the installation process to make it quicker and more
  straightforward. This improvement reduces setup time and helps new users get started with geoCosiCorr 3D more
  efficiently.

- **Continuous Integration via GitHub Actions (CI-GHA)**: The integration of Continuous Integration practices through
  GitHub Actions enhances the development process, ensuring that each update is thoroughly tested and stable before
  release.

- **Cosicorr Command Line Interface (CLI) Implementation**: The new Cosicorr CLI feature allows users to execute
  geoCosiCorr 3D tasks directly from the command line, providing a more flexible and scriptable interface for advanced
  users.

- **Enhanced Elevation Extraction Handling**: The process of elevation extraction has been improved for better accuracy
  and efficiency. This enhancement allows for more precise 3D surface modeling and analysis.

---




