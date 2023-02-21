# from distutils.core import setup
from setuptools import setup
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()
    
print(long_description)
    
setup(
  name = 'resype',         # How you named your package folder (MyLib)
  packages = ['resype'],   # Chose the same as "name"
  version = '0.1',     
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'recommender system using machine learning',
  long_description=long_description,
  long_description_content_type = 'text/markdown',
  author = 'Prince Javier',                   # Type in your name
  author_email = 'othepjavier@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/phdinds-aim/resype',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/phdinds-aim/resype/archive/refs/tags/v_001.tar.gz',
  keywords = ['machine learning', 'recommender system'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'pandas',
          'matplotlib',
          'seaborn',
          'scipy',
          'scikit-learn',
          'scikit-surprise',
          'nltk',
          'xlrd',
          'parquet'
 
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable"
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
